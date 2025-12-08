"""
Otimiza pesos (cost, emissions, reliability, social) usando metaheuristicas de vizinhanca inspiradas em:
- I2PLS (Iterated Two-Phase Local Search)
- MTS (Multi-Neighborhood Tabu Search)
- WILB (Weighted Iterated Local Branching)
- ILS (Iterative Local Search com criterio SA)
- LBH (Local Branching Heuristic)

Objetivo: minimizar f(W) = alpha*CR_proxy + beta*(1 - rho), onde:
- CR_proxy = |sum(pesos) - 1|
- rho = correlacao de Spearman entre o ranking Fuzzy-TOPSIS obtido e o baseline (pesos da literatura)

Saidas:
- best_<algo>.json (melhor peso e metricas)
- runs_<algo>.csv (todas as execucoes, 10 casas decimais)

Uso:
python optimize_weights_neighborhood.py --csv dados_preprocessados/reopt_ALL_blocks_v3_8.csv --out-dir neighborhood_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apply_mcdm_profiles import compute_profile_results
from build_ahp_structure import DATA_PATH, aggregate_metrics_by_alternative, parse_diesel_map


BASE_WEIGHTS = {"cost": 0.40, "emissions": 0.30, "reliability": 0.20, "social": 0.10}
OBJECTIVE_KEYS = ["entropy", "merec", "lopcow", "critic", "mean", "bayes"]


def cr_proxy(weights: Dict[str, float]) -> float:
    return abs(sum(weights.values()) - 1.0)


def normalize_weights(w: np.ndarray) -> Dict[str, float]:
    w = np.clip(w, 0, None)
    total = w.sum()
    if total == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / total
    keys = ["cost", "emissions", "reliability", "social"]
    return dict(zip(keys, w.tolist()))


def spearman_corr(rank_a: pd.Series, rank_b: pd.Series) -> float:
    a = rank_a.reindex(rank_b.index).astype(float).values
    b = rank_b.astype(float).values
    n = len(a)
    if n < 2:
        return 0.0
    diff = a - b
    num = 6 * np.sum(diff**2)
    denom = n * (n**2 - 1)
    return 1 - num / denom if denom else 0.0


def detect_directions_simple(columns: List[str]) -> Dict[str, str]:
    directions: Dict[str, str] = {}
    for col in columns:
        low = col.lower()
        if any(k in low for k in ["cost", "lcoe", "emission", "fossil", "tlcc"]):
            directions[col] = "min"
        elif "diesel_cost_share" in low:
            directions[col] = "min"
        elif "percent_load_target" in low:
            directions[col] = "max"
        else:
            directions[col] = "max"
    return directions


def compute_merec_weights(metrics_df: pd.DataFrame) -> Dict[str, float]:
    data = metrics_df.to_numpy(dtype=float)
    m, n = data.shape
    if m == 0 or n == 0:
        return normalize_weights(np.ones(4))
    directions = detect_directions_simple(metrics_df.columns.tolist())
    norm = np.zeros_like(data, dtype=float)
    for j, col in enumerate(metrics_df.columns):
        col_data = data[:, j]
        if directions.get(col, "max") == "max":
            minv = np.nanmin(col_data)
            denom = np.nanmax(col_data) - minv
            denom = denom if denom != 0 else 1.0
            norm[:, j] = (col_data - minv) / denom
        else:
            maxv = np.nanmax(col_data)
            denom = maxv - np.nanmin(col_data)
            denom = denom if denom != 0 else 1.0
            norm[:, j] = (maxv - col_data) / denom
    norm = np.clip(norm, 1e-12, None)
    S = np.log(1 + np.sum(np.abs(np.log(norm)), axis=1))
    E = np.zeros(n)
    for j in range(n):
        temp = np.delete(norm, j, axis=1)
        S_prime = np.log(1 + np.sum(np.abs(np.log(temp)), axis=1))
        E[j] = np.sum(np.abs(S - S_prime))
    weights = E / np.clip(E.sum(), 1e-12, None)
    return normalize_weights(weights)


def compute_lopcow_weights(metrics_df: pd.DataFrame) -> Dict[str, float]:
    data = metrics_df.to_numpy(dtype=float)
    m, n = data.shape
    if m == 0 or n == 0:
        return normalize_weights(np.ones(4))
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = (col_max - col_min)
    denom[denom == 0] = 1.0
    norm = (data - col_min) / denom
    std = np.std(norm, axis=0, ddof=0)
    pv = std * np.log1p(denom)
    weights = pv / np.clip(pv.sum(), 1e-12, None)
    return normalize_weights(weights)


def compute_mean_weights(metrics_df: pd.DataFrame) -> Dict[str, float]:
    n = metrics_df.shape[1]
    if n == 0:
        return normalize_weights(np.ones(4))
    w = np.ones(n) / n
    return normalize_weights(w)


def compute_bayes_weights(base_w: Dict[str, float], obj_w: Dict[str, float]) -> Dict[str, float]:
    keys = list(base_w.keys())
    base_vec = np.array([base_w[k] for k in keys], dtype=float)
    obj_vec = np.array([obj_w.get(k, 0.0) for k in keys], dtype=float)
    combined = base_vec * obj_vec
    if combined.sum() == 0:
        combined = base_vec
    combined = combined / combined.sum()
    return dict(zip(keys, combined.tolist()))


def random_weights(rng: np.random.Generator) -> Dict[str, float]:
    return normalize_weights(rng.random(4))


def perturb(weights: Dict[str, float], rng: np.random.Generator, step: float, dims: List[int] | None = None) -> Dict[str, float]:
    w_vec = np.array(list(weights.values()))
    if dims:
        mask = np.zeros_like(w_vec)
        mask[dims] = 1
        noise = rng.normal(0, step, size=w_vec.shape) * mask
    else:
        noise = rng.normal(0, step, size=w_vec.shape)
    return normalize_weights(w_vec + noise)


def compute_entropy_weights(metrics_df: pd.DataFrame) -> Dict[str, float]:
    data = metrics_df.to_numpy(dtype=float)
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = (col_max - col_min)
    denom[denom == 0] = 1.0
    norm = (data - col_min) / denom
    prob = norm / np.clip(norm.sum(axis=0, keepdims=True), 1e-12, None)
    prob[prob <= 0] = 1e-12
    m = data.shape[0]
    k = 1.0 / np.log(m) if m > 1 else 0.0
    entropy = -k * np.sum(prob * np.log(prob), axis=0)
    dj = 1 - entropy
    weights = dj / np.clip(dj.sum(), 1e-12, None)
    return normalize_weights(weights)


def compute_critic_weights(metrics_df: pd.DataFrame) -> Dict[str, float]:
    data = metrics_df.to_numpy(dtype=float)
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = (col_max - col_min)
    denom[denom == 0] = 1.0
    norm = (data - col_min) / denom
    std = np.std(norm, axis=0, ddof=0)
    corr = np.corrcoef(norm, rowvar=False)
    if np.isnan(corr).any():
        corr = np.nan_to_num(corr, nan=0.0)
    c_info = std * (1 - np.mean(corr, axis=0))
    weights = c_info / np.clip(c_info.sum(), 1e-12, None)
    return normalize_weights(weights)


def evaluate_multi(
    weights: Dict[str, float],
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    fuzziness: float,
    vikor_v: float,
    weight_refs: Dict[str, Dict[str, float]],
    objective_key: str,
) -> Dict[str, float]:
    cr = cr_proxy(weights)
    res = compute_profile_results(metrics_df, profile_name="opt", profile_weights=weights, fuzziness=fuzziness, vikor_v=vikor_v)
    rhos = []
    for key, col in [
        ("topsis", "fuzzy_topsis_rank"),
        ("vikor", "vikor_rank"),
        ("copras", "copras_rank"),
        ("moora", "moora_rank"),
    ]:
        rhos.append(spearman_corr(res[col], baseline_ranks[key]))
    rho_mean = float(np.mean(rhos))

    deltas = []
    for key, col in [
        ("topsis", "fuzzy_topsis_score"),
        ("vikor", "vikor_score"),
        ("copras", "copras_score"),
        ("moora", "moora_score"),
    ]:
        base = baseline_scores[key]
        cur = res[col]
        deltas.append(np.mean(np.abs(cur - base) / (np.abs(base) + 1e-9)))
    score_delta = float(np.mean(deltas))

    diffs = {}
    for name, ref in weight_refs.items():
        diffs[f"{name}_diff"] = float(np.mean(np.abs(np.array(list(weights.values())) - np.array(list(ref.values())))))

    if objective_key not in weight_refs:
        raise ValueError(f"Objetivo desconhecido: {objective_key}")
    objective = diffs[f"{objective_key}_diff"]
    return {
        "objective": objective,
        "cr": float(cr),
        "rho_mean": rho_mean,
        "score_delta": score_delta,
        **diffs,
        "res": res,
    }


def eval_objective(
    weights: Dict[str, float],
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    objective_key: str,
) -> Dict[str, float]:
    return evaluate_multi(weights, metrics_df, baseline_ranks, baseline_scores, fuzziness, vikor_v, weight_refs, objective_key=objective_key)


def run_i2pls(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    rng: np.random.Generator,
    config: Dict,
    objective_key: str,
) -> Tuple[Dict[str, float], List[float]]:
    """Iterated Two-Phase Local Search: VND+TS (explore) e perturbação por frequência (escape)."""
    current = random_weights(rng)
    best = current
    best_eval = eval_objective(best, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
    best_obj = best_eval["objective"]
    history = [best_obj]
    freq = np.zeros(4)
    stall = 0
    stall_limit = config.get("stall_limit", 10)
    max_iters = config.get("max_iters", 80)
    steps_explore = config.get("steps_explore", [0.03, 0.08])
    step_tabu = config.get("step_tabu", 0.05)
    escape_step = config.get("escape_step", 0.15)

    tabu: List[Tuple[float, ...]] = []
    tabu_size = config.get("tabu_size", 12)

    for _ in range(max_iters):
        # Fase Explore: VND (duas vizinhancas) + TS simples
        improved = False
        for step in steps_explore:
            for _ in range(6):
                dims = [int(rng.integers(0, 4))]
                cand = perturb(current, rng, step, dims=dims)
                cand_eval = eval_objective(cand, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
                obj = cand_eval["objective"]
                history.append(obj)
                freq[dims[0]] += 1
                if obj < best_obj:
                    best, best_obj = cand, obj
                    current = cand
                    improved = True
                    stall = 0
                    break
            if improved:
                break

        # TS sobre vizinhanca diferente (swap/drop simulada)
        key_cur = tuple(round(v, 6) for v in current.values())
        if key_cur not in tabu:
            tabu.append(key_cur)
        if len(tabu) > tabu_size:
            tabu.pop(0)
            cand = perturb(current, rng, step_tabu)
            key = tuple(round(v, 6) for v in cand.values())
            tries = 0
            while key in tabu and tries < 5:
                cand = perturb(current, rng, step_tabu)
                key = tuple(round(v, 6) for v in cand.values())
                tries += 1
            cand_eval = eval_objective(cand, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
            obj = cand_eval["objective"]
            history.append(obj)
            if obj < best_obj:
                best, best_obj = cand, obj
                current = cand
                improved = True
            stall = 0
        else:
            stall += 1
            current = cand if obj <= best_obj else current

        # Fase Escape por frequencia
        if stall >= stall_limit:
            least_moved = np.argsort(freq)[:2].tolist()
            current = perturb(best, rng, escape_step, dims=least_moved)
            stall = 0

    return best, history


def run_mts(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    rng: np.random.Generator,
    config: Dict,
    objective_key: str,
) -> Tuple[Dict[str, float], List[float]]:
    """Multi-Neighborhood Tabu Search com quatro passos de vizinhanca restritos."""
    current = random_weights(rng)
    best = current
    best_eval = eval_objective(best, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
    best_obj = best_eval["objective"]
    history = [best_obj]
    tabu: List[Tuple[float, ...]] = []
    tabu_size = config.get("tabu_size", 25)
    steps = config.get("steps", [0.02, 0.05, 0.08, 0.12])  # Ne, Nr, Nc, NT
    max_iters = config.get("iters", 100)

    for _ in range(max_iters):
        candidates = []
        for step in steps:
            for _ in range(3):
                cand = perturb(current, rng, step)
                key = tuple(round(v, 6) for v in cand.values())
                if key in tabu:
                    continue
                cand_eval = eval_objective(cand, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
                obj = cand_eval["objective"]
                candidates.append((obj, cand, key))
                history.append(obj)
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        obj, cand, key = candidates[0]
        tabu.append(key)
        if len(tabu) > tabu_size:
            tabu.pop(0)
        current = cand
        if obj < best_obj:
            best, best_obj = cand, obj

    return best, history


def run_wilb(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    rng: np.random.Generator,
    config: Dict,
    objective_key: str,
) -> Tuple[Dict[str, float], List[float]]:
    """Weighted Iterated Local Branching simplificado com grupos de variaveis e deltas diferentes."""
    current = random_weights(rng)
    best = current
    best_eval = eval_objective(best, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
    best_obj = best_eval["objective"]
    history = [best_obj]
    delta = config.get("delta", 0.05)
    delta_inc = config.get("delta_inc", 0.03)
    irreg = False
    max_iters = config.get("iters", 60)

    for _ in range(max_iters):
        w_vec = np.array(list(current.values()))
        # pesos (quanto dista de media)
        contributions = np.abs(w_vec - w_vec.mean())
        order = np.argsort(contributions)
        g1 = order[:1].tolist()
        g2 = order[1:3].tolist()
        g3 = order[3:].tolist()

        # intensificacao: pequenos passos em g1/g2
        cand = perturb(current, rng, delta, dims=g1 + g2)
        cand_eval = eval_objective(cand, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
        obj = cand_eval["objective"]
        history.append(obj)

        if obj < best_obj:
            best, best_obj = cand, obj
            current = cand
            irreg = False
            delta = max(0.02, delta - 0.01)
            continue

        # diversificacao leve: mexe em g3 com passo maior
        cand = perturb(current, rng, delta + 0.05, dims=g3 if g3 else None)
        cand_eval2 = eval_objective(cand, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
        obj2 = cand_eval2["objective"]
        history.append(obj2)
        if obj2 < best_obj:
            best, best_obj = cand, obj2
            current = cand
            irreg = False
            delta = max(0.02, delta - 0.01)
        else:
            irreg = True
            delta += delta_inc

        # iteracao irregular opcional: forcar mudanca em g3
        if irreg and g3:
            cand = perturb(current, rng, delta + 0.08, dims=g3)
            cand_eval3 = eval_objective(cand, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
            obj3 = cand_eval3["objective"]
            history.append(obj3)
            if obj3 < best_obj:
                best, best_obj = cand, obj3
                current = cand
                delta = max(0.02, delta - 0.01)

    return best, history


def run_ils_sa(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    rng: np.random.Generator,
    config: Dict,
    objective_key: str,
) -> Tuple[Dict[str, float], List[float]]:
    """Iterated Local Search com aceitacao SA (ILS1/ILS2)."""
    current = random_weights(rng)
    best = current
    best_eval = eval_objective(best, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
    best_obj = best_eval["objective"]
    history = [best_obj]
    temp = config.get("temp", 1.0)
    cooling = config.get("cooling", 0.97)
    step_small = config.get("step_small", 0.03)
    step_large = config.get("step_large", 0.10)
    max_iters = config.get("iters", 120)

    for _ in range(max_iters):
        # ILS1: passos menores (insertion-like)
        candidates = [perturb(current, rng, step_small) for _ in range(4)]
        # ILS2: passos maiores (permutation-like)
        candidates += [perturb(current, rng, step_large) for _ in range(2)]
        objs = []
        for cand in candidates:
            cand_eval = eval_objective(cand, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
            obj = cand_eval["objective"]
            objs.append((obj, cand))
            history.append(obj)
        objs.sort(key=lambda x: x[0])
        best_neighbor_obj, best_neighbor = objs[0]

        if best_neighbor_obj < best_obj:
            current = best_neighbor
            best = best_neighbor
            best_obj = best_neighbor_obj
        else:
            delta = best_neighbor_obj - best_obj
            if rng.random() < np.exp(-delta / max(temp, 1e-6)):
                current = best_neighbor
        temp *= cooling

    return best, history


def run_lbh(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    rng: np.random.Generator,
    config: Dict,
    objective_key: str,
) -> Tuple[Dict[str, float], List[float]]:
    """Local Branching Heuristic simplificado com raio (Lambda) adaptativo."""
    current = random_weights(rng)
    best = current
    best_eval = eval_objective(best, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
    best_obj = best_eval["objective"]
    history = [best_obj]
    radius = config.get("radius", 0.08)
    base_radius = radius
    max_iters = config.get("iters", 80)

    for _ in range(max_iters):
        cand = perturb(current, rng, radius)
        cand_eval = eval_objective(cand, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
        obj = cand_eval["objective"]
        history.append(obj)
        if obj < best_obj:
            best, best_obj = cand, obj
            current = cand
            radius = base_radius  # intensificacao
        else:
            radius += base_radius / 2  # diversificacao leve

    return best, history


def run_algo(
    name: str,
    config: Dict,
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    rng: np.random.Generator,
    objective_key: str,
) -> Tuple[Dict[str, float], Dict]:
    if name == "I2PLS":
        best, hist = run_i2pls(metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, rng, config, objective_key)
    elif name == "MTS":
        best, hist = run_mts(metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, rng, config, objective_key)
    elif name == "WILB":
        best, hist = run_wilb(metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, rng, config, objective_key)
    elif name == "ILS":
        best, hist = run_ils_sa(metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, rng, config, objective_key)
    else:  # LBH
        best, hist = run_lbh(metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, rng, config, objective_key)
    eval_res = evaluate_multi(best, metrics_df, baseline_ranks, baseline_scores, fuzziness, vikor_v, weight_refs, objective_key)
    return best, {"history": hist, **eval_res}


def aggregate_histories(histories: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not histories:
        return np.array([]), np.array([])
    max_len = min(len(h) for h in histories)
    trimmed = np.array([h[:max_len] for h in histories])
    mean = np.nanmean(trimmed, axis=0)
    std = np.nanstd(trimmed, axis=0)
    return mean, std


def plot_history(histories: List[List[float]], out_path: Path) -> None:
    mean, std = aggregate_histories(histories)
    if mean.size == 0:
        return
    x = np.arange(len(mean))
    plt.figure(figsize=(7, 4))
    plt.plot(x, mean, label="media")
    plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Iteracao")
    plt.ylabel("Objetivo (menor melhor)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_best_history(history: List[float], out_path: Path) -> None:
    if not history:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(history)), history, label="best run")
    plt.xlabel("Iteracao")
    plt.ylabel("Objetivo (menor melhor)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_objective_boxplot(objectives: List[float], out_path: Path) -> None:
    if not objectives:
        return
    plt.figure(figsize=(6, 4))
    plt.boxplot(objectives, labels=["runs"], showmeans=True)
    plt.ylabel("Objetivo (menor melhor)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_weights(before: Dict[str, float], after: Dict[str, float], out_path: Path) -> None:
    labels = list(before.keys())
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, [before[k] for k in labels], width, label="Base")
    plt.bar(x + width / 2, [after[k] for k in labels], width, label="Melhor")
    plt.xticks(x, labels)
    plt.ylabel("Peso")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Metaheuristicas de vizinhanca para otimizar pesos AHP/MCDM (I2PLS, MTS, WILB, ILS, LBH).")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="CSV consolidado")
    parser.add_argument("--diesel-price", type=float, default=1.2, help="Preco do diesel (USD/L)")
    parser.add_argument("--diesel-map", type=str, help="Mapa Region:preco (ex: Accra:0.95,Lusaka:1.16,Lodwar:0.85)")
    parser.add_argument("--fuzziness", type=float, default=0.05, help="Fator fuzzy")
    parser.add_argument("--vikor-v", type=float, default=0.5, help="Parametro v VIKOR")
    parser.add_argument("--runs", type=int, default=30, help="Execucoes por configuracao")
    parser.add_argument("--seed", type=int, default=123, help="Semente base")
    parser.add_argument("--out-dir", type=Path, default=Path("neighborhood_results"), help="Diretorio de saida")
    parser.add_argument("--objective", type=str, choices=OBJECTIVE_KEYS, default="entropy", help="Funcao objetivo (tabela do README)")
    args = parser.parse_args()

    out_base = args.out_dir / args.objective
    out_base.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(args.seed)

    price_map = parse_diesel_map(args.diesel_map)
    df_raw = pd.read_csv(args.csv)
    metrics_df = aggregate_metrics_by_alternative(
        df_raw, diesel_price_per_liter=args.diesel_price, diesel_price_map=price_map
    )
    metrics_df = metrics_df.dropna(axis=1, how="all")
    base_res = compute_profile_results(metrics_df, profile_name="base", profile_weights=BASE_WEIGHTS, fuzziness=args.fuzziness, vikor_v=args.vikor_v)
    baseline_ranks = {
        "topsis": base_res["fuzzy_topsis_rank"],
        "vikor": base_res["vikor_rank"],
        "copras": base_res["copras_rank"],
        "moora": base_res["moora_rank"],
    }
    baseline_scores = {
        "topsis": base_res["fuzzy_topsis_score"],
        "vikor": base_res["vikor_score"],
        "copras": base_res["copras_score"],
        "moora": base_res["moora_score"],
    }
    entropy_w = compute_entropy_weights(metrics_df)
    critic_w = compute_critic_weights(metrics_df)
    merec_w = compute_merec_weights(metrics_df)
    lopcow_w = compute_lopcow_weights(metrics_df)
    mean_w = compute_mean_weights(metrics_df)
    bayes_w = compute_bayes_weights(BASE_WEIGHTS, entropy_w)
    weight_refs = {
        "entropy": entropy_w,
        "critic": critic_w,
        "merec": merec_w,
        "lopcow": lopcow_w,
        "mean": mean_w,
        "bayes": bayes_w,
    }
    baseline_eval = evaluate_multi(BASE_WEIGHTS, metrics_df, baseline_ranks, baseline_scores, args.fuzziness, args.vikor_v, weight_refs, args.objective)
    baseline_obj = baseline_eval["objective"]
    baseline_cr = baseline_eval["cr"]
    baseline_rho = baseline_eval["rho_mean"]

    # grades de configuracoes (8 cada)
    i2pls_grid = [
        {"name": "i2pls_1", "steps_explore": [0.03, 0.08], "step_tabu": 0.05, "escape_step": 0.15, "stall_limit": 8, "max_iters": 70, "tabu_size": 10},
        {"name": "i2pls_2", "steps_explore": [0.02, 0.07], "step_tabu": 0.04, "escape_step": 0.12, "stall_limit": 10, "max_iters": 80, "tabu_size": 12},
        {"name": "i2pls_3", "steps_explore": [0.04, 0.10], "step_tabu": 0.06, "escape_step": 0.18, "stall_limit": 12, "max_iters": 90, "tabu_size": 14},
        {"name": "i2pls_4", "steps_explore": [0.05, 0.12], "step_tabu": 0.07, "escape_step": 0.20, "stall_limit": 9, "max_iters": 70, "tabu_size": 10},
        {"name": "i2pls_5", "steps_explore": [0.025, 0.09], "step_tabu": 0.05, "escape_step": 0.16, "stall_limit": 11, "max_iters": 100, "tabu_size": 16},
        {"name": "i2pls_6", "steps_explore": [0.03, 0.06], "step_tabu": 0.04, "escape_step": 0.14, "stall_limit": 7, "max_iters": 80, "tabu_size": 12},
        {"name": "i2pls_7", "steps_explore": [0.02, 0.05], "step_tabu": 0.03, "escape_step": 0.10, "stall_limit": 6, "max_iters": 60, "tabu_size": 10},
        {"name": "i2pls_8", "steps_explore": [0.04, 0.09], "step_tabu": 0.05, "escape_step": 0.17, "stall_limit": 10, "max_iters": 110, "tabu_size": 18},
    ]
    mts_grid = [
        {"name": "mts_1", "steps": [0.02, 0.05, 0.08, 0.12], "tabu_size": 20, "iters": 80},
        {"name": "mts_2", "steps": [0.03, 0.06, 0.09, 0.13], "tabu_size": 22, "iters": 90},
        {"name": "mts_3", "steps": [0.015, 0.04, 0.07, 0.1], "tabu_size": 18, "iters": 70},
        {"name": "mts_4", "steps": [0.025, 0.05, 0.1, 0.15], "tabu_size": 25, "iters": 110},
        {"name": "mts_5", "steps": [0.02, 0.06, 0.1, 0.14], "tabu_size": 28, "iters": 120},
        {"name": "mts_6", "steps": [0.03, 0.05, 0.07, 0.11], "tabu_size": 24, "iters": 100},
        {"name": "mts_7", "steps": [0.018, 0.045, 0.085, 0.12], "tabu_size": 20, "iters": 85},
        {"name": "mts_8", "steps": [0.025, 0.055, 0.095, 0.13], "tabu_size": 26, "iters": 105},
    ]
    wilb_grid = [
        {"name": "wilb_1", "delta": 0.05, "delta_inc": 0.03, "iters": 60},
        {"name": "wilb_2", "delta": 0.04, "delta_inc": 0.025, "iters": 70},
        {"name": "wilb_3", "delta": 0.06, "delta_inc": 0.035, "iters": 80},
        {"name": "wilb_4", "delta": 0.03, "delta_inc": 0.02, "iters": 90},
        {"name": "wilb_5", "delta": 0.07, "delta_inc": 0.04, "iters": 70},
        {"name": "wilb_6", "delta": 0.05, "delta_inc": 0.025, "iters": 100},
        {"name": "wilb_7", "delta": 0.045, "delta_inc": 0.03, "iters": 110},
        {"name": "wilb_8", "delta": 0.055, "delta_inc": 0.035, "iters": 120},
    ]
    ils_grid = [
        {"name": "ils_1", "temp": 1.0, "cooling": 0.97, "step_small": 0.03, "step_large": 0.10, "iters": 100},
        {"name": "ils_2", "temp": 1.2, "cooling": 0.96, "step_small": 0.025, "step_large": 0.09, "iters": 120},
        {"name": "ils_3", "temp": 0.9, "cooling": 0.95, "step_small": 0.035, "step_large": 0.11, "iters": 130},
        {"name": "ils_4", "temp": 1.5, "cooling": 0.94, "step_small": 0.02, "step_large": 0.08, "iters": 140},
        {"name": "ils_5", "temp": 1.1, "cooling": 0.93, "step_small": 0.03, "step_large": 0.12, "iters": 90},
        {"name": "ils_6", "temp": 0.8, "cooling": 0.92, "step_small": 0.04, "step_large": 0.13, "iters": 110},
        {"name": "ils_7", "temp": 1.3, "cooling": 0.95, "step_small": 0.025, "step_large": 0.1, "iters": 115},
        {"name": "ils_8", "temp": 1.0, "cooling": 0.96, "step_small": 0.035, "step_large": 0.11, "iters": 125},
    ]
    lbh_grid = [
        {"name": "lbh_1", "radius": 0.08, "iters": 80},
        {"name": "lbh_2", "radius": 0.06, "iters": 90},
        {"name": "lbh_3", "radius": 0.1, "iters": 100},
        {"name": "lbh_4", "radius": 0.05, "iters": 110},
        {"name": "lbh_5", "radius": 0.12, "iters": 90},
        {"name": "lbh_6", "radius": 0.07, "iters": 120},
        {"name": "lbh_7", "radius": 0.09, "iters": 130},
        {"name": "lbh_8", "radius": 0.11, "iters": 140},
    ]

    grids = {"I2PLS": i2pls_grid, "MTS": mts_grid, "WILB": wilb_grid, "ILS": ils_grid, "LBH": lbh_grid}

    summary_global = []

    def try_ttest(values: List[float], baseline: float) -> Tuple[float, float]:
        try:
            from math import sqrt
            import math

            n = len(values)
            if n < 2:
                return float("nan"), float("nan")
            arr = np.array(values, dtype=float)
            mean = arr.mean()
            std = arr.std(ddof=1)
            if std == 0:
                return float("inf"), 0.0
            t_stat = (mean - baseline) / (std / sqrt(n))
            try:
                from scipy import stats

                p_val = stats.t.sf(abs(t_stat), df=n - 1) * 2
            except Exception:
                p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / np.sqrt(2))))
            return float(t_stat), float(p_val)
        except Exception:
            return float("nan"), float("nan")

    for algo, grid in grids.items():
        for cfg in grid:
            cfg_name = cfg["name"]
            cfg_dir = out_base / algo / cfg_name
            cfg_dir.mkdir(parents=True, exist_ok=True)

            records = []
            best_global = None
            best_obj = float("inf")
            histories: List[List[float]] = []

            for _ in range(args.runs):
                seed_run = int(rng_master.integers(0, 1_000_000_000))
                rng = np.random.default_rng(seed_run)
                best_w, info = run_algo(algo, cfg, metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, rng, args.objective)
                record = {
                    "seed": seed_run,
                    "objective": info["objective"],
                    "rho": info["rho_mean"],
                    "cr": info["cr"],
                    "score_delta": info.get("score_delta"),
                    "entropy_diff": info.get("entropy_diff"),
                    "critic_diff": info.get("critic_diff"),
                    "merec_diff": info.get("merec_diff"),
                    "lopcow_diff": info.get("lopcow_diff"),
                    "mean_diff": info.get("mean_diff"),
                    "bayes_diff": info.get("bayes_diff"),
                    **{f"w_{k}": v for k, v in best_w.items()},
                }
                histories.append(info["history"])
                records.append(record)
                if info["objective"] < best_obj:
                    best_obj = info["objective"]
                    best_global = record

            df = pd.DataFrame(records)
            df.to_csv(cfg_dir / f"runs_{cfg_name}.csv", index=False, float_format="%.10f")
            (cfg_dir / f"best_{cfg_name}.json").write_text(json.dumps(best_global, indent=2), encoding="utf-8")

            objs = df["objective"].values
            rhos = df["rho"].values
            crs = df["cr"].values
            sdelta = df["score_delta"].values
            entd = df["entropy_diff"].values
            crid = df["critic_diff"].values
            merd = df["merec_diff"].values
            lopd = df["lopcow_diff"].values
            meand = df["mean_diff"].values
            bayd = df["bayes_diff"].values
            t_stat, p_val = try_ttest(objs.tolist(), baseline_obj)
            stats_row = {
                "objective_mean": float(np.mean(objs)),
                "objective_std": float(np.std(objs, ddof=0)),
                "rho_mean": float(np.mean(rhos)),
                "rho_std": float(np.std(rhos, ddof=0)),
                "cr_mean": float(np.mean(crs)),
                "cr_std": float(np.std(crs, ddof=0)),
                "score_delta_mean": float(np.mean(sdelta)),
                "score_delta_std": float(np.std(sdelta, ddof=0)),
                "entropy_diff_mean": float(np.mean(entd)),
                "critic_diff_mean": float(np.mean(crid)),
                "merec_diff_mean": float(np.mean(merd)),
                "lopcow_diff_mean": float(np.mean(lopd)),
                "mean_diff_mean": float(np.mean(meand)),
                "bayes_diff_mean": float(np.mean(bayd)),
                "t_stat_vs_baseline": t_stat,
                "p_value_vs_baseline": p_val,
                "significant_vs_baseline": bool(p_val < 0.05),
                "better_than_baseline": bool(np.mean(objs) < baseline_obj),
                "runs": len(records),
                "baseline_objective": baseline_obj,
                "baseline_rho": baseline_rho,
                "baseline_cr": baseline_cr,
            }
            pd.DataFrame([stats_row]).to_csv(cfg_dir / "stats_summary.csv", index=False, float_format="%.10f")
            # melhor execucao
            best_impr = {
                "objective": best_global["objective"],
                "rho": best_global["rho"],
                "cr": best_global["cr"],
                "score_delta": best_global.get("score_delta"),
                "entropy_diff": best_global.get("entropy_diff"),
                "critic_diff": best_global.get("critic_diff"),
                "merec_diff": best_global.get("merec_diff"),
                "lopcow_diff": best_global.get("lopcow_diff"),
                "mean_diff": best_global.get("mean_diff"),
                "bayes_diff": best_global.get("bayes_diff"),
                "seed": best_global["seed"],
            }
            pd.DataFrame([best_impr]).to_csv(cfg_dir / "best_improvements.csv", index=False, float_format="%.10f")
            best_weights = {k.replace("w_", ""): v for k, v in best_global.items() if k.startswith("w_")}
            pd.DataFrame([best_weights]).to_csv(cfg_dir / "best_weights.csv", index=False, float_format="%.10f")

            # ranks base e melhor
            base_res.to_csv(cfg_dir / "ranks_base.csv", index=False, float_format="%.10f")
            best_res = compute_profile_results(metrics_df, profile_name="best", profile_weights=best_weights, fuzziness=args.fuzziness, vikor_v=args.vikor_v)
            best_res.to_csv(cfg_dir / "ranks_best.csv", index=False, float_format="%.10f")

            # graficos
            plot_history(histories, cfg_dir / "objective_history.png")
            best_history = histories[int(np.argmin(objs))] if len(histories) == len(objs) else []
            plot_best_history(best_history, cfg_dir / "best_run_history.png")
            plot_objective_boxplot(objs.tolist(), cfg_dir / "objective_boxplot.png")
            plot_weights(BASE_WEIGHTS, best_weights, cfg_dir / f"weights_{algo}.png")

            summary_global.append(
                {
                    "algorithm": algo,
                    "config": cfg_name,
                    "objective_best": best_obj,
                    "objective_mean": stats_row["objective_mean"],
                    "objective_std": stats_row["objective_std"],
                    "rho_mean": stats_row["rho_mean"],
                    "cr_mean": stats_row["cr_mean"],
                    "score_delta_mean": stats_row.get("score_delta_mean"),
                    "entropy_diff_mean": stats_row.get("entropy_diff_mean"),
                    "critic_diff_mean": stats_row.get("critic_diff_mean"),
                    "merec_diff_mean": stats_row.get("merec_diff_mean"),
                    "lopcow_diff_mean": stats_row.get("lopcow_diff_mean"),
                    "mean_diff_mean": stats_row.get("mean_diff_mean"),
                    "bayes_diff_mean": stats_row.get("bayes_diff_mean"),
                    "p_value_vs_baseline": p_val,
                    "significant_vs_baseline": stats_row["significant_vs_baseline"],
                    "better_than_baseline": stats_row["better_than_baseline"],
                    **{f"w_{k}": v for k, v in best_global.items() if k.startswith("w_")},
                    "seed_best": best_global["seed"] if best_global else None,
                }
            )

    pd.DataFrame(summary_global).to_csv(out_base / "summary_global.csv", index=False, float_format="%.10f")
    print("Concluido. Resultados em", out_base)


if __name__ == "__main__":
    print("Iniciando optimize_weights_neighborhood.py ...")
    main()
