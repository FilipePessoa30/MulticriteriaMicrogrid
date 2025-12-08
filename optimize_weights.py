"""
Otimiza pesos dos criterios (cost, emissions, reliability, social) com tres blocos de metaheuristicas:
- ABC (Cherif & Ladhari, 2016)
- HC/SA/PSO (Kizielewicz & Salabun, 2020)
- PSO-SA hibrido (Sarani Rad et al., 2024)
Cada bloco roda 8 configuracoes e 30 execucoes por configuracao (ajustavel via --runs).

Saidas organizadas em subpastas: out_dir/<algoritmo>/<config_name>/ com:
- runs.csv (todas as execucoes, 10 casas decimais)
- best_weights.csv, best_improvements.csv, stats_summary.csv
- ranks_base.csv, ranks_best.csv
- objective_history.png (media +/- desvio), best_run_history.png (melhor seed),
  objective_boxplot.png, weights_comparison.png, weights_<algoritmo>.png

Tambem gera summary_global.csv com o melhor resultado de cada configuracao.
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


# Pesos iniciais da literatura (AHP)
BASE_WEIGHTS = {"cost": 0.6923, "emissions": 0.3328, "reliability": 0.3492, "social": 0.3351}

# Objetivos suportados (tabela do README)
OBJECTIVE_KEYS = ["entropy", "merec", "lopcow", "critic", "mean", "bayes"]


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


def try_ttest(values: List[float], baseline: float) -> Tuple[float, float]:
    """t-test bicaudal H0: media = baseline. Retorna (t_stat, p_val)."""
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


def spearman_corr(rank_a: pd.Series, rank_b: pd.Series) -> float:
    a = rank_a.reindex(rank_b.index).astype(float).values
    b = rank_b.astype(float).values
    n = len(a)
    if n < 2:
        return 0.0
    diff = a - b
    num = 6 * np.sum(diff ** 2)
    denom = n * (n**2 - 1)
    return 1 - num / denom if denom else 0.0


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
    result = compute_profile_results(metrics_df, profile_name="opt", profile_weights=weights, fuzziness=fuzziness, vikor_v=vikor_v)

    rhos = []
    for key, col in [
        ("topsis", "fuzzy_topsis_rank"),
        ("vikor", "vikor_rank"),
        ("copras", "copras_rank"),
        ("moora", "moora_rank"),
    ]:
        rhos.append(spearman_corr(result[col], baseline_ranks[key]))
    rho_mean = float(np.mean(rhos))

    deltas = []
    for key, col in [
        ("topsis", "fuzzy_topsis_score"),
        ("vikor", "vikor_score"),
        ("copras", "copras_score"),
        ("moora", "moora_score"),
    ]:
        base = baseline_scores[key]
        cur = result[col]
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
    }


def compute_entropy_weights(metrics_df: pd.DataFrame) -> Dict[str, float]:
    data = metrics_df.to_numpy(dtype=float)
    # min-max normalize cols to avoid negatives
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = (col_max - col_min)
    denom[denom == 0] = 1.0
    norm = (data - col_min) / denom
    # avoid log(0)
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


def compute_score_delta(metrics_df: pd.DataFrame, weights: Dict[str, float], baseline_scores: Dict[str, pd.Series], fuzziness: float, vikor_v: float) -> float:
    res = compute_profile_results(metrics_df, profile_name="opt", profile_weights=weights, fuzziness=fuzziness, vikor_v=vikor_v)
    deltas = []
    for key, col in [("topsis", "fuzzy_topsis_score"), ("vikor", "vikor_score"), ("copras", "copras_score"), ("moora", "moora_score")]:
        base = baseline_scores[key]
        cur = res[col]
        deltas.append(np.mean(np.abs(cur - base) / (np.abs(base) + 1e-9)))
    return float(np.mean(deltas))


def compute_mean_weights(metrics_df: pd.DataFrame) -> Dict[str, float]:
    n = metrics_df.shape[1]
    if n == 0:
        return normalize_weights(np.ones(4))
    w = np.ones(n) / n
    return normalize_weights(w)


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


def compute_bayes_weights(base_w: Dict[str, float], obj_w: Dict[str, float]) -> Dict[str, float]:
    # combinacao simples via produto seguido de normalizacao (media geometrica ponderada)
    keys = list(base_w.keys())
    base_vec = np.array([base_w[k] for k in keys], dtype=float)
    obj_vec = np.array([obj_w.get(k, 0.0) for k in keys], dtype=float)
    combined = base_vec * obj_vec
    if combined.sum() == 0:
        combined = base_vec
    combined = combined / combined.sum()
    return dict(zip(keys, combined.tolist()))


def evaluate_objective_value(
    weights: Dict[str, float],
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    objective_key: str,
) -> Tuple[float, float, float]:
    """Retorna (objetivo, cr, rho_mean) usando a funcao objetivo escolhida."""
    res = evaluate_multi(weights, metrics_df, baseline_ranks, baseline_scores, fuzziness, vikor_v, weight_refs, objective_key)
    return res["objective"], res["cr"], res["rho_mean"]


def mutate(vec: np.ndarray, rng: np.random.Generator, rate: float = 0.1) -> np.ndarray:
    noise = rng.normal(0, rate, size=vec.shape)
    mutated = np.clip(vec + noise, 1e-9, None)
    return mutated / mutated.sum()


def run_pso(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    objective_key: str,
    rng: np.random.Generator,
    config: Dict,
) -> Tuple[Dict[str, float], List[float]]:
    dim = 4
    particles = config.get("particles", 20)
    iterations = config.get("iterations", 50)
    w = config.get("w", 0.7)
    c1 = config.get("c1", 1.4)
    c2 = config.get("c2", 1.4)

    positions = rng.dirichlet(np.ones(dim), size=particles)
    velocities = np.zeros_like(positions)
    pbest = positions.copy()
    pbest_scores = np.full(particles, np.inf)
    gbest = positions[0]
    gbest_score = np.inf
    history = []

    for _ in range(iterations):
        for i in range(particles):
            weights = normalize_weights(positions[i])
            obj, _, _ = evaluate_objective_value(weights, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
            if obj < pbest_scores[i]:
                pbest_scores[i] = obj
                pbest[i] = positions[i].copy()
            if obj < gbest_score:
                gbest_score = obj
                gbest = positions[i].copy()

        r1, r2 = rng.random((particles, dim)), rng.random((particles, dim))
        velocities = w * velocities + c1 * r1 * (pbest - positions) + c2 * r2 * (gbest - positions)
        positions = positions + velocities
        positions = np.clip(positions, 1e-9, None)
        positions = positions / positions.sum(axis=1, keepdims=True)
        history.append(gbest_score)

    return normalize_weights(gbest), history


def run_hc(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    objective_key: str,
    rng: np.random.Generator,
    config: Dict,
) -> Tuple[Dict[str, float], List[float]]:
    """Hill-Climbing simples: aceita apenas vizinhos que melhoram a funcao objetivo."""
    steps = config.get("steps", 200)
    neighbor_rate = config.get("neighbor_rate", 0.05)

    current = rng.dirichlet(np.ones(4))
    current_w = normalize_weights(current)
    current_obj, _, _ = evaluate_objective_value(current_w, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
    best = current.copy()
    best_obj = current_obj
    history = [best_obj]

    for _ in range(steps):
        neighbor = mutate(current, rng, rate=neighbor_rate)
        neigh_w = normalize_weights(neighbor)
        neigh_obj, _, _ = evaluate_objective_value(neigh_w, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
        if neigh_obj < current_obj:
            current = neighbor
            current_obj = neigh_obj
            if neigh_obj < best_obj:
                best = neighbor
                best_obj = neigh_obj
        history.append(best_obj)

    return normalize_weights(best), history


def run_sa(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    objective_key: str,
    rng: np.random.Generator,
    config: Dict,
    start: np.ndarray | None = None,
) -> Tuple[Dict[str, float], List[float]]:
    steps = config.get("steps", 200)
    temp = config.get("temp", 1.0)
    cooling = config.get("cooling", 0.97)
    neighbor_rate = config.get("neighbor_rate", 0.05)

    current = start.copy() if start is not None else rng.dirichlet(np.ones(4))
    current_w = normalize_weights(current)
    current_obj, _, _ = evaluate_objective_value(current_w, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
    best = current.copy()
    best_obj = current_obj
    history = [best_obj]

    for _ in range(steps):
        neighbor = mutate(current, rng, rate=neighbor_rate)
        neigh_w = normalize_weights(neighbor)
        neigh_obj, _, _ = evaluate_objective_value(neigh_w, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)

        if neigh_obj < current_obj or rng.random() < np.exp(-(neigh_obj - current_obj) / max(temp, 1e-6)):
            current = neighbor
            current_obj = neigh_obj

        if neigh_obj < best_obj:
            best = neighbor
            best_obj = neigh_obj

        history.append(best_obj)
        temp *= cooling

    return normalize_weights(best), history


def run_abc(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    objective_key: str,
    rng: np.random.Generator,
    config: Dict,
) -> Tuple[Dict[str, float], List[float]]:
    """Artificial Bee Colony adaptado para pesos VIKOR (soma 1)."""
    dim = 4
    foods = config.get("foods", 20)  # fontes = abelhas empregadas = onlookers
    iterations = config.get("iterations", 60)
    limit = config.get("limit", 10)  # limite sem melhoria para virar scout

    positions = rng.dirichlet(np.ones(dim), size=foods)
    trials = np.zeros(foods, dtype=int)
    objs = np.zeros(foods)
    fits = np.zeros(foods)

    def eval_source(vec: np.ndarray) -> Tuple[float, float]:
        w = normalize_weights(vec)
        obj, _, _ = evaluate_objective_value(w, metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key)
        fitness = 1.0 / (obj + 1e-12)
        return obj, fitness

    for i in range(foods):
        obj, fit = eval_source(positions[i])
        objs[i] = obj
        fits[i] = fit

    best_idx = int(np.argmin(objs))
    best_vec = positions[best_idx].copy()
    best_obj = objs[best_idx]
    history: List[float] = [best_obj]

    def mutate_neighbor(i: int) -> np.ndarray:
        k = int(rng.integers(0, foods))
        while k == i:
            k = int(rng.integers(0, foods))
        j = int(rng.integers(0, dim))
        phi = rng.uniform(-1, 1)
        candidate = positions[i].copy()
        candidate[j] = positions[i][j] + phi * (positions[i][j] - positions[k][j])
        candidate = np.clip(candidate, 1e-9, None)
        candidate = candidate / candidate.sum()
        return candidate

    for _ in range(iterations):
        # Abelhas empregadas
        for i in range(foods):
            cand = mutate_neighbor(i)
            cand_obj, cand_fit = eval_source(cand)
            if cand_obj < objs[i]:
                positions[i] = cand
                objs[i] = cand_obj
                fits[i] = cand_fit
                trials[i] = 0
            else:
                trials[i] += 1

        # Abelhas observadoras
        prob = fits / (fits.sum() + 1e-12)
        for _ in range(foods):
            i = int(rng.choice(foods, p=prob))
            cand = mutate_neighbor(i)
            cand_obj, cand_fit = eval_source(cand)
            if cand_obj < objs[i]:
                positions[i] = cand
                objs[i] = cand_obj
                fits[i] = cand_fit
                trials[i] = 0
            else:
                trials[i] += 1

        # Abelhas exploradoras
        for i in range(foods):
            if trials[i] >= limit:
                positions[i] = rng.dirichlet(np.ones(dim))
                objs[i], fits[i] = eval_source(positions[i])
                trials[i] = 0

        best_idx = int(np.argmin(objs))
        if objs[best_idx] < best_obj:
            best_obj = objs[best_idx]
            best_vec = positions[best_idx].copy()
        history.append(best_obj)

    return normalize_weights(best_vec), history


def run_pso_sa_hybrid(
    metrics_df: pd.DataFrame,
    baseline_ranks: Dict[str, pd.Series],
    baseline_scores: Dict[str, pd.Series],
    weight_refs: Dict[str, Dict[str, float]],
    fuzziness: float,
    vikor_v: float,
    objective_key: str,
    rng: np.random.Generator,
    config: Dict,
) -> Tuple[Dict[str, float], List[float]]:
    """Hibrido: PSO seguido de refinamento por SA iniciando do melhor enxame."""
    # separar parametros de PSO e SA
    pso_cfg = {
        "particles": config.get("pso_particles", config.get("particles", 20)),
        "iterations": config.get("pso_iterations", config.get("iterations", 40)),
        "w": config.get("w", 0.7),
        "c1": config.get("c1", 1.4),
        "c2": config.get("c2", 1.4),
    }
    sa_cfg = {
        "steps": config.get("sa_steps", 80),
        "temp": config.get("sa_temp", 1.0),
        "cooling": config.get("sa_cooling", 0.97),
        "neighbor_rate": config.get("sa_neighbor_rate", 0.05),
    }

    # PSO
    best_pso_weights, pso_hist = run_pso(metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key, rng, pso_cfg)
    # SA refinando
    start_vec = np.array(list(best_pso_weights.values()), dtype=float)
    sa_weights, sa_hist = run_sa(metrics_df, baseline_ranks, baseline_scores, weight_refs, fuzziness, vikor_v, objective_key, rng, sa_cfg, start=start_vec)
    # concatena historico
    history = pso_hist + sa_hist
    return sa_weights, history


def aggregate_histories(histories: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not histories:
        return np.array([]), np.array([])
    max_len = min(len(h) for h in histories)
    trimmed = np.array([h[:max_len] for h in histories])
    mean = np.nanmean(trimmed, axis=0)
    std = np.nanstd(trimmed, axis=0)
    return mean, std


def plot_history(histories: Dict[str, List[List[float]]], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for name, runs in histories.items():
        mean, std = aggregate_histories(runs)
        if mean.size == 0:
            continue
        x = np.arange(len(mean))
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.xlabel("Iteracao")
    plt.ylabel("Objetivo (menor melhor)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_best_histories(best_histories: Dict[str, List[float]], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for name, hist in best_histories.items():
        x = np.arange(len(hist))
        plt.plot(x, hist, label=f"{name} (best seed)")
    plt.xlabel("Iteracao")
    plt.ylabel("Objetivo (melhor execucao)")
    plt.legend()
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


def plot_weights_by_algo(before: Dict[str, float], bests: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    for algo, w in bests.items():
        plot_weights(before, w, out_dir / f"weights_{algo}.png")


def plot_objective_boxplot(objectives: Dict[str, List[float]], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    data = [vals for vals in objectives.values()]
    labels = list(objectives.keys())
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Objetivo (menor melhor)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_runs_table(runs: List[Dict], out_path: Path) -> None:
    rows = []
    for r in runs:
        row = {
            "seed": r["seed"],
            "objective": r["objective"],
            "rho": r["rho"],
            "cr_proxy": r["cr"],
            "score_delta": r.get("score_delta"),
            "entropy_diff": r.get("entropy_diff"),
            "critic_diff": r.get("critic_diff"),
        }
        row.update({f"w_{k}": v for k, v in r["weights"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.10f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Otimiza pesos AHP com ABC, HC/SA/PSO e PSO-SA em multiplas configuracoes.")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="CSV consolidado (dados_preprocessados/reopt_ALL_blocks_v3_8.csv)")
    parser.add_argument("--diesel-price", type=float, default=1.2, help="Preco do diesel (USD/L)")
    parser.add_argument("--diesel-map", type=str, help="Mapa Region:preco (ex: Accra:0.95,Lusaka:1.16,Lodwar:0.85)")
    parser.add_argument("--fuzziness", type=float, default=0.05, help="Fator fuzzy para Fuzzy-TOPSIS")
    parser.add_argument("--vikor-v", type=float, default=0.5, help="Parametro v do VIKOR")
    parser.add_argument("--runs", type=int, default=30, help="Numero de execucoes por configuracao")
    parser.add_argument("--seed", type=int, default=123, help="Semente base")
    parser.add_argument("--out-dir", type=Path, default=Path("weight_optimization"), help="Diretorio de saida")
    parser.add_argument("--objective", type=str, choices=OBJECTIVE_KEYS, default="entropy", help="Funcao objetivo (tabela do README)")
    args = parser.parse_args()

    out_base = args.out_dir / args.objective
    out_base.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)

    # Grids de configuracao
    abc_grid = [
        {"name": "abc_1", "foods": 15, "iterations": 40, "limit": 8},
        {"name": "abc_2", "foods": 20, "iterations": 50, "limit": 10},
        {"name": "abc_3", "foods": 25, "iterations": 60, "limit": 12},
        {"name": "abc_4", "foods": 18, "iterations": 70, "limit": 9},
        {"name": "abc_5", "foods": 22, "iterations": 55, "limit": 10},
        {"name": "abc_6", "foods": 30, "iterations": 80, "limit": 15},
        {"name": "abc_7", "foods": 16, "iterations": 90, "limit": 7},
        {"name": "abc_8", "foods": 24, "iterations": 65, "limit": 11},
    ]
    kiz_grid = [
        {"name": "kiz_hc_1", "method": "HC", "steps": 120, "neighbor_rate": 0.05},
        {"name": "kiz_hc_2", "method": "HC", "steps": 200, "neighbor_rate": 0.07},
        {"name": "kiz_hc_3", "method": "HC", "steps": 300, "neighbor_rate": 0.05},
        {"name": "kiz_sa_1", "method": "SA", "steps": 180, "temp": 1.0, "cooling": 0.95, "neighbor_rate": 0.05},
        {"name": "kiz_sa_2", "method": "SA", "steps": 250, "temp": 1.2, "cooling": 0.94, "neighbor_rate": 0.08},
        {"name": "kiz_sa_3", "method": "SA", "steps": 300, "temp": 1.6, "cooling": 0.95, "neighbor_rate": 0.07},
        {"name": "kiz_pso_1", "method": "PSO", "particles": 20, "iterations": 50, "w": 0.7, "c1": 1.4, "c2": 1.4},
        {"name": "kiz_pso_2", "method": "PSO", "particles": 28, "iterations": 60, "w": 0.6, "c1": 1.6, "c2": 1.6},
    ]
    pso_sa_grid = [
        {"name": "pso_sa_1", "particles": 15, "iterations": 30, "w": 0.7, "c1": 1.4, "c2": 1.4, "sa_steps": 60, "sa_temp": 1.0, "sa_cooling": 0.96, "sa_neighbor_rate": 0.05},
        {"name": "pso_sa_2", "particles": 20, "iterations": 40, "w": 0.6, "c1": 1.6, "c2": 1.6, "sa_steps": 80, "sa_temp": 1.2, "sa_cooling": 0.95, "sa_neighbor_rate": 0.06},
        {"name": "pso_sa_3", "particles": 25, "iterations": 50, "w": 0.8, "c1": 1.2, "c2": 1.6, "sa_steps": 100, "sa_temp": 1.5, "sa_cooling": 0.94, "sa_neighbor_rate": 0.05},
        {"name": "pso_sa_4", "particles": 18, "iterations": 60, "w": 0.5, "c1": 2.0, "c2": 2.0, "sa_steps": 90, "sa_temp": 1.0, "sa_cooling": 0.93, "sa_neighbor_rate": 0.07},
        {"name": "pso_sa_5", "particles": 30, "iterations": 35, "w": 0.9, "c1": 1.0, "c2": 1.0, "sa_steps": 70, "sa_temp": 0.8, "sa_cooling": 0.95, "sa_neighbor_rate": 0.08},
        {"name": "pso_sa_6", "particles": 18, "iterations": 45, "w": 0.65, "c1": 1.8, "c2": 1.3, "sa_steps": 85, "sa_temp": 1.3, "sa_cooling": 0.96, "sa_neighbor_rate": 0.06},
        {"name": "pso_sa_7", "particles": 28, "iterations": 55, "w": 0.75, "c1": 1.3, "c2": 1.8, "sa_steps": 95, "sa_temp": 1.1, "sa_cooling": 0.94, "sa_neighbor_rate": 0.05},
        {"name": "pso_sa_8", "particles": 22, "iterations": 70, "w": 0.55, "c1": 2.1, "c2": 1.7, "sa_steps": 110, "sa_temp": 1.6, "sa_cooling": 0.95, "sa_neighbor_rate": 0.07},
    ]
    # Dados agregados
    price_map = parse_diesel_map(args.diesel_map)
    df_raw = pd.read_csv(args.csv)
    metrics_df = aggregate_metrics_by_alternative(
        df_raw, diesel_price_per_liter=args.diesel_price, diesel_price_map=price_map
    )
    metrics_df = metrics_df.dropna(axis=1, how="all")

    base_result = compute_profile_results(metrics_df, profile_name="base", profile_weights=BASE_WEIGHTS, fuzziness=args.fuzziness, vikor_v=args.vikor_v)
    baseline_ranks = {
        "topsis": base_result["fuzzy_topsis_rank"],
        "vikor": base_result["vikor_rank"],
        "copras": base_result["copras_rank"],
        "moora": base_result["moora_rank"],
    }
    baseline_scores = {
        "topsis": base_result["fuzzy_topsis_score"],
        "vikor": base_result["vikor_score"],
        "copras": base_result["copras_score"],
        "moora": base_result["moora_score"],
    }
    entropy_weights = compute_entropy_weights(metrics_df)
    critic_weights = compute_critic_weights(metrics_df)
    merec_weights = compute_merec_weights(metrics_df)
    lopcow_weights = compute_lopcow_weights(metrics_df)
    mean_weights = compute_mean_weights(metrics_df)
    bayes_weights = compute_bayes_weights(BASE_WEIGHTS, entropy_weights)
    weight_refs = {
        "entropy": entropy_weights,
        "critic": critic_weights,
        "merec": merec_weights,
        "lopcow": lopcow_weights,
        "mean": mean_weights,
        "bayes": bayes_weights,
    }
    baseline_eval = evaluate_multi(BASE_WEIGHTS, metrics_df, baseline_ranks, baseline_scores, args.fuzziness, args.vikor_v, weight_refs, args.objective)
    baseline_obj = baseline_eval["objective"]
    baseline_cr = baseline_eval["cr"]
    baseline_rho = baseline_eval["rho_mean"]

    global_summary_rows = []

    def run_config(algo: str, config: Dict) -> None:
        cfg_name = config["name"]
        cfg_dir = out_base / algo / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        histories_all: Dict[str, List[List[float]]] = {algo: []}
        best_histories: Dict[str, List[float]] = {}
        objectives_all: Dict[str, List[float]] = {algo: []}
        run_records: List[Dict] = []

        for _ in range(args.runs):
            seed = int(master_rng.integers(0, 1_000_000_000))
            rng = np.random.default_rng(seed)

            if algo == "PSO":
                weights, hist = run_pso(metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, args.objective, rng, config)
            elif algo == "ABC":
                weights, hist = run_abc(metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, args.objective, rng, config)
            elif algo == "HC":
                weights, hist = run_hc(metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, args.objective, rng, config)
            elif algo == "SA":
                weights, hist = run_sa(metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, args.objective, rng, config)
            elif algo == "PSO_SA":
                weights, hist = run_pso_sa_hybrid(metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, args.objective, rng, config)
            elif algo == "HC_SA_PSO":
                method = config.get("method")
                if method == "HC":
                    weights, hist = run_hc(metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, args.objective, rng, config)
                elif method == "SA":
                    weights, hist = run_sa(metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, args.objective, rng, config)
                elif method == "PSO":
                    weights, hist = run_pso(metrics_df, baseline_ranks, baseline_scores, weight_refs, args.fuzziness, args.vikor_v, args.objective, rng, config)
                else:
                    raise ValueError(f"Metodo desconhecido em HC_SA_PSO: {method}")
            else:
                raise ValueError(f"Algoritmo desconhecido: {algo}")

            eval_res = evaluate_multi(weights, metrics_df, baseline_ranks, baseline_scores, args.fuzziness, args.vikor_v, weight_refs, args.objective)
            obj = eval_res["objective"]
            cr = eval_res["cr"]
            rho = eval_res["rho_mean"]
            histories_all[algo].append(hist)
            objectives_all[algo].append(obj)
            run_records.append(
                {
                    "seed": seed,
                    "weights": weights,
                    "objective": obj,
                    "cr": cr,
                    "rho": rho,
                    "score_delta": eval_res["score_delta"],
                    "entropy_diff": eval_res.get("entropy_diff"),
                    "critic_diff": eval_res.get("critic_diff"),
                    "merec_diff": eval_res.get("merec_diff"),
                    "lopcow_diff": eval_res.get("lopcow_diff"),
                    "mean_diff": eval_res.get("mean_diff"),
                    "bayes_diff": eval_res.get("bayes_diff"),
                    "history": hist,
                }
            )

        # salvar runs
        save_runs_table(run_records, cfg_dir / "runs.csv")

        # melhor execucao
        objs = np.array([r["objective"] for r in run_records])
        rhos = np.array([r["rho"] for r in run_records])
        crs = np.array([r["cr"] for r in run_records])
        sdelta = np.array([r.get("score_delta", np.nan) for r in run_records])
        entd = np.array([r.get("entropy_diff", np.nan) for r in run_records])
        crid = np.array([r.get("critic_diff", np.nan) for r in run_records])
        merd = np.array([r.get("merec_diff", np.nan) for r in run_records])
        lopd = np.array([r.get("lopcow_diff", np.nan) for r in run_records])
        meand = np.array([r.get("mean_diff", np.nan) for r in run_records])
        bayd = np.array([r.get("bayes_diff", np.nan) for r in run_records])
        best_idx = int(np.argmin(objs))
        best_weights = run_records[best_idx]["weights"]
        best_histories[algo] = run_records[best_idx]["history"]
        improvements = {
            "objective": run_records[best_idx]["objective"],
            "rho": run_records[best_idx]["rho"],
            "cr": run_records[best_idx]["cr"],
            "score_delta": run_records[best_idx].get("score_delta"),
            "entropy_diff": run_records[best_idx].get("entropy_diff"),
            "critic_diff": run_records[best_idx].get("critic_diff"),
            "merec_diff": run_records[best_idx].get("merec_diff"),
            "lopcow_diff": run_records[best_idx].get("lopcow_diff"),
            "mean_diff": run_records[best_idx].get("mean_diff"),
            "bayes_diff": run_records[best_idx].get("bayes_diff"),
            "seed": run_records[best_idx]["seed"],
        }
        t_stat, p_val = try_ttest(objs.tolist(), baseline_obj)
        stats = {
            "objective_mean": float(objs.mean()),
            "objective_std": float(objs.std(ddof=0)),
            "rho_mean": float(rhos.mean()),
            "rho_std": float(rhos.std(ddof=0)),
            "cr_mean": float(crs.mean()),
            "cr_std": float(crs.std(ddof=0)),
            "score_delta_mean": float(np.nanmean(sdelta)),
            "score_delta_std": float(np.nanstd(sdelta)),
            "entropy_diff_mean": float(np.nanmean(entd)),
            "critic_diff_mean": float(np.nanmean(crid)),
            "merec_diff_mean": float(np.nanmean(merd)),
            "lopcow_diff_mean": float(np.nanmean(lopd)),
            "mean_diff_mean": float(np.nanmean(meand)),
            "bayes_diff_mean": float(np.nanmean(bayd)),
            "t_stat_vs_baseline": t_stat,
            "p_value_vs_baseline": p_val,
            "significant_vs_baseline": bool(p_val < 0.05),
            "better_than_baseline": bool(objs.mean() < baseline_obj),
        }

        # ranks com melhor peso
        best_result = compute_profile_results(metrics_df, profile_name="best", profile_weights=best_weights, fuzziness=args.fuzziness, vikor_v=args.vikor_v)

        # salvar tabelas
        pd.DataFrame([best_weights]).to_csv(cfg_dir / "best_weights.csv", index=False, float_format="%.10f")
        pd.DataFrame([improvements]).to_csv(cfg_dir / "best_improvements.csv", index=False, float_format="%.10f")
        pd.DataFrame([stats]).to_csv(cfg_dir / "stats_summary.csv", index=False, float_format="%.10f")
        base_result.to_csv(cfg_dir / "ranks_base.csv", float_format="%.10f")
        best_result.to_csv(cfg_dir / "ranks_best.csv", float_format="%.10f")

        # graficos
        plot_history(histories_all, cfg_dir / "objective_history.png")
        plot_best_histories(best_histories, cfg_dir / "best_run_history.png")
        plot_objective_boxplot(objectives_all, cfg_dir / "objective_boxplot.png")
        plot_weights(BASE_WEIGHTS, best_weights, cfg_dir / f"weights_{algo}.png")

        # resumo global
        global_summary_rows.append(
            {
                "algorithm": algo,
                "config": cfg_name,
                "objective": improvements["objective"],
                "rho": improvements["rho"],
                "cr": improvements["cr"],
                "score_delta": improvements.get("score_delta"),
                "entropy_diff": improvements.get("entropy_diff"),
                "critic_diff": improvements.get("critic_diff"),
                "merec_diff": improvements.get("merec_diff"),
                "lopcow_diff": improvements.get("lopcow_diff"),
                "mean_diff": improvements.get("mean_diff"),
                "bayes_diff": improvements.get("bayes_diff"),
                "seed": improvements["seed"],
                **{f"w_{k}": v for k, v in best_weights.items()},
                "objective_mean": stats["objective_mean"],
                "objective_std": stats["objective_std"],
                "p_value_vs_baseline": stats["p_value_vs_baseline"],
                "significant_vs_baseline": stats["significant_vs_baseline"],
                "better_than_baseline": stats["better_than_baseline"],
            }
        )

    for cfg in abc_grid:
        run_config("ABC", cfg)
    for cfg in kiz_grid:
        run_config("HC_SA_PSO", cfg)
    for cfg in pso_sa_grid:
        run_config("PSO_SA", cfg)

    pd.DataFrame(global_summary_rows).to_csv(out_base / "summary_global.csv", index=False, float_format="%.10f")

    summary = {
        "source_csv": str(args.csv),
        "baseline_weights": BASE_WEIGHTS,
        "baseline_objective": baseline_obj,
        "objective_key": args.objective,
        "grids": {"ABC": abc_grid, "HC_SA_PSO": kiz_grid, "PSO_SA": pso_sa_grid},
        "runs_per_config": args.runs,
        "seed_base": args.seed,
    }
    (out_base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Execucao completa. Resultados em", out_base)


if __name__ == "__main__":
    print("Iniciando optimize_weights.py ...")
    main()
