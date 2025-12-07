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

import numpy as np
import pandas as pd

from apply_mcdm_profiles import compute_profile_results
from build_ahp_structure import DATA_PATH, aggregate_metrics_by_alternative, parse_diesel_map


BASE_WEIGHTS = {"cost": 0.40, "emissions": 0.30, "reliability": 0.20, "social": 0.10}
ALPHA = 0.5
BETAB = 0.5


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


def evaluate(weights: Dict[str, float], metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float) -> Tuple[float, float, float]:
    cr = abs(sum(weights.values()) - 1.0)
    res = compute_profile_results(metrics_df, profile_name="opt", profile_weights=weights, fuzziness=fuzziness, vikor_v=vikor_v)
    rho = spearman_corr(res["fuzzy_topsis_rank"], baseline_ranks)
    obj = ALPHA * cr + BETAB * (1 - rho)
    return float(obj), float(cr), float(rho)


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


def run_i2pls(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, config: Dict) -> Tuple[Dict[str, float], List[float]]:
    """Iterated Two-Phase Local Search: VND+TS (explore) e perturbação por frequência (escape)."""
    current = random_weights(rng)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
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
                obj, _, _ = evaluate(cand, metrics_df, baseline_ranks, fuzziness, vikor_v)
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
        obj, _, _ = evaluate(cand, metrics_df, baseline_ranks, fuzziness, vikor_v)
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


def run_mts(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, config: Dict) -> Tuple[Dict[str, float], List[float]]:
    """Multi-Neighborhood Tabu Search com quatro passos de vizinhanca restritos."""
    current = random_weights(rng)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
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
                obj, _, _ = evaluate(cand, metrics_df, baseline_ranks, fuzziness, vikor_v)
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


def run_wilb(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, config: Dict) -> Tuple[Dict[str, float], List[float]]:
    """Weighted Iterated Local Branching simplificado com grupos de variaveis e deltas diferentes."""
    current = random_weights(rng)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
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
        obj, _, _ = evaluate(cand, metrics_df, baseline_ranks, fuzziness, vikor_v)
        history.append(obj)

        if obj < best_obj:
            best, best_obj = cand, obj
            current = cand
            irreg = False
            delta = max(0.02, delta - 0.01)
            continue

        # diversificacao leve: mexe em g3 com passo maior
        cand = perturb(current, rng, delta + 0.05, dims=g3 if g3 else None)
        obj2, _, _ = evaluate(cand, metrics_df, baseline_ranks, fuzziness, vikor_v)
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
            obj3, _, _ = evaluate(cand, metrics_df, baseline_ranks, fuzziness, vikor_v)
            history.append(obj3)
            if obj3 < best_obj:
                best, best_obj = cand, obj3
                current = cand
                delta = max(0.02, delta - 0.01)

    return best, history


def run_ils_sa(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, config: Dict) -> Tuple[Dict[str, float], List[float]]:
    """Iterated Local Search com aceitacao SA (ILS1/ILS2)."""
    current = random_weights(rng)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
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
            obj, _, _ = evaluate(cand, metrics_df, baseline_ranks, fuzziness, vikor_v)
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


def run_lbh(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, config: Dict) -> Tuple[Dict[str, float], List[float]]:
    """Local Branching Heuristic simplificado com raio (Lambda) adaptativo."""
    current = random_weights(rng)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
    history = [best_obj]
    radius = config.get("radius", 0.08)
    base_radius = radius
    max_iters = config.get("iters", 80)

    for _ in range(max_iters):
        cand = perturb(current, rng, radius)
        obj, _, _ = evaluate(cand, metrics_df, baseline_ranks, fuzziness, vikor_v)
        history.append(obj)
        if obj < best_obj:
            best, best_obj = cand, obj
            current = cand
            radius = base_radius  # intensificacao
        else:
            radius += base_radius / 2  # diversificacao leve

    return best, history


def run_algo(name: str, config: Dict, metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator) -> Tuple[Dict[str, float], Dict]:
    if name == "I2PLS":
        best, hist = run_i2pls(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, config)
    elif name == "MTS":
        best, hist = run_mts(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, config)
    elif name == "WILB":
        best, hist = run_wilb(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, config)
    elif name == "ILS":
        best, hist = run_ils_sa(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, config)
    else:  # LBH
        best, hist = run_lbh(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, config)
    obj, cr, rho = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
    return best, {"objective": obj, "cr": cr, "rho": rho, "history": hist}


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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(args.seed)

    price_map = parse_diesel_map(args.diesel_map)
    df_raw = pd.read_csv(args.csv)
    metrics_df = aggregate_metrics_by_alternative(
        df_raw, diesel_price_per_liter=args.diesel_price, diesel_price_map=price_map
    )
    metrics_df = metrics_df.dropna(axis=1, how="all")
    base_res = compute_profile_results(metrics_df, profile_name="base", profile_weights=BASE_WEIGHTS, fuzziness=args.fuzziness, vikor_v=args.vikor_v)
    baseline_ranks = base_res["fuzzy_topsis_rank"]
    baseline_obj, baseline_cr, baseline_rho = evaluate(BASE_WEIGHTS, metrics_df, baseline_ranks, args.fuzziness, args.vikor_v)

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
            cfg_dir = args.out_dir / algo / cfg_name
            cfg_dir.mkdir(parents=True, exist_ok=True)

            records = []
            best_global = None
            best_obj = float("inf")
            histories: List[List[float]] = []

            for _ in range(args.runs):
                seed_run = int(rng_master.integers(0, 1_000_000_000))
                rng = np.random.default_rng(seed_run)
                best_w, info = run_algo(algo, cfg, metrics_df, baseline_ranks, args.fuzziness, args.vikor_v, rng)
                record = {
                    "seed": seed_run,
                    "objective": info["objective"],
                    "rho": info["rho"],
                    "cr": info["cr"],
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
            t_stat, p_val = try_ttest(objs.tolist(), baseline_obj)
            stats_row = {
                "objective_mean": float(np.mean(objs)),
                "objective_std": float(np.std(objs, ddof=0)),
                "rho_mean": float(np.mean(rhos)),
                "rho_std": float(np.std(rhos, ddof=0)),
                "cr_mean": float(np.mean(crs)),
                "cr_std": float(np.std(crs, ddof=0)),
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

            summary_global.append(
                {
                    "algorithm": algo,
                    "config": cfg_name,
                    "objective_best": best_obj,
                    "objective_mean": stats_row["objective_mean"],
                    "objective_std": stats_row["objective_std"],
                    "rho_mean": stats_row["rho_mean"],
                    "cr_mean": stats_row["cr_mean"],
                    "p_value_vs_baseline": p_val,
                    "significant_vs_baseline": stats_row["significant_vs_baseline"],
                    "better_than_baseline": stats_row["better_than_baseline"],
                    **{f"w_{k}": v for k, v in best_global.items() if k.startswith("w_")},
                    "seed_best": best_global["seed"] if best_global else None,
                }
            )

    pd.DataFrame(summary_global).to_csv(args.out_dir / "summary_global.csv", index=False, float_format="%.10f")
    print("Concluido. Resultados em", args.out_dir)


if __name__ == "__main__":
    print("Iniciando optimize_weights_neighborhood.py ...")
    main()
