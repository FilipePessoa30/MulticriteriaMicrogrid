"""
Otimiza pesos (cost, emissions, reliability, social) usando metaheuristicas de vizinhanca:
- VNS (Variable Neighborhood Search)
- Tabu Search
- Iterated Local Search (ILS)
- Hybrid VNS + VND/Tabu leve

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


def perturb(weights: Dict[str, float], rng: np.random.Generator, step: float) -> Dict[str, float]:
    w_vec = np.array(list(weights.values()))
    noise = rng.normal(0, step, size=w_vec.shape)
    return normalize_weights(w_vec + noise)


def run_vns(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, neighborhoods: List[float], iters_per_neigh: int) -> Tuple[Dict[str, float], List[float]]:
    current = random_weights(rng)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
    history = [best_obj]
    for step in neighborhoods:
        improved = False
        for _ in range(iters_per_neigh):
            candidate = perturb(current, rng, step)
            obj, _, _ = evaluate(candidate, metrics_df, baseline_ranks, fuzziness, vikor_v)
            history.append(obj)
            if obj < best_obj:
                best = candidate
                best_obj = obj
                current = candidate
                improved = True
                break
        if not improved:
            # move para proxima vizinhanca
            continue
    return best, history


def run_tabu(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, step: float, iters: int, tabu_size: int) -> Tuple[Dict[str, float], List[float]]:
    current = random_weights(rng)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
    tabu: List[Tuple[float, ...]] = []
    history = [best_obj]
    for _ in range(iters):
        candidate = perturb(current, rng, step)
        key = tuple(round(v, 6) for v in candidate.values())
        tries = 0
        while key in tabu and tries < 5:
            candidate = perturb(current, rng, step)
            key = tuple(round(v, 6) for v in candidate.values())
            tries += 1
        obj, _, _ = evaluate(candidate, metrics_df, baseline_ranks, fuzziness, vikor_v)
        history.append(obj)
        if obj < best_obj:
            best = candidate
            best_obj = obj
        current = candidate if obj <= best_obj else current
        tabu.append(key)
        if len(tabu) > tabu_size:
            tabu.pop(0)
    return best, history


def run_ils(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, step: float, iters: int, perturb_jump: float) -> Tuple[Dict[str, float], List[float]]:
    current = random_weights(rng)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
    history = [best_obj]
    for _ in range(iters):
        # local search pequena
        candidate = perturb(current, rng, step)
        obj, _, _ = evaluate(candidate, metrics_df, baseline_ranks, fuzziness, vikor_v)
        history.append(obj)
        if obj < best_obj:
            best = candidate
            best_obj = obj
            current = candidate
        else:
            # perturba salto
            current = perturb(best, rng, perturb_jump)
    return best, history


def run_hybrid(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator) -> Tuple[Dict[str, float], List[float]]:
    # VNS pequena + tabu leve
    neighborhoods = [0.05, 0.1, 0.2]
    current, history = run_vns(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, neighborhoods, iters_per_neigh=5)
    best = current
    best_obj, _, _ = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
    # tabu refinamento
    refine, hist2 = run_tabu(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, step=0.05, iters=20, tabu_size=10)
    history.extend(hist2)
    obj, _, _ = evaluate(refine, metrics_df, baseline_ranks, fuzziness, vikor_v)
    if obj < best_obj:
        best = refine
    return best, history


def run_algo(name: str, metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator) -> Tuple[Dict[str, float], Dict]:
    if name == "VNS":
        best, hist = run_vns(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, neighborhoods=[0.05, 0.1, 0.2, 0.3], iters_per_neigh=10)
    elif name == "TS":
        best, hist = run_tabu(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, step=0.08, iters=60, tabu_size=15)
    elif name == "ILS":
        best, hist = run_ils(metrics_df, baseline_ranks, fuzziness, vikor_v, rng, step=0.05, iters=60, perturb_jump=0.15)
    else:  # Hybrid
        best, hist = run_hybrid(metrics_df, baseline_ranks, fuzziness, vikor_v, rng)
    obj, cr, rho = evaluate(best, metrics_df, baseline_ranks, fuzziness, vikor_v)
    return best, {"objective": obj, "cr": cr, "rho": rho, "history": hist}


def main() -> None:
    parser = argparse.ArgumentParser(description="Metaheuristicas de vizinhanca para otimizar pesos AHP/MCDM.")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="CSV consolidado")
    parser.add_argument("--diesel-price", type=float, default=1.2, help="Preco do diesel (USD/L)")
    parser.add_argument("--diesel-map", type=str, help="Mapa Region:preco (ex: Accra:0.95,Lusaka:1.16,Lodwar:0.85)")
    parser.add_argument("--fuzziness", type=float, default=0.05, help="Fator fuzzy")
    parser.add_argument("--vikor-v", type=float, default=0.5, help="Parametro v VIKOR")
    parser.add_argument("--runs", type=int, default=20, help="Execucoes por algoritmo")
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

    algos = ["VNS", "TS", "ILS", "HYBRID"]
    summary = {}

    for algo in algos:
        records = []
        best_global = None
        best_obj = float("inf")
        for _ in range(args.runs):
            seed_run = int(rng_master.integers(0, 1_000_000_000))
            rng = np.random.default_rng(seed_run)
            best_w, info = run_algo(algo, metrics_df, baseline_ranks, args.fuzziness, args.vikor_v, rng)
            record = {
                "seed": seed_run,
                "objective": info["objective"],
                "rho": info["rho"],
                "cr": info["cr"],
                **{f"w_{k}": v for k, v in best_w.items()},
            }
            records.append(record)
            if info["objective"] < best_obj:
                best_obj = info["objective"]
                best_global = record
        df = pd.DataFrame(records)
        algo_dir = args.out_dir / algo
        algo_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(algo_dir / f"runs_{algo}.csv", index=False, float_format="%.10f")
        (algo_dir / f"best_{algo}.json").write_text(json.dumps(best_global, indent=2), encoding="utf-8")
        summary[algo] = {"best": best_global, "runs": len(records)}

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Concluido. Resultados em", args.out_dir)


if __name__ == "__main__":
    print("Iniciando optimize_weights_neighborhood.py ...")
    main()
