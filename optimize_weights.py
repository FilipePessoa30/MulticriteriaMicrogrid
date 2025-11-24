"""
Otimiza pesos dos criterios (cost, emissions, reliability, social) com PSO, GA e SA
usando multiplas configuracoes por metaheuristica e 30 execucoes por configuracao.

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
from build_ahp_structure import DATA_PATH, aggregate_metrics_by_alternative


# Pesos iniciais da literatura (AHP)
BASE_WEIGHTS = {"cost": 0.40, "emissions": 0.30, "reliability": 0.20, "social": 0.10}
ALPHA = 0.5  # peso da consistencia
BETA = 0.5   # peso da correlacao de ranking (rho)


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


def evaluate(weights: Dict[str, float], metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float) -> Tuple[float, float, float]:
    cr = cr_proxy(weights)
    result = compute_profile_results(metrics_df, profile_name="opt", profile_weights=weights, fuzziness=fuzziness, vikor_v=vikor_v)
    rho = spearman_corr(result["fuzzy_topsis_rank"], baseline_ranks)
    objective = ALPHA * cr + BETA * (1 - rho)
    return float(objective), float(cr), float(rho)


def crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    alpha = rng.uniform(0.2, 0.8)
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


def mutate(vec: np.ndarray, rng: np.random.Generator, rate: float = 0.1) -> np.ndarray:
    noise = rng.normal(0, rate, size=vec.shape)
    mutated = np.clip(vec + noise, 1e-9, None)
    return mutated / mutated.sum()


def run_pso(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, config: Dict) -> Tuple[Dict[str, float], List[float]]:
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
            obj, _, _ = evaluate(weights, metrics_df, baseline_ranks, fuzziness, vikor_v)
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


def run_ga(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, config: Dict) -> Tuple[Dict[str, float], List[float]]:
    dim = 4
    pop_size = config.get("pop_size", 30)
    generations = config.get("generations", 50)
    mut_rate = config.get("mutation_rate", 0.1)

    population = rng.dirichlet(np.ones(dim), size=pop_size)
    history = []

    def fitness(individual: np.ndarray) -> float:
        weights = normalize_weights(individual)
        obj, _, _ = evaluate(weights, metrics_df, baseline_ranks, fuzziness, vikor_v)
        return obj

    for _ in range(generations):
        scores = np.array([fitness(ind) for ind in population])
        best_idx = int(np.argmin(scores))
        best = population[best_idx].copy()
        history.append(scores[best_idx])

        new_pop = [best]
        while len(new_pop) < pop_size:
            parents_idx = rng.choice(pop_size, size=2, replace=False)
            p1, p2 = population[parents_idx[0]], population[parents_idx[1]]
            c1, c2 = crossover(p1, p2, rng)
            if rng.random() < mut_rate:
                c1 = mutate(c1, rng, rate=mut_rate)
            if rng.random() < mut_rate:
                c2 = mutate(c2, rng, rate=mut_rate)
            new_pop.extend([c1, c2])
        population = np.array(new_pop[:pop_size])

    final_scores = np.array([fitness(ind) for ind in population])
    best_idx = int(np.argmin(final_scores))
    return normalize_weights(population[best_idx]), history


def run_sa(metrics_df: pd.DataFrame, baseline_ranks: pd.Series, fuzziness: float, vikor_v: float, rng: np.random.Generator, config: Dict) -> Tuple[Dict[str, float], List[float]]:
    steps = config.get("steps", 200)
    temp = config.get("temp", 1.0)
    cooling = config.get("cooling", 0.97)
    neighbor_rate = config.get("neighbor_rate", 0.05)

    current = rng.dirichlet(np.ones(4))
    current_w = normalize_weights(current)
    current_obj, _, _ = evaluate(current_w, metrics_df, baseline_ranks, fuzziness, vikor_v)
    best = current.copy()
    best_obj = current_obj
    history = [best_obj]

    for _ in range(steps):
        neighbor = mutate(current, rng, rate=neighbor_rate)
        neigh_w = normalize_weights(neighbor)
        neigh_obj, _, _ = evaluate(neigh_w, metrics_df, baseline_ranks, fuzziness, vikor_v)

        if neigh_obj < current_obj or rng.random() < np.exp(-(neigh_obj - current_obj) / max(temp, 1e-6)):
            current = neighbor
            current_obj = neigh_obj

        if neigh_obj < best_obj:
            best = neighbor
            best_obj = neigh_obj

        history.append(best_obj)
        temp *= cooling

    return normalize_weights(best), history


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
        }
        row.update({f"w_{k}": v for k, v in r["weights"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.10f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Otimiza pesos AHP com PSO/GA/SA em multiplas configuracoes.")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="CSV consolidado (dados_preprocessados/reopt_ALL_blocks_v3_8.csv)")
    parser.add_argument("--diesel-price", type=float, default=1.2, help="Preco do diesel (USD/L)")
    parser.add_argument("--fuzziness", type=float, default=0.05, help="Fator fuzzy para Fuzzy-TOPSIS")
    parser.add_argument("--vikor-v", type=float, default=0.5, help="Parametro v do VIKOR")
    parser.add_argument("--runs", type=int, default=30, help="Numero de execucoes por configuracao")
    parser.add_argument("--seed", type=int, default=123, help="Semente base")
    parser.add_argument("--out-dir", type=Path, default=Path("weight_optimization"), help="Diretorio de saida")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(args.seed)

    # Grids de configuracao (>=8 cada)
    pso_grid = [
        {"name": "pso_1", "particles": 15, "iterations": 30, "w": 0.7, "c1": 1.4, "c2": 1.4},
        {"name": "pso_2", "particles": 25, "iterations": 40, "w": 0.6, "c1": 1.6, "c2": 1.6},
        {"name": "pso_3", "particles": 35, "iterations": 50, "w": 0.8, "c1": 1.2, "c2": 1.6},
        {"name": "pso_4", "particles": 20, "iterations": 60, "w": 0.5, "c1": 2.0, "c2": 2.0},
        {"name": "pso_5", "particles": 30, "iterations": 35, "w": 0.9, "c1": 1.0, "c2": 1.0},
        {"name": "pso_6", "particles": 18, "iterations": 45, "w": 0.65, "c1": 1.8, "c2": 1.3},
        {"name": "pso_7", "particles": 28, "iterations": 55, "w": 0.75, "c1": 1.3, "c2": 1.8},
        {"name": "pso_8", "particles": 22, "iterations": 70, "w": 0.55, "c1": 2.1, "c2": 1.7},
    ]
    ga_grid = [
        {"name": "ga_1", "pop_size": 20, "generations": 30, "mutation_rate": 0.05},
        {"name": "ga_2", "pop_size": 30, "generations": 40, "mutation_rate": 0.10},
        {"name": "ga_3", "pop_size": 40, "generations": 50, "mutation_rate": 0.15},
        {"name": "ga_4", "pop_size": 25, "generations": 60, "mutation_rate": 0.12},
        {"name": "ga_5", "pop_size": 35, "generations": 35, "mutation_rate": 0.08},
        {"name": "ga_6", "pop_size": 28, "generations": 45, "mutation_rate": 0.20},
        {"name": "ga_7", "pop_size": 32, "generations": 55, "mutation_rate": 0.07},
        {"name": "ga_8", "pop_size": 24, "generations": 70, "mutation_rate": 0.18},
    ]
    sa_grid = [
        {"name": "sa_1", "steps": 120, "temp": 1.0, "cooling": 0.95, "neighbor_rate": 0.05},
        {"name": "sa_2", "steps": 180, "temp": 1.5, "cooling": 0.96, "neighbor_rate": 0.07},
        {"name": "sa_3", "steps": 200, "temp": 2.0, "cooling": 0.97, "neighbor_rate": 0.05},
        {"name": "sa_4", "steps": 250, "temp": 1.2, "cooling": 0.94, "neighbor_rate": 0.08},
        {"name": "sa_5", "steps": 160, "temp": 0.8, "cooling": 0.93, "neighbor_rate": 0.06},
        {"name": "sa_6", "steps": 220, "temp": 1.8, "cooling": 0.92, "neighbor_rate": 0.04},
        {"name": "sa_7", "steps": 140, "temp": 1.0, "cooling": 0.90, "neighbor_rate": 0.10},
        {"name": "sa_8", "steps": 300, "temp": 2.2, "cooling": 0.95, "neighbor_rate": 0.05},
    ]

    # Dados agregados
    df_raw = pd.read_csv(args.csv)
    metrics_df = aggregate_metrics_by_alternative(df_raw, diesel_price_per_liter=args.diesel_price)
    metrics_df = metrics_df.dropna(axis=1, how="all")

    base_result = compute_profile_results(metrics_df, profile_name="base", profile_weights=BASE_WEIGHTS, fuzziness=args.fuzziness, vikor_v=args.vikor_v)
    baseline_ranks = base_result["fuzzy_topsis_rank"]
    baseline_obj, baseline_cr, baseline_rho = evaluate(BASE_WEIGHTS, metrics_df, baseline_ranks, args.fuzziness, args.vikor_v)

    global_summary_rows = []

    def run_config(algo: str, config: Dict) -> None:
        cfg_name = config["name"]
        cfg_dir = args.out_dir / algo / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        histories_all: Dict[str, List[List[float]]] = {algo: []}
        best_histories: Dict[str, List[float]] = {}
        objectives_all: Dict[str, List[float]] = {algo: []}
        run_records: List[Dict] = []

        for _ in range(args.runs):
            seed = int(master_rng.integers(0, 1_000_000_000))
            rng = np.random.default_rng(seed)

            if algo == "PSO":
                weights, hist = run_pso(metrics_df, baseline_ranks, args.fuzziness, args.vikor_v, rng, config)
            elif algo == "GA":
                weights, hist = run_ga(metrics_df, baseline_ranks, args.fuzziness, args.vikor_v, rng, config)
            else:
                weights, hist = run_sa(metrics_df, baseline_ranks, args.fuzziness, args.vikor_v, rng, config)

            obj, cr, rho = evaluate(weights, metrics_df, baseline_ranks, args.fuzziness, args.vikor_v)
            histories_all[algo].append(hist)
            objectives_all[algo].append(obj)
            run_records.append({"seed": seed, "weights": weights, "objective": obj, "cr": cr, "rho": rho, "history": hist})

        # salvar runs
        save_runs_table(run_records, cfg_dir / "runs.csv")

        # melhor execucao
        objs = np.array([r["objective"] for r in run_records])
        rhos = np.array([r["rho"] for r in run_records])
        crs = np.array([r["cr"] for r in run_records])
        best_idx = int(np.argmin(objs))
        best_weights = run_records[best_idx]["weights"]
        best_histories[algo] = run_records[best_idx]["history"]
        improvements = {
            "objective": run_records[best_idx]["objective"],
            "rho": run_records[best_idx]["rho"],
            "cr": run_records[best_idx]["cr"],
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
            "t_stat_vs_baseline": t_stat,
            "p_value_vs_baseline": p_val,
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
                "seed": improvements["seed"],
                **{f"w_{k}": v for k, v in best_weights.items()},
                "objective_mean": stats["objective_mean"],
                "objective_std": stats["objective_std"],
                "p_value_vs_baseline": stats["p_value_vs_baseline"],
            }
        )

    for cfg in pso_grid:
        run_config("PSO", cfg)
    for cfg in ga_grid:
        run_config("GA", cfg)
    for cfg in sa_grid:
        run_config("SA", cfg)

    pd.DataFrame(global_summary_rows).to_csv(args.out_dir / "summary_global.csv", index=False, float_format="%.10f")

    summary = {
        "source_csv": str(args.csv),
        "baseline_weights": BASE_WEIGHTS,
        "baseline_objective": baseline_obj,
        "grids": {"PSO": pso_grid, "GA": ga_grid, "SA": sa_grid},
        "runs_per_config": args.runs,
        "seed_base": args.seed,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Execucao completa. Resultados em", args.out_dir)


if __name__ == "__main__":
    print("Iniciando optimize_weights.py ...")
    main()
