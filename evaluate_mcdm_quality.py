"""
Avaliacao de desempenho de pesos MCDM comparando com pesos base.

Metricas avaliadas (para cada conjunto de pesos):
- Robustez de ranking: media das correlacoes de Spearman entre ranks dos metodos (TOPSIS/VIKOR/COPRAS/MOORA).
- Regret/utility: media do regret (gap do melhor score Fuzzy-TOPSIS para os demais).
- Dominancia/Pareto: indicador se o vencedor pelo Fuzzy-TOPSIS esta na fronteira Pareto.
- Consistencia (CR proxy): desvio da soma dos pesos em relacao a 1.
- Estabilidade a ruido: fracao de vezes que o vencedor se mantem sob ruido e variacao media de rank.
- Validacao cruzada/ruido: igual a estabilidade, mas em "folds" de cenarios ruidosos.

Saidas: CSV/JSON com as metricas para pesos base e pesos candidatos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from apply_mcdm_profiles import compute_profile_results
from build_ahp_structure import DATA_PATH, aggregate_metrics_by_alternative


BASE_WEIGHTS = {"cost": 0.40, "emissions": 0.30, "reliability": 0.20, "social": 0.10}
METHOD_RANK_COLS = ["fuzzy_topsis_rank", "vikor_rank", "copras_rank", "moora_rank"]
METHOD_SCORE_COL = "fuzzy_topsis_score"


def read_weights_file(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    row = df.iloc[0]
    weights = {}
    for k in ("cost", "emissions", "reliability", "social"):
        if k in row:
            weights[k] = float(row[k])
        elif f"w_{k}" in row:
            weights[k] = float(row[f"w_{k}"])
    return weights


def cr_proxy(weights: Dict[str, float]) -> float:
    return abs(sum(weights.values()) - 1.0)


def pairwise_spearman(ranks: pd.DataFrame) -> float:
    vals = []
    cols = ranks.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = ranks[cols[i]].astype(float)
            b = ranks[cols[j]].astype(float)
            n = len(a)
            if n < 2:
                continue
            diff = a - b
            num = 6 * np.sum(diff**2)
            denom = n * (n**2 - 1)
            vals.append(1 - num / denom if denom else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def regret(scores: pd.Series) -> float:
    best = scores.max()
    return float((best - scores).mean())


def is_pareto_front(metrics_df: pd.DataFrame) -> List[bool]:
    cols = metrics_df.columns.tolist()
    directions = {c: ("min" if "cost" in c.lower() or "emission" in c.lower() else "max") for c in cols}
    data = metrics_df.values
    n = data.shape[0]
    mask = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal = True
            strictly_better = False
            for k, c in enumerate(cols):
                if directions[c] == "max":
                    if data[j, k] > data[i, k] + 1e-12:
                        strictly_better = True
                    elif data[j, k] < data[i, k] - 1e-12:
                        better_or_equal = False
                        break
                else:
                    if data[j, k] < data[i, k] - 1e-12:
                        strictly_better = True
                    elif data[j, k] > data[i, k] + 1e-12:
                        better_or_equal = False
                        break
            if better_or_equal and strictly_better:
                mask[i] = False
                break
    return mask


def add_noise(df: pd.DataFrame, rng: np.random.Generator, noise_level: float) -> pd.DataFrame:
    noisy = df.copy().astype(float)
    for c in noisy.columns:
        perturb = rng.normal(0.0, noise_level, size=len(noisy))
        noisy[c] = noisy[c] * (1 + perturb)
    return noisy


def stability_metrics(
    metrics_df: pd.DataFrame,
    weights: Dict[str, float],
    fuzziness: float,
    vikor_v: float,
    baseline_winner: str,
    runs: int = 50,
    noise_level: float = 0.02,
    seed: int = 123,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    winners = []
    rank_changes = []
    for _ in range(runs):
        noisy = add_noise(metrics_df, rng, noise_level)
        res = compute_profile_results(noisy, profile_name="stab", profile_weights=weights, fuzziness=fuzziness, vikor_v=vikor_v)
        winners.append(res.index[res["fuzzy_topsis_rank"] == 1][0] if not res.empty else None)
        rank_changes.append((res["fuzzy_topsis_rank"] - res["fuzzy_topsis_rank"].rank()).abs().mean() if not res.empty else np.nan)
    win_rate = float(np.mean([w == baseline_winner for w in winners if w is not None])) if winners else float("nan")
    mean_rank_change = float(np.nanmean(rank_changes)) if rank_changes else float("nan")
    return win_rate, mean_rank_change


def evaluate_weights(
    metrics_df: pd.DataFrame,
    baseline_ranks: pd.Series,
    weights: Dict[str, float],
    fuzziness: float,
    vikor_v: float,
    seed: int,
) -> Dict[str, float]:
    result = compute_profile_results(metrics_df, profile_name="eval", profile_weights=weights, fuzziness=fuzziness, vikor_v=vikor_v)
    ranks = result[METHOD_RANK_COLS]
    robust = pairwise_spearman(ranks)
    reg = regret(result[METHOD_SCORE_COL])

    pareto_mask = is_pareto_front(metrics_df)
    winner = result.index[result["fuzzy_topsis_rank"] == 1][0]
    win_is_pareto = bool(pareto_mask[list(metrics_df.index).index(winner)])

    win_rate, mean_rank_change = stability_metrics(
        metrics_df, weights, fuzziness, vikor_v, baseline_winner=winner, runs=50, noise_level=0.02, seed=seed
    )

    # validacao cruzada via ruido (mesma rotina, diferente seed)
    val_rate, val_rank_change = stability_metrics(
        metrics_df, weights, fuzziness, vikor_v, baseline_winner=winner, runs=30, noise_level=0.05, seed=seed + 999
    )

    return {
        "robust_spearman": robust,
        "regret_mean": reg,
        "winner_pareto": float(win_is_pareto),
        "cr_proxy": cr_proxy(weights),
        "stability_win_rate": win_rate,
        "stability_rank_change": mean_rank_change,
        "cv_win_rate": val_rate,
        "cv_rank_change": val_rank_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia pesos MCDM com multiplas metricas de desempenho.")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="CSV consolidado (dados_preprocessados/reopt_ALL_blocks_v3_8.csv)")
    parser.add_argument("--diesel-price", type=float, default=1.2, help="Preco do diesel (USD/L)")
    parser.add_argument("--weights", type=Path, nargs="+", required=False, help="Arquivos CSV com pesos (colunas cost/emissions/reliability/social)")
    parser.add_argument("--fuzziness", type=float, default=0.05, help="Fator fuzzy")
    parser.add_argument("--vikor-v", type=float, default=0.5, help="Parametro v do VIKOR")
    parser.add_argument("--seed", type=int, default=42, help="Semente base")
    parser.add_argument("--out", type=Path, default=Path("eval_quality.csv"), help="Saida CSV")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.csv)
    metrics_df = aggregate_metrics_by_alternative(df_raw, diesel_price_per_liter=args.diesel_price)
    metrics_df = metrics_df.dropna(axis=1, how="all")

    # baseline ranks para Spearman com metodos
    base_res = compute_profile_results(metrics_df, profile_name="base", profile_weights=BASE_WEIGHTS, fuzziness=args.fuzziness, vikor_v=args.vikor_v)
    baseline_ranks = base_res["fuzzy_topsis_rank"]

    rows = []
    # sempre avalia base
    rows.append(
        {
            "label": "baseline",
            **evaluate_weights(metrics_df, baseline_ranks, BASE_WEIGHTS, args.fuzziness, args.vikor_v, seed=args.seed),
            **{f"w_{k}": v for k, v in BASE_WEIGHTS.items()},
        }
    )

    if args.weights:
        for w_path in args.weights:
            weights = read_weights_file(w_path)
            rows.append(
                {
                    "label": w_path.stem,
                    **evaluate_weights(metrics_df, baseline_ranks, weights, args.fuzziness, args.vikor_v, seed=args.seed),
                    **{f"w_{k}": v for k, v in weights.items()},
                }
            )

    pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.10f")
    print(f"Avaliacao salva em {args.out}")


if __name__ == "__main__":
    main()
