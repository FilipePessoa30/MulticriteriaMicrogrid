"""
Aplica os metodos MCDM (Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA) usando os pesos
otimizados das metaheuristicas de vizinhanca/estocasticas.

Pesos predefinidos (os melhores regret_mean encontrados):
- VNS
- Tabu Search
- ILS
- Hybrid VNS+Tabu

Saidas:
- CSV por conjunto de pesos em `optimized_mcdm_results/`
- Summary JSON com o vencedor (Fuzzy-TOPSIS) de cada conjunto.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from apply_mcdm_profiles import compute_profile_results
from build_ahp_structure import DATA_PATH, aggregate_metrics_by_alternative


BEST_WEIGHTS: Dict[str, Dict[str, float]] = {
    "VNS_best": {
        "cost": 0.3206608381,
        "emissions": 0.0000000000,
        "reliability": 0.1560573560,
        "social": 0.5232818060,
    },
    "Tabu_best": {
        "cost": 0.5763681853,
        "emissions": 0.0048817464,
        "reliability": 0.2765960002,
        "social": 0.1421540680,
    },
    "ILS_best": {
        "cost": 0.3074530348,
        "emissions": 0.0181989677,
        "reliability": 0.2041180049,
        "social": 0.4702299925,
    },
    "Hybrid_best": {
        "cost": 0.3337491064,
        "emissions": 0.0312942730,
        "reliability": 0.1355886034,
        "social": 0.4993680173,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aplica MCDM com pesos otimizados (VNS/Tabu/ILS/Hybrid).")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="CSV consolidado (dados_preprocessados/reopt_ALL_blocks_v3_8.csv)")
    parser.add_argument("--diesel-price", type=float, default=1.2, help="Preco do diesel (USD/L)")
    parser.add_argument("--fuzziness", type=float, default=0.05, help="Fator fuzzy para Fuzzy-TOPSIS")
    parser.add_argument("--vikor-v", type=float, default=0.5, help="Parametro v do VIKOR")
    parser.add_argument("--out-dir", type=Path, default=Path("optimized_mcdm_results"), help="Diretorio de saida")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv)
    metrics_df = aggregate_metrics_by_alternative(df_raw, diesel_price_per_liter=args.diesel_price)
    metrics_df = metrics_df.dropna(axis=1, how="all")

    summary = {
        "source_csv": str(args.csv),
        "diesel_price_per_liter": args.diesel_price,
        "fuzziness": args.fuzziness,
        "vikor_v": args.vikor_v,
        "results": {},
    }

    for label, weights in BEST_WEIGHTS.items():
        result = compute_profile_results(metrics_df, profile_name=label, profile_weights=weights, fuzziness=args.fuzziness, vikor_v=args.vikor_v)
        out_csv = args.out_dir / f"mcdm_{label}.csv"
        result.to_csv(out_csv, float_format="%.10f")
        winner = result.index[result["fuzzy_topsis_rank"] == 1].tolist()
        summary["results"][label] = {
            "weights": weights,
            "csv": str(out_csv),
            "winner_fuzzy_topsis": winner,
        }
        print(f"[{label}] vencedor (Fuzzy-TOPSIS): {winner}")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Resultados salvos em {args.out_dir}")


if __name__ == "__main__":
    print("Iniciando apply_optimized_weights.py ...")
    main()
