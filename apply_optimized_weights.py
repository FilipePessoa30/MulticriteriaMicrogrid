"""
Aplica os metodos MCDM (Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA) usando pesos otimizados
encontrados pelos scripts de busca (genetico e vizinhanca).

Coleta de pesos:
- --auto: busca best_weights.csv / best_*weights.csv / best_*.json / summary_global.csv em
  weight_optimization/ e neighborhood_results/.
- --weights: arquivos CSV/JSON explicitamente informados.

Saidas:
- CSV por conjunto de pesos em `optimized_mcdm_results/`
- Summary JSON com o vencedor (Fuzzy-TOPSIS) de cada conjunto.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from apply_mcdm_profiles import compute_profile_results
from build_ahp_structure import DATA_PATH, aggregate_metrics_by_alternative, parse_diesel_map


def extract_weight_rows(df: pd.DataFrame) -> List[Dict[str, float]]:
    cols = df.columns
    keys = []
    for base in ("cost", "emissions", "reliability", "social"):
        if base in cols:
            keys.append(base)
        elif f"w_{base}" in cols:
            keys.append(f"w_{base}")
    if len(keys) < 4:
        return []
    rows = []
    for _, row in df.iterrows():
        weights = {}
        for base in ("cost", "emissions", "reliability", "social"):
            if base in row:
                weights[base] = float(row[base])
            elif f"w_{base}" in row:
                weights[base] = float(row[f"w_{base}"])
        rows.append(weights)
    return rows


def read_weights_file(path: Path) -> List[Dict[str, float]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        weights = {}
        for k in ("cost", "emissions", "reliability", "social"):
            if k in data:
                weights[k] = float(data[k])
            elif f"w_{k}" in data:
                weights[k] = float(data[f"w_{k}"])
        return [weights] if weights else []
    df = pd.read_csv(path)
    return extract_weight_rows(df)


def label_for_path(path: Path, roots: List[Path]) -> str:
    for root in roots:
        try:
            rel = path.relative_to(root)
            parts = list(rel.parts)
            if parts[-1].startswith("best_"):
                parts[-1] = parts[-1].removeprefix("best_").removesuffix(path.suffix)
            else:
                parts[-1] = parts[-1].removesuffix(path.suffix)
            return "_".join(parts)
        except ValueError:
            continue
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Aplica MCDM com pesos otimizados (VNS/Tabu/ILS/Hybrid).")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="CSV consolidado (dados_preprocessados/reopt_ALL_blocks_v3_8.csv)")
    parser.add_argument("--diesel-price", type=float, default=1.2, help="Preco do diesel default (USD/L)")
    parser.add_argument("--diesel-map", type=str, help="Mapa Region:preco (ex: Accra:0.95,Lusaka:1.16,Lodwar:0.85)")
    parser.add_argument("--fuzziness", type=float, default=0.05, help="Fator fuzzy para Fuzzy-TOPSIS")
    parser.add_argument("--vikor-v", type=float, default=0.5, help="Parametro v do VIKOR")
    parser.add_argument("--weights", type=Path, nargs="+", help="Arquivos de pesos adicionais (CSV/JSON com colunas cost/emissions/reliability/social ou w_*)")
    parser.add_argument("--auto", action="store_true", help="Coletar pesos automaticamente em weight_optimization/ e neighborhood_results/")
    parser.add_argument("--out-dir", type=Path, default=Path("optimized_mcdm_results"), help="Diretorio de saida")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    price_map = parse_diesel_map(args.diesel_map)
    df_raw = pd.read_csv(args.csv)
    metrics_df = aggregate_metrics_by_alternative(
        df_raw, diesel_price_per_liter=args.diesel_price, diesel_price_map=price_map
    )
    metrics_df = metrics_df.dropna(axis=1, how="all")

    weight_files: List[Path] = []
    roots = [Path("weight_optimization"), Path("neighborhood_results")]
    if args.auto:
        patterns = ["best_weights.csv", "best_*weights.csv", "best_*.json", "summary_global.csv"]
        for root in roots:
            if root.exists():
                for pat in patterns:
                    weight_files.extend(root.rglob(pat))
    if args.weights:
        weight_files.extend(args.weights)

    if not weight_files:
        print("Nenhum arquivo de pesos encontrado. Use --auto ou --weights.")
        return

    seen = set()
    summary = {
        "source_csv": str(args.csv),
        "diesel_price_per_liter": args.diesel_price,
        "fuzziness": args.fuzziness,
        "vikor_v": args.vikor_v,
        "results": {},
    }

    for w_path in weight_files:
        if not w_path.exists():
            continue
        key = str(w_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        weights_list = read_weights_file(w_path)
        if not weights_list:
            continue
        base_label = label_for_path(w_path, roots)
        for idx, weights in enumerate(weights_list):
            label = base_label if len(weights_list) == 1 else f"{base_label}_{idx}"
            out_csv = args.out_dir / f"mcdm_{label}.csv"
            result = compute_profile_results(metrics_df, profile_name=label, profile_weights=weights, fuzziness=args.fuzziness, vikor_v=args.vikor_v)
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
