"""
Aplica metodos multicriterio (Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA) nas
alternativas agregadas a partir do CSV processado do REopt.

Fluxo:
1) Le o CSV consolidado (dados_preprocessados/reopt_ALL_blocks_v3_8.csv)
2) Usa build_ahp_structure.aggregate_metrics_by_alternative para obter
   metricas medias por alternativa (C1 Diesel-only, C2 PV + Battery, C3 Hibrido)
3) Calcula pesos por perfil (economic, sustainable, resilient, social)
   e distribui os pesos entre as metricas de cada criterio
4) Calcula scores e ranks por metodo para cada perfil
5) Salva CSV por perfil + um JSON resumo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from build_ahp_structure import (
    CRITERIA_TREE,
    DATA_PATH,
    PROFILES,
    aggregate_metrics_by_alternative,
    parse_diesel_map,
)


def detect_directions(columns: List[str]) -> Dict[str, str]:
    """
    Define direcoes (beneficio = max, custo = min) para cada metrica.
    """
    directions: Dict[str, str] = {}
    for col in columns:
        low_col = col.lower()
        if any(k in low_col for k in ["cost", "lcoe", "emission", "fossil", "tlcc"]):
            directions[col] = "min"
        elif "diesel_cost_share" in low_col:
            directions[col] = "min"
        elif "percent_load_target" in low_col:
            directions[col] = "max"
        else:
            directions[col] = "max"
    return directions


def map_metric_to_criterion(metric: str) -> str:
    low = metric.lower()
    if any(k in low for k in ["lcoe", "tlcc", "fuel_cost"]):
        return "cost"
    if any(k in low for k in ["emission", "fossil"]):
        return "emissions"
    if "diesel_cost_share" in low:
        return "reliability"
    if "percent_load_target" in low:
        return "social"
    return "social"  # fallback para social se nada combinar


def distribute_weights(metrics: List[str], profile_weights: Dict[str, float]) -> pd.Series:
    """
    Distribui o peso de cada criterio entre as metricas pertencentes a ele.
    """
    by_criterion: Dict[str, List[str]] = {}
    for m in metrics:
        crit = map_metric_to_criterion(m)
        by_criterion.setdefault(crit, []).append(m)

    weights = {}
    for crit, cols in by_criterion.items():
        base = float(profile_weights.get(crit, 0.0))
        if not cols:
            continue
        share = base / len(cols) if len(cols) else 0.0
        for c in cols:
            weights[c] = share

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        weights = {k: 0.0 for k in metrics}
    return pd.Series(weights)


def normalize_matrix(df: pd.DataFrame, directions: Dict[str, str]) -> pd.DataFrame:
    """
    Normalizacao vetorial diferenciando beneficio (max) e custo (min).
    """
    V = df.copy().astype(float)
    for c in V.columns:
        col = V[c].values.astype(float)
        denom = np.sqrt(np.nansum(col ** 2))
        if denom == 0:
            V[c] = np.nan
            continue
        norm = col / denom
        if directions.get(c, "max") == "min":
            norm = 1 - norm
        V[c] = norm
    return V


def fuzzy_topsis(df: pd.DataFrame, weights: pd.Series, directions: Dict[str, str], fuzziness: float = 0.05) -> pd.Series:
    """
    Implementacao simples de Fuzzy-TOPSIS com numeros triangulares em torno de cada valor.
    """
    X = df.copy().astype(float)

    def tri(x):
        if np.isnan(x):
            return (np.nan, np.nan, np.nan)
        d = abs(x) * float(fuzziness)
        return (x - d, x, x + d)

    F = {c: X[c].apply(tri).tolist() for c in X.columns}

    def ideal(col_tris, direction: str, which: str):
        vals = [t for t in col_tris if not any(np.isnan(t))]
        if not vals:
            return (np.nan, np.nan, np.nan)
        l = [t[0] for t in vals]
        m = [t[1] for t in vals]
        u = [t[2] for t in vals]
        if direction == "max":
            return (max(l), max(m), max(u)) if which == "pos" else (min(l), min(m), min(u))
        return (min(l), min(m), min(u)) if which == "pos" else (max(l), max(m), max(u))

    ideals = {}
    for c in X.columns:
        ideals[c] = {"pos": ideal(F[c], directions[c], "pos"), "neg": ideal(F[c], directions[c], "neg")}

    def dist(t1, t2):
        if any(np.isnan(t1)) or any(np.isnan(t2)):
            return np.nan
        return np.sqrt(((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2 + (t1[2] - t2[2]) ** 2) / 3.0)

    scores = []
    for idx in X.index:
        dp, dn = [], []
        for c in X.columns:
            t = F[c][list(X.index).index(idx)]
            w = float(weights.get(c, 0.0))
            dp.append(w * dist(t, ideals[c]["pos"]))
            dn.append(w * dist(t, ideals[c]["neg"]))
        dp_val = np.nansum(dp)
        dn_val = np.nansum(dn)
        scores.append(dn_val / (dp_val + dn_val + 1e-12))
    return pd.Series(scores, index=X.index)


def vikor(df: pd.DataFrame, weights: pd.Series, directions: Dict[str, str], v: float = 0.5) -> pd.Series:
    best = {}
    worst = {}
    for c in df.columns:
        col = df[c].astype(float)
        if directions[c] == "max":
            best[c] = np.nanmax(col)
            worst[c] = np.nanmin(col)
        else:
            best[c] = np.nanmin(col)
            worst[c] = np.nanmax(col)

    S, R = [], []
    for _, row in df.iterrows():
        s_terms = []
        r_terms = []
        for c in df.columns:
            denom = worst[c] - best[c]
            if denom == 0:
                s_terms.append(0.0)
                r_terms.append(0.0)
                continue
            val = abs((best[c] - row[c]) / denom)
            s_terms.append(weights.get(c, 0.0) * val)
            r_terms.append(weights.get(c, 0.0) * val)
        S.append(np.nansum(s_terms))
        R.append(np.nanmax(r_terms))

    S = np.array(S, dtype=float)
    R = np.array(R, dtype=float)
    S_min, S_max = np.nanmin(S), np.nanmax(S)
    R_min, R_max = np.nanmin(R), np.nanmax(R)

    Q = []
    for s, r in zip(S, R):
        q_s = 0 if S_max == S_min else (s - S_min) / (S_max - S_min)
        q_r = 0 if R_max == R_min else (r - R_min) / (R_max - R_min)
        Q.append(v * q_s + (1 - v) * q_r)
    return pd.Series(Q, index=df.index)


def copras(df: pd.DataFrame, weights: pd.Series, directions: Dict[str, str]) -> pd.Series:
    X = df.copy().astype(float)
    weighted = pd.DataFrame(index=X.index)
    for c in X.columns:
        col = X[c]
        sum_col = np.nansum(col)
        if sum_col == 0:
            weighted[c] = np.nan
            continue
        weighted[c] = (col / sum_col) * weights.get(c, 0.0)

    benefit = weighted.loc[:, [c for c in weighted.columns if directions[c] == "max"]].sum(axis=1, skipna=True)
    cost = weighted.loc[:, [c for c in weighted.columns if directions[c] == "min"]].sum(axis=1, skipna=True)
    scores = benefit + (cost.min() / (cost + 1e-12))
    return scores


def moora(df: pd.DataFrame, weights: pd.Series, directions: Dict[str, str]) -> pd.Series:
    X = df.copy().astype(float)
    for c in X.columns:
        denom = np.sqrt(np.nansum(X[c].astype(float) ** 2))
        X[c] = X[c] / denom if denom else np.nan

    weighted = X.mul(weights, axis=1)
    benefit = weighted.loc[:, [c for c in weighted.columns if directions[c] == "max"]].sum(axis=1, skipna=True)
    cost = weighted.loc[:, [c for c in weighted.columns if directions[c] == "min"]].sum(axis=1, skipna=True)
    return benefit - cost


def rank_scores(scores: pd.Series, higher_is_better: bool = True) -> pd.Series:
    return scores.rank(ascending=not higher_is_better, method="min")


def compute_profile_results(
    metrics_df: pd.DataFrame, profile_name: str, profile_weights: Dict[str, float], fuzziness: float, vikor_v: float
) -> pd.DataFrame:
    directions = detect_directions(metrics_df.columns.tolist())
    weights = distribute_weights(metrics_df.columns.tolist(), profile_weights)

    fz = fuzzy_topsis(metrics_df, weights, directions, fuzziness=fuzziness)
    vk = vikor(metrics_df, weights, directions, v=vikor_v)
    cp = copras(metrics_df, weights, directions)
    mo = moora(metrics_df, weights, directions)

    result = pd.DataFrame(
        {
            "fuzzy_topsis_score": fz,
            "fuzzy_topsis_rank": rank_scores(fz, higher_is_better=True),
            "vikor_score": vk,
            "vikor_rank": rank_scores(vk, higher_is_better=False),
            "copras_score": cp,
            "copras_rank": rank_scores(cp, higher_is_better=True),
            "moora_score": mo,
            "moora_rank": rank_scores(mo, higher_is_better=True),
        }
    )
    result.index.name = "Alternative"
    result.sort_values("fuzzy_topsis_rank", inplace=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aplica metodos MCDM nas alternativas agregadas por AHP.")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="Caminho para dados_preprocessados/reopt_ALL_blocks_v3_8.csv")
    parser.add_argument("--diesel-price", type=float, default=1.0, help="Preco do diesel (USD/L) default")
    parser.add_argument("--diesel-map", type=str, help="Mapa Region:preco (ex: Accra:0.95,Lusaka:1.16,Lodwar:0.85)")
    parser.add_argument("--fuzziness", type=float, default=0.05, help="Fator de fuzzificacao para Fuzzy-TOPSIS (0-1)")
    parser.add_argument("--vikor-v", type=float, default=0.5, help="Parametro v no VIKOR (0-1)")
    parser.add_argument("--out-dir", type=Path, default=Path("ahp_mcdm_results"), help="Diretorio de saida")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv)
    price_map = parse_diesel_map(args.diesel_map)
    metrics_df = aggregate_metrics_by_alternative(
        df_raw, diesel_price_per_liter=args.diesel_price, diesel_price_map=price_map
    )
    metrics_df = metrics_df.dropna(axis=1, how="all")

    summary = {
        "source_csv": str(args.csv),
        "diesel_price_per_liter": args.diesel_price,
        "profiles": {},
        "criteria_tree": CRITERIA_TREE,
    }

    for profile_name, profile_weights in PROFILES.items():
        result = compute_profile_results(metrics_df, profile_name, profile_weights, args.fuzziness, args.vikor_v)
        out_csv = args.out_dir / f"mcdm_{profile_name}.csv"
        result.to_csv(out_csv)
        summary["profiles"][profile_name] = {
            "weights": profile_weights,
            "csv": str(out_csv),
            "winner_by_fuzzy_topsis": result.index[result["fuzzy_topsis_rank"] == 1].tolist(),
        }
        print(f"[{profile_name}] vencedor (Fuzzy-TOPSIS): {summary['profiles'][profile_name]['winner_by_fuzzy_topsis']}")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Resultados salvos em {args.out_dir} (JSON: {summary_path})")


if __name__ == "__main__":
    main()
