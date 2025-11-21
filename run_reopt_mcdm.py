"""
Roda Fuzzy-TOPSIS e VIKOR diretamente no CSV consolidado do REopt
(`dados_preprocessados/reopt_ALL_blocks_v3_8.csv`) usando três critérios:
- custo: LCOE_$_per_kWh (fallback para REopt_LCC_$ se ausente)
- emissões (proxy): Fuel_cost_$ (quanto menor, melhor)
- confiabilidade: Percent_Load_Target (quanto maior, melhor)

Gera ranks por perfil (econômico/sustentável/resiliente) e um resumo de vencedores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


PROFILES = {
    "economic":    {"cost": 0.6,  "emissions": 0.2,  "reliability": 0.2},
    "sustainable": {"cost": 0.25, "emissions": 0.45, "reliability": 0.30},
    "resilient":   {"cost": 0.3,  "emissions": 0.1,  "reliability": 0.6},
}
directions = {"cost": "min", "emissions": "min", "reliability": "max"}


def fuzzy_topsis(df_values: pd.DataFrame, weights: Dict[str, float], fuzziness: float) -> pd.Series:
    X = df_values.copy().astype(float)

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
        l = [t[0] for t in vals]; m = [t[1] for t in vals]; u = [t[2] for t in vals]
        if direction == "max":
            return (max(l), max(m), max(u)) if which == "pos" else (min(l), min(m), min(u))
        else:
            return (min(l), min(m), min(u)) if which == "pos" else (max(l), max(m), max(u))

    ideals = {}
    for c in X.columns:
        ideals[c] = {"pos": ideal(F[c], directions[c], "pos"),
                     "neg": ideal(F[c], directions[c], "neg")}

    def dist(t1, t2):
        if any(np.isnan(t1)) or any(np.isnan(t2)):
            return np.nan
        return np.sqrt(((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2) / 3.0)

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


def topsis(df_values: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    V = df_values.copy().astype(float)
    # normalização vetorial
    for c in V.columns:
        col = V[c].values.astype(float)
        denom = np.sqrt(np.nansum(col ** 2))
        if denom == 0 or np.isnan(denom):
            V[c] = 0.0
        else:
            V[c] = col / denom
    # ponderação
    for c in V.columns:
        w = weights.get(c, 0.0)
        V[c] = V[c] * w
    # ideais
    ideal_pos, ideal_neg = {}, {}
    for c in V.columns:
        if directions[c] == "max":
            ideal_pos[c] = np.nanmax(V[c].values); ideal_neg[c] = np.nanmin(V[c].values)
        else:
            ideal_pos[c] = np.nanmin(V[c].values); ideal_neg[c] = np.nanmax(V[c].values)
    def dist(row, ideal):
        return np.sqrt(np.nansum((row - np.array([ideal[c] for c in V.columns])) ** 2))
    d_pos = V.apply(lambda r: dist(r.values, ideal_pos), axis=1)
    d_neg = V.apply(lambda r: dist(r.values, ideal_neg), axis=1)
    closeness = d_neg / (d_pos + d_neg + 1e-12)
    return closeness


def vikor(df_values: pd.DataFrame, weights: Dict[str, float], v: float) -> pd.Series:
    X = df_values.copy().astype(float)
    best, worst = {}, {}
    for c in X.columns:
        col = X[c].values.astype(float)
        if directions[c] == "max":
            best[c] = np.nanmax(col); worst[c] = np.nanmin(col)
        else:
            best[c] = np.nanmin(col); worst[c] = np.nanmax(col)

    gaps = {}
    for c in X.columns:
        denom = float(best[c] - worst[c])
        if abs(denom) < 1e-12 or np.isnan(denom):
            gaps[c] = np.zeros(len(X))
            continue
        w = float(weights.get(c, 0.0))
        if directions[c] == "max":
            gaps[c] = w * (best[c] - X[c].values) / denom
        else:
            gaps[c] = w * (X[c].values - best[c]) / denom
    G = pd.DataFrame(gaps, index=X.index)
    S = G.sum(axis=1); R = G.max(axis=1)
    S_min, S_max = float(S.min()), float(S.max())
    R_min, R_max = float(R.min()), float(R.max())

    def safe_norm(x, xmin, xmax):
        if abs(xmax - xmin) < 1e-12:
            return 0.0
        return (x - xmin) / (xmax - xmin)

    Q = pd.Series(index=X.index, dtype=float)
    for i in X.index:
        Q.loc[i] = v * safe_norm(S[i], S_min, S_max) + (1 - v) * safe_norm(R[i], R_min, R_max)
    return Q


def main():
    ap = argparse.ArgumentParser(description="TOPSIS, Fuzzy-TOPSIS e VIKOR no reopt_ALL_blocks_v3_8.csv com custo/emissões/confiabilidade.")
    ap.add_argument("--csv", default="dados_preprocessados/reopt_ALL_blocks_v3_8.csv", help="CSV consolidado do REopt.")
    ap.add_argument("--out", default="reopt_mcdm_results", help="Pasta de saída.")
    ap.add_argument("--fuzziness", type=float, default=0.05, help="Delta para triângulo fuzzy (± delta*valor).")
    ap.add_argument("--vikor_v", type=float, default=0.5, help="Parâmetro v do VIKOR (0..1).")
    ap.add_argument("--kgco2_per_litre", type=float, default=2.68, help="Fator de emissão do diesel (kgCO2/L).")
    ap.add_argument("--diesel_price_override", type=float, default=None, help="Preço do diesel ($/L) para calcular litros (opcional).")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    # montar alternativa
    df["alternative"] = df.apply(lambda r: f"{r.get('Region','')}_PLS{int(r.get('Percent_Load_Target',0))}_{r.get('System','')}", axis=1)

    # custo = LCOE; fallback LCC
    cost = df.get("LCOE_$_per_kWh") if "LCOE_$_per_kWh" in df.columns else None
    if cost is None:
        cost = pd.Series([np.nan]*len(df))
    cost_fallback = df.get("REopt_LCC_$", pd.Series([np.nan]*len(df)))
    cost = cost.fillna(cost_fallback)

    # emissões: calcula litros = Fuel_cost / preço (código 32/36/44 -> 3.2/3.6/4.4 $/L por padrão)
    price_map = {"32": 3.2, "36": 3.6, "44": 4.4}
    if args.diesel_price_override:
        price_series = pd.Series([args.diesel_price_override] * len(df))
    else:
        price_series = df.get("Diesel_Price_Code", pd.Series([""] * len(df))).astype(str).map(price_map)
    fuel_cost = df.get("Fuel_cost_$", pd.Series([np.nan] * len(df)))
    litres = fuel_cost / price_series.replace(0, np.nan)
    emissions = litres * args.kgco2_per_litre
    emissions = emissions.fillna(fuel_cost)  # fallback
    reliability = df.get("Percent_Load_Target", pd.Series([np.nan]*len(df)))

    mat = pd.DataFrame({
        "alternative": df["alternative"],
        "cost": cost,
        "emissions": emissions,
        "reliability": reliability,
        "Region": df.get("Region"),
        "PLS": df.get("Percent_Load_Target"),
        "System": df.get("System"),
    }).dropna(subset=["cost","emissions","reliability","alternative"])

    X = mat[["cost","emissions","reliability"]].astype(float)

    all_ranks = []
    winners = {}
    for profile, weights in PROFILES.items():
        s_topsis = topsis(X, weights)
        s_fuzzy = fuzzy_topsis(X, weights, fuzziness=args.fuzziness)
        s_vikor = vikor(X, weights, v=args.vikor_v)

        topsis_rank = pd.DataFrame({
            "alternative": mat["alternative"],
            "score": s_topsis.values,
            "Region": mat["Region"],
            "PLS": mat["PLS"],
            "System": mat["System"],
            "method": "topsis",
            "profile": profile,
        }).sort_values("score", ascending=False).reset_index(drop=True)
        topsis_rank["rank"] = np.arange(1, len(topsis_rank)+1)

        fuzzy_rank = pd.DataFrame({
            "alternative": mat["alternative"],
            "score": s_fuzzy.values,
            "Region": mat["Region"],
            "PLS": mat["PLS"],
            "System": mat["System"],
            "method": "fuzzy_topsis",
            "profile": profile,
        }).sort_values("score", ascending=False).reset_index(drop=True)
        fuzzy_rank["rank"] = np.arange(1, len(fuzzy_rank)+1)

        vikor_rank = pd.DataFrame({
            "alternative": mat["alternative"],
            "Q": s_vikor.values,
            "Region": mat["Region"],
            "PLS": mat["PLS"],
            "System": mat["System"],
            "method": "vikor",
            "profile": profile,
        }).sort_values("Q", ascending=True).reset_index(drop=True)
        vikor_rank["rank"] = np.arange(1, len(vikor_rank)+1)

        topsis_rank.to_csv(outdir / f"topsis_{profile}.csv", index=False)
        fuzzy_rank.to_csv(outdir / f"fuzzy_topsis_{profile}.csv", index=False)
        vikor_rank.to_csv(outdir / f"vikor_{profile}.csv", index=False)

        winners[profile] = {
            "topsis": topsis_rank.iloc[0]["alternative"],
            "fuzzy_topsis": fuzzy_rank.iloc[0]["alternative"],
            "vikor": vikor_rank.iloc[0]["alternative"],
        }
        all_ranks.extend([topsis_rank, fuzzy_rank, vikor_rank])

    summary = {
        "csv_used": str(Path(args.csv).resolve()),
        "profiles": PROFILES,
        "fuzziness": args.fuzziness,
        "vikor_v": args.vikor_v,
        "winners": winners,
    }
    with open(outdir/"summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # tabela geral
    df_all = pd.concat(all_ranks, ignore_index=True)
    df_all.to_csv(outdir / "all_ranks.csv", index=False)

    print("[OK] Resultados em", outdir.resolve())
    for p, ws in winners.items():
        print(f"- {p}: topsis={ws['topsis']} | fuzzy_topsis={ws['fuzzy_topsis']} | vikor={ws['vikor']}")


if __name__ == "__main__":
    main()
