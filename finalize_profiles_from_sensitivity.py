"""
Finaliza vencedor por perfil usando os ranks gerados em reopt_mcdm_sensitivity.

Regra:
1) Maioria entre vencedores dos métodos (TOPSIS, fuzzy TOPSIS, VIKOR) por perfil.
2) Empate -> Borda (pontos = n - rank + 1 por método).
3) Empate -> menor média de rank; persistindo -> ordem alfabética.

Entrada: all_ranks_sensitivity.csv (ex.: saída de run_reopt_sensitivity.py).
Usa vikor_v=0.5 por padrão (pode alterar via CLI).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pick_winner(ranks: pd.DataFrame) -> dict:
    # ranks: method, alternative, rank
    winners_by_method = ranks.sort_values("rank").groupby("method").first()["alternative"]
    counts = winners_by_method.value_counts()
    if not counts.empty and counts.max() >= 2:
        return {"winner": counts.idxmax(), "rule": "majority"}

    # Borda entre todos os candidatos
    candidates = ranks["alternative"].unique()
    scores = {a: 0.0 for a in candidates}
    for _, g in ranks.groupby("method"):
        n = len(g)
        for _, r in g.iterrows():
            scores[r["alternative"]] += (n - r["rank"] + 1)
    max_score = max(scores.values())
    tied = [a for a, s in scores.items() if s == max_score]
    if len(tied) == 1:
        return {"winner": tied[0], "rule": "borda", "note": str(scores)}

    # média de rank
    means = ranks.groupby("alternative")["rank"].mean()
    min_mean = means.min()
    tied_mean = means[means == min_mean].index.tolist()
    if len(tied_mean) == 1:
        return {"winner": tied_mean[0], "rule": "mean_rank", "note": str(means.to_dict())}

    return {"winner": sorted(tied_mean)[0], "rule": "alphabetical_tie", "note": str(means.to_dict())}


def main():
    ap = argparse.ArgumentParser(description="Finaliza vencedores por perfil a partir de all_ranks_sensitivity.csv")
    ap.add_argument("--ranks", default="reopt_mcdm_sensitivity/all_ranks_sensitivity.csv", help="CSV de ranks da sensibilidade")
    ap.add_argument("--vikor_v", type=float, default=0.5, help="Valor de v do VIKOR a filtrar")
    ap.add_argument("--out", default="final_profile_decisions.csv", help="CSV de saída")
    args = ap.parse_args()

    df = pd.read_csv(args.ranks)
    # normaliza nomes de método (topsis, fuzzy_topsis, vikor)
    df["method"] = df["method"].str.lower()
    df = df[df["vikor_v"] == args.vikor_v] if "vikor_v" in df.columns else df

    rows = []
    for profile, grp in df.groupby("profile"):
        # pegar colunas: alternative, rank, method
        sub = grp[["alternative", "rank", "method"]].copy()
        # para TOPSIS e fuzzy, rank já está; para VIKOR também
        res = pick_winner(sub)
        # vencedores individuais
        winners_by_method = sub.sort_values("rank").groupby("method").first()["alternative"].to_dict()
        rows.append({
            "profile": profile,
            "winner_topsis": winners_by_method.get("topsis"),
            "winner_fuzzy_topsis": winners_by_method.get("fuzzy_topsis"),
            "winner_vikor": winners_by_method.get("vikor"),
            "final_winner": res.get("winner"),
            "decision_rule": res.get("rule"),
            "note": res.get("note", ""),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print("Final por perfil salvo em", Path(args.out).resolve())
    print(out_df)


if __name__ == "__main__":
    main()
