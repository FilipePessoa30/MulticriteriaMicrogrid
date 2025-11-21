# build_mcdm_from_reopt.py
# ------------------------------------------------------------
# Constrói o MCDM (TOPSIS) a partir do CSV consolidado do REopt.
# Ajustes implementados:
# - Emissões de CO2 como critério primário ambiental (com várias rotas):
#   1) Coluna direta de CO2 (kg)
#   2) Energia do diesel (kWh) * fator_kgco2_kwh
#   3) Litros de diesel * kgCO2/L
#   4) Custo do combustível / preço_L * kgCO2/L
#   5) Fallback: usa Fuel Cost como proxy (se nada disponível)
# - Confiabilidade incluída se variar entre alternativas (senão é descartada).
# - Índice “Renováveis+Armazenamento” como default (PV + Battery) com pesos configuráveis.
# - Perfis de pesos atualizados e reponderação automática quando algum critério cair fora.
# - Sensibilidade opcional aos pesos.
# - Saídas padronizadas (contexto, bruta, normalizada, TOPSIS, leaderboards, winners).
#
# Uso (Windows, uma linha):
#   python build_mcdm_from_reopt.py --csv ".\reopt_ALL_blocks_v3_6.csv" --outdir ".\mcdm_out" --region "Accra" --load_profile "residential" --percent_load 100
#
# Uso (Windows, com quebras ^):
#   python build_mcdm_from_reopt.py ^
#     --csv ".\reopt_ALL_blocks_v3_6.csv" ^
#     --outdir ".\mcdm_out" ^
#     --region "Accra" ^
#     --load_profile "residential" ^
#     --percent_load 100 ^
#     --sensitivity
#
# Dependências: pandas, numpy
# ------------------------------------------------------------

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- util ---------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_float(x):
    """Converte diversos formatos numéricos para float de forma robusta.

    Regras:
    - Trata None/NaN vazios -> np.nan
    - Remove texto, símbolos de moeda e unidades deixando apenas dígitos, ponto, vírgula e sinais
    - Parênteses envolvendo números indicam negativo
    - Trata separadores de milhar e decimal em formatos 1,234.56 ou 1.234,56 ou 0,25
    - Retorna float ou np.nan
    """
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except:
            return np.nan
    if x is None:
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan

    # marcador de negativo por parênteses
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # remove espaços de NBSP
    s = s.replace("\u00A0", " ").replace("\xa0", " ")

    # remove percent sign and common words/units/currency symbols
    # manter apenas dígitos, ponto, vírgula, + e -
    s_clean = re.sub(r"[^0-9.,+\-]", "", s)
    if not s_clean:
        return np.nan

    # decidir separador decimal
    try:
        if "." in s_clean and "," in s_clean:
            # último separador define a casa decimal
            if s_clean.rfind(".") > s_clean.rfind(","):
                # ponto é decimal, remover vírgulas (milhar)
                s_num = s_clean.replace(",", "")
            else:
                # vírgula é decimal, remover pontos (milhar) e trocar vírgula por ponto
                s_num = s_clean.replace(".", "").replace(",", ".")
        elif "," in s_clean:
            # apenas vírgula presente
            if s_clean.count(",") > 1:
                # múltiplas vírgulas → provavelmente milhares
                s_num = s_clean.replace(",", "")
            else:
                # uma vírgula: decidir se decimal ou milhar (se parte decimal tiver 3 dígitos é milhar)
                left, right = s_clean.split(",")
                if len(right) == 3 and len(left) > 0:
                    s_num = s_clean.replace(",", "")
                else:
                    s_num = s_clean.replace(",", ".")
        else:
            # apenas ponto ou nenhum separador
            s_num = s_clean

        # garantir formato numérico válido (apenas dígitos, ponto, sinais)
        if not re.match(r"^[+-]?[0-9]*\.?[0-9]+$", s_num):
            # tentativa adicional: extrair primeiro trecho numérico
            m = re.search(r"[+-]?[0-9]+(?:[\.,][0-9]+)?", s)
            if m:
                t = m.group(0)
                t = t.replace(',', '.')
                val = float(t)
            else:
                return np.nan
        else:
            val = float(s_num)

        if neg:
            val = -val
        return float(val)
    except Exception:
        return np.nan


def ci(s: str) -> str:
    """lower sem acentos básicos para casar colunas de forma tolerante."""
    if s is None:
        return ""
    s = str(s).lower()
    table = str.maketrans("áàãâäéèêëíìîïóòõôöúùûüç", "aaaaaeeeeiiiiooooouuuuc")
    s = s.translate(table)
    # normaliza separadores e pontuação para espaços (underscore, $, %, parênteses etc.)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_alt_name(name: str) -> str:
    """Normalize alternative name for merging: strip, lower, collapse spaces."""
    if name is None:
        return ""
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lmap = {ci(c): c for c in cols}
    for pat in candidates:
        for k, orig in lmap.items():
            if pat in k:
                return orig
    return None


def pick_first(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """Atalho para tentar 1 coluna por vários padrões."""
    return find_col(df, patterns)


# ----------------------------- config ---------------------------------

@dataclass
class EmissionsConfig:
    kgco2_per_kwh: float = 0.42         # fator default pedido
    kgco2_per_litre: float = 2.68       # diesel ~2.68 kgCO2/L
    diesel_price_per_litre: Optional[float] = None  # se quiser derivar Litros de Custo

@dataclass
class RenewablesWeights:
    w_pv: float = 0.7
    w_batt: float = 0.3

# perfis padrão (serão reponderados conforme critérios presentes)
DEFAULT_PROFILES = {
    "TOPSIS_economic":     {"cost": 0.60, "co2": 0.20, "res_storage": 0.20, "reliability": 0.00},
    "TOPSIS_sustainable":  {"cost": 0.25, "co2": 0.45, "res_storage": 0.30, "reliability": 0.00},
    "TOPSIS_resilient":    {"cost": 0.30, "co2": 0.10, "res_storage": 0.35, "reliability": 0.25},
}

# nomes prováveis de colunas
CANDS = {
    "region": [
        "geographical region", "region"
    ],
    "load_profile": [
        "load profile"
    ],
    "percent_load_param": [  # parâmetro de entrada do REopt (pode ser constante por planilha)
        "percent of load served"
    ],
    "alt_name": [
        "scenario", "technology", "tech mix", "variant", "name", "option"
    ],
    "lcoe": [
        "lcoe", "levelized cost", "levelised cost", "lcoe ($/kwh)", "levelized cost of energy"
    ],
    "fuel_cost": [
        "fuel cost", "diesel cost", "generator fuel cost"
    ],
    "diesel_kwh": [
        "diesel kwh", "generator kwh", "generator energy (kwh)", "fuel kwh", "gen kwh"
    ],
    "fuel_litres": [
        "fuel volume", "diesel liters", "diesel litres", "fuel l", "liters of diesel", "litres of diesel"
    ],
    "co2_kg": [
        "co2", "emissions", "co2 emissions (kg)", "emissions (kg)", "kgco2"
    ],
    "pv_kw": [
        "pv kw", "pv capacity kw", "solar kw", "pv installed kw"
    ],
    "batt_kwh": [
        "battery kwh", "battery capacity kwh", "storage kwh"
    ],
    "reliability": [
        # se existir confiabilidade por alternativa
        "reliability", "loss of load probability", "lolp", "served (%)", "served (percent)"
    ],
}


# ------------------------ leitura e filtro de contexto -----------------

def filter_context(df: pd.DataFrame, args) -> pd.DataFrame:
    """Filtra por região, load_profile, percent_load se existirem. Se não existirem, segue em frente."""
    out = df.copy()

    col_region = pick_first(out, CANDS["region"])
    col_lp     = pick_first(out, CANDS["load_profile"])
    col_pct    = pick_first(out, CANDS["percent_load_param"])

    if args.region and col_region and col_region in out.columns:
        out = out[out[col_region].astype(str).str.contains(args.region, case=False, na=False)]

    if args.load_profile and col_lp and col_lp in out.columns:
        out = out[out[col_lp].astype(str).str.contains(args.load_profile, case=False, na=False)]

    if args.percent_load is not None and col_pct and col_pct in out.columns:
        # aceita 95, 100, 80 etc; tolerância ±0.5
        pl = out[col_pct].map(safe_float)
        out = out[(pl - args.percent_load).abs() <= 0.5]

    return out.reset_index(drop=True)


def pick_alternative_name(row: pd.Series) -> str:
    for key in CANDS["alt_name"]:
        cands = CANDS["alt_name"]
        # fallback: usaremos a primeira coluna que existir
    # acima não usa row. Vamos varrer direto as chaves candidatas no índice
    for col in row.index:
        if ci(col) in [ci(x) for x in CANDS["alt_name"]]:
            val = str(row[col]).strip()
            if val:
                return val
    # fallback heurístico: compose por tecnologias
    pv = safe_float(row.get(pick_first(pd.DataFrame([row]), CANDS["pv_kw"]) or "", np.nan))
    bk = safe_float(row.get(pick_first(pd.DataFrame([row]), CANDS["batt_kwh"]) or "", np.nan))
    fc = safe_float(row.get(pick_first(pd.DataFrame([row]), CANDS["fuel_cost"]) or "", np.nan))
    parts = []
    if not np.isnan(pv) and pv > 0: parts.append("PV")
    if not np.isnan(bk) and bk > 0: parts.append("Battery")
    if not np.isnan(fc) and fc > 0: parts.append("Diesel")
    nm = "+".join(parts) if parts else "Alternative"
    return nm


# ------------------------ critérios e decisão --------------------------

def compute_emissions(row: pd.Series, cols: Dict[str, Optional[str]], eco: EmissionsConfig) -> Tuple[float, str]:
    """Retorna (valor_emissoes_kgCO2, rota_usada)."""
    # 1) coluna direta
    if cols["co2_kg"]:
        v = safe_float(row[cols["co2_kg"]])
        if not np.isnan(v):
            return float(v), "direct_col"

    # 2) diesel_kwh
    if cols["diesel_kwh"]:
        e = safe_float(row[cols["diesel_kwh"]])
        if not np.isnan(e):
            return float(e) * eco.kgco2_per_kwh, "diesel_kwh*fator"

    # 3) fuel_litres
    if cols["fuel_litres"]:
        L = safe_float(row[cols["fuel_litres"]])
        if not np.isnan(L):
            return float(L) * eco.kgco2_per_litre, "litros*kgco2_L"

    # 4) fuel_cost -> litros -> CO2
    if cols["fuel_cost"] and eco.diesel_price_per_litre:
        cost = safe_float(row[cols["fuel_cost"]])
        if not np.isnan(cost) and eco.diesel_price_per_litre > 0:
            L = float(cost) / eco.diesel_price_per_litre
            return L * eco.kgco2_per_litre, "cost/preco_L*kgco2_L"

    # 5) fallback: usa Fuel cost como proxy (kgCO2 ~ custo) para manter coerência monotônica
    if cols["fuel_cost"]:
        fc = safe_float(row[cols["fuel_cost"]])
        if not np.isnan(fc):
            return float(fc), "fuel_cost_proxy"

    return np.nan, "not_available"


def valid_and_nonconstant(series: pd.Series) -> bool:
    # Deprecated: keep for compatibility; prefer calling with epsilon below.
    return valid_and_nonconstant_eps(series, 1e-12)


def valid_and_nonconstant_eps(series: pd.Series, epsilon: float) -> bool:
    """Retorna True se a série tiver pelo menos 2 valores não-NaN e variação maior que epsilon."""
    if series is None or series.isna().all():
        return False
    vals = series.dropna().astype(float).values
    if len(vals) <= 1:
        return False
    return (np.nanmax(vals) - np.nanmin(vals)) > float(epsilon)


def topsis(df_values: pd.DataFrame, directions: Dict[str, str], weights: Dict[str, float]) -> pd.Series:
    """
    df_values: linhas=alternativas; colunas=critérios (numéricos)
    directions: {"crit":"max|min"}
    weights:    {"crit": weight} (deve somar ~1)
    """
    # normalização vetorial
    V = df_values.copy().astype(float)
    for c in V.columns:
        col = V[c].values.astype(float)
        denom = np.sqrt(np.nansum(col ** 2))
        if denom == 0 or np.isnan(denom):
            # se zerado (ou NaN), fica tudo zero
            V[c] = 0.0
        else:
            V[c] = col / denom

    # ponderação
    for c in V.columns:
        w = weights.get(c, 0.0)
        V[c] = V[c] * w

    # ideal positivo/negativo
    ideal_pos = {}
    ideal_neg = {}
    for c in V.columns:
        if directions[c] == "max":
            ideal_pos[c] = np.nanmax(V[c].values)
            ideal_neg[c] = np.nanmin(V[c].values)
        else:
            ideal_pos[c] = np.nanmin(V[c].values)
            ideal_neg[c] = np.nanmax(V[c].values)

    # distâncias
    def dist(row, ideal):
        return np.sqrt(np.nansum((row - np.array([ideal[c] for c in V.columns])) ** 2))

    d_pos = V.apply(lambda r: dist(r.values, ideal_pos), axis=1)
    d_neg = V.apply(lambda r: dist(r.values, ideal_neg), axis=1)
    score = d_neg / (d_pos + d_neg + 1e-12)
    return score


def vikor(df_values: pd.DataFrame, directions: Dict[str, str], weights: Dict[str, float], v: float = 0.5) -> pd.Series:
    """
    Implementação simples do VIKOR (solução de compromisso).
    Retorna série Q (menor é melhor).
    """
    X = df_values.copy().astype(float)
    best, worst = {}, {}
    for c in X.columns:
        col = X[c].values.astype(float)
        if directions[c] == "max":
            best[c] = np.nanmax(col)
            worst[c] = np.nanmin(col)
        else:
            best[c] = np.nanmin(col)
            worst[c] = np.nanmax(col)

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

    S = G.sum(axis=1)
    R = G.max(axis=1)

    S_min, S_max = float(S.min()), float(S.max())
    R_min, R_max = float(R.min()), float(R.max())

    def safe_norm(x, xmin, xmax):
        if abs(xmax - xmin) < 1e-12:
            return 0.0
        return (x - xmin) / (xmax - xmin)

    Q = pd.Series(index=X.index, dtype=float)
    for i in X.index:
        q_i = v * safe_norm(S[i], S_min, S_max) + (1 - v) * safe_norm(R[i], R_min, R_max)
        Q.loc[i] = q_i
    return Q


def fuzzy_topsis(df_values: pd.DataFrame, directions: Dict[str, str], weights: Dict[str, float], fuzziness: float = 0.05) -> pd.Series:
    """
    Variante leve do TOPSIS fuzzy com números triangulares.
    Converte valores e pesos crisp em triângulos (l, m, u) com faixa +/-fuzziness*|x|.
    Retorna índice de closeness (maior é melhor).
    """
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
        col_tris = F[c]
        ideals[c] = {
            "pos": ideal(col_tris, directions[c], "pos"),
            "neg": ideal(col_tris, directions[c], "neg"),
        }

    def dist(t1, t2):
        if any(np.isnan(t1)) or any(np.isnan(t2)):
            return np.nan
        return np.sqrt(((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2) / 3.0)

    scores = []
    for idx in X.index:
        dp = []
        dn = []
        for c in X.columns:
            t = F[c][list(X.index).index(idx)]
            w = float(weights.get(c, 0.0))
            dp.append(w * dist(t, ideals[c]["pos"]))
            dn.append(w * dist(t, ideals[c]["neg"]))
        dp_val = np.nansum(dp)
        dn_val = np.nansum(dn)
        scores.append(dn_val / (dp_val + dn_val + 1e-12))
    return pd.Series(scores, index=X.index)


def reweight(weights: Dict[str, float], present_criteria: List[str]) -> Dict[str, float]:
    """Repondera pesos só nos critérios presentes, preservando a razão relativa."""
    w = {k: v for k, v in weights.items() if k in present_criteria}
    s = sum(w.values())
    if s <= 0:
        # se tudo zerado, distribui uniforme
        uni = 1.0 / max(1, len(present_criteria))
        return {k: uni for k in present_criteria}
    return {k: v / s for k, v in w.items()}


def grid_perturb(weights: Dict[str, float], step: float = 0.1) -> List[Dict[str, float]]:
    """
    Gera pequenas variações nos pesos: para cada critério, +/- step,
    redistribuindo proporcionalmente nos demais.
    """
    keys = list(weights.keys())
    grids = [weights.copy()]

    for k in keys:
        for sgn in (+1, -1):
            w = weights.copy()
            delta = sgn * step
            new_k = w[k] + delta
            if new_k < 0:
                continue
            rest_keys = [r for r in keys if r != k]
            rest_sum = sum(w[r] for r in rest_keys)
            if rest_sum <= 0:
                continue
            scale = (rest_sum - delta) / rest_sum
            for r in rest_keys:
                w[r] *= scale
            w[k] = new_k
            # renormaliza por segurança
            s = sum(w.values())
            if s > 0:
                w = {kk: vv / s for kk, vv in w.items()}
                grids.append(w)
    # remove duplicatas grosseiras
    uniq = []
    seen = set()
    for g in grids:
        key = tuple(round(g[k], 6) for k in sorted(g))
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    return uniq


# ----------------------------- pipeline --------------------------------

def build(args):
    ensure_dir(args.outdir)

    df = pd.read_csv(args.csv)
    df_f = filter_context(df, args)

    if df_f.empty:
        raise RuntimeError("Filtro de contexto zerou a base. Verifique region/load_profile/percent_load.")

    # mapeia colunas presentes
    cols = {
        "region":          pick_first(df_f, CANDS["region"]),
        "load_profile":    pick_first(df_f, CANDS["load_profile"]),
        "percent_param":   pick_first(df_f, CANDS["percent_load_param"]),
        "lcoe":            pick_first(df_f, CANDS["lcoe"]),
        "fuel_cost":       pick_first(df_f, CANDS["fuel_cost"]),
        "diesel_kwh":      pick_first(df_f, CANDS["diesel_kwh"]),
        "fuel_litres":     pick_first(df_f, CANDS["fuel_litres"]),
        "co2_kg":          pick_first(df_f, CANDS["co2_kg"]),
        "pv_kw":           pick_first(df_f, CANDS["pv_kw"]),
        "batt_kwh":        pick_first(df_f, CANDS["batt_kwh"]),
        "reliability":     pick_first(df_f, CANDS["reliability"]),
        "alt_name":        pick_first(df_f, CANDS["alt_name"]),
    }

    eco = EmissionsConfig(
        kgco2_per_kwh=args.kgco2_per_kwh,
        kgco2_per_litre=args.kgco2_per_litre,
        diesel_price_per_litre=args.diesel_price_per_litre
    )
    rw = RenewablesWeights(w_pv=args.w_pv, w_batt=args.w_batt)

    # constrói tabela de alternativas
    records = []
    emissions_route_count: Dict[str, int] = {}
    for _, row in df_f.iterrows():
        # alternativa: preferir coluna candidata, senão gerar a partir de presença de tecnologias
        if cols.get("alt_name"):
            alt_raw = row.get(cols["alt_name"])
            alt = str(alt_raw).strip() if pd.notna(alt_raw) else ""
        else:
            # heurística: PV/Battery/Diesel a partir de colunas detectadas
            pv_val = safe_float(row[cols["pv_kw"]]) if cols.get("pv_kw") else np.nan
            batt_val = safe_float(row[cols["batt_kwh"]]) if cols.get("batt_kwh") else np.nan
            fuel_val = None
            if cols.get("fuel_cost"):
                fuel_val = safe_float(row[cols["fuel_cost"]])
            elif cols.get("diesel_kwh"):
                fuel_val = safe_float(row[cols["diesel_kwh"]])

            parts = []
            if not np.isnan(pv_val) and pv_val > 0:
                parts.append("PV")
            if not np.isnan(batt_val) and batt_val > 0:
                parts.append("Battery")
            if fuel_val is not None and not np.isnan(fuel_val) and float(fuel_val) > 0:
                parts.append("Diesel")
            alt = "+".join(parts) if parts else "Alternative"

        lcoe = safe_float(row[cols["lcoe"]]) if cols.get("lcoe") else np.nan
        pv   = safe_float(row[cols["pv_kw"]]) if cols.get("pv_kw") else np.nan
        bat  = safe_float(row[cols["batt_kwh"]]) if cols.get("batt_kwh") else np.nan
        rel  = safe_float(row[cols["reliability"]]) if cols.get("reliability") else np.nan
        co2, route = compute_emissions(row, cols, eco)

        # contar rota de emissões
        emissions_route_count[route] = emissions_route_count.get(route, 0) + 1

        records.append({
            "alternative": alt,
            "LCOE_USD_per_kWh": lcoe,
            "PV_kW": pv,
            "Battery_kWh": bat,
            "Reliability_%": rel,
            "CO2_kg": co2,
            "CO2_route": route
        })

    alts = pd.DataFrame(records)

    # Agrega múltiplas linhas que representem a mesma 'alternative' (p.ex. várias simulações)
    # para garantir que cada alternativa apareça uma vez na matriz de decisão.
    if not alts.empty:
        aggs = {
            'LCOE_USD_per_kWh': 'mean',
            'PV_kW': 'mean',
            'Battery_kWh': 'mean',
            'Reliability_%': 'mean',
            'CO2_kg': 'mean',
            'CO2_route': 'first'
        }
        alts = alts.groupby('alternative', as_index=False).agg(aggs)

    # normaliza nome-chave para merges e agregações
    if not alts.empty:
        alts["alt_key"] = alts["alternative"].map(normalize_alt_name)

    # se o usuário forneceu CSV de confiabilidade, fazer o merge por alternativa
    if args.reliability_csv:
        try:
            rel_df = pd.read_csv(args.reliability_csv)
            # localizar colunas (case-insensitive)
            rel_cols = {ci(c): c for c in rel_df.columns}
            key_col = rel_cols.get(ci(args.reliability_key), args.reliability_key)
            val_col = rel_cols.get(ci(args.reliability_value), args.reliability_value)
            if key_col not in rel_df.columns or val_col not in rel_df.columns:
                raise RuntimeError(f"Arquivo de confiabilidade não contém colunas {args.reliability_key} e/ou {args.reliability_value}")
            rel_df = rel_df[[key_col, val_col]].copy()
            rel_df["alt_key"] = rel_df[key_col].astype(str).map(lambda x: normalize_alt_name(x))
            rel_df[val_col] = rel_df[val_col].map(safe_float)
            rel_agg = rel_df.groupby("alt_key")[val_col].mean().reset_index().rename(columns={val_col: "Reliability_from_csv"})
            if not alts.empty:
                alts = alts.merge(rel_agg, on="alt_key", how="left")
                # sobrescrever/usar valor de confiabilidade vindo do CSV
                alts["Reliability_%"] = alts.apply(lambda r: r["Reliability_from_csv"] if not np.isnan(r.get("Reliability_from_csv", np.nan)) else r.get("Reliability_%"), axis=1)
                alts.drop(columns=[c for c in ["Reliability_from_csv"] if c in alts.columns], inplace=True)
        except Exception as e:
            print(f"[WARN] Falha ao ler/mergear reliability_csv: {e}")
    if args.aggregate_renewables:
        # normaliza internamente PV e Battery com min-max local para composição do índice
        def mm(x):
            x = x.astype(float)
            mn, mx = np.nanmin(x), np.nanmax(x)
            if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
                return pd.Series([0.0] * len(x), index=x.index)
            return (x - mn) / (mx - mn)

        pv_n = mm(alts["PV_kW"]) if alts["PV_kW"].notna().any() else pd.Series([0]*len(alts))
        bt_n = mm(alts["Battery_kWh"]) if alts["Battery_kWh"].notna().any() else pd.Series([0]*len(alts))

        alts["RES_Storage_Index"] = rw.w_pv * pv_n + rw.w_batt * bt_n
    else:
        alts["RES_Storage_Index"] = np.nan

    # salva todas as colunas auditoráveis
    alts.to_csv(os.path.join(args.outdir, "alternatives_raw.csv"), index=False)

    # Decide quais critérios vão para a matriz de decisão
    crits = {}
    directions = {}
    criteria_included: List[str] = []
    criteria_dropped: Dict[str, str] = {}
    # custo
    # custo: incluir se variável acima do epsilon ou se --force_cost e >=2 valores não-NaN
    lcoe_series = alts.get("LCOE_USD_per_kWh", pd.Series(dtype=float))
    lcoe_non_na = int(lcoe_series.notna().sum())
    if valid_and_nonconstant_eps(lcoe_series, args.min_variation_epsilon):
        crits["cost"] = lcoe_series.astype(float)
        directions["cost"] = "min"
        criteria_included.append("cost")
    else:
        if args.force_cost and lcoe_non_na >= 2:
            crits["cost"] = lcoe_series.astype(float)
            directions["cost"] = "min"
            criteria_included.append("cost")
            criteria_dropped.pop("cost", None)
        else:
            # motivo de drop
            if lcoe_series.isna().all():
                criteria_dropped["cost"] = "all_nan"
            elif lcoe_non_na < 2:
                criteria_dropped["cost"] = "insufficient_values"
            else:
                criteria_dropped["cost"] = "constant_or_below_epsilon"
    # co2 (ou proxy)
    if valid_and_nonconstant_eps(alts.get("CO2_kg", pd.Series(dtype=float)), args.min_variation_epsilon):
        crits["co2"] = alts["CO2_kg"].astype(float)
        directions["co2"] = "min"
        criteria_included.append("co2")
    else:
        if alts.get("CO2_kg", pd.Series()).isna().all():
            criteria_dropped["co2"] = "all_nan"
        else:
            criteria_dropped["co2"] = "constant_or_below_epsilon"
    # renováveis + armazenamento
    if args.aggregate_renewables:
        if valid_and_nonconstant_eps(alts.get("RES_Storage_Index", pd.Series(dtype=float)), args.min_variation_epsilon):
            crits["res_storage"] = alts["RES_Storage_Index"].astype(float)
            directions["res_storage"] = "max"
            criteria_included.append("res_storage")
        else:
            criteria_dropped["res_storage"] = "constant_or_below_epsilon_or_missing"
    else:
        if valid_and_nonconstant_eps(alts.get("PV_kW", pd.Series(dtype=float)), args.min_variation_epsilon):
            crits["PV_kW"] = alts["PV_kW"].astype(float); directions["PV_kW"] = "max"; criteria_included.append("PV_kW")
        else:
            criteria_dropped["PV_kW"] = "constant_or_below_epsilon_or_missing"
        if valid_and_nonconstant_eps(alts.get("Battery_kWh", pd.Series(dtype=float)), args.min_variation_epsilon):
            crits["Battery_kWh"] = alts["Battery_kWh"].astype(float); directions["Battery_kWh"] = "max"; criteria_included.append("Battery_kWh")
        else:
            criteria_dropped["Battery_kWh"] = "constant_or_below_epsilon_or_missing"
    # confiabilidade
    if valid_and_nonconstant_eps(alts.get("Reliability_%", pd.Series(dtype=float)), args.min_variation_epsilon):
        crits["reliability"] = alts["Reliability_%"].astype(float)
        directions["reliability"] = "max"
        criteria_included.append("reliability")
    else:
        if "Reliability_%" in alts.columns and alts["Reliability_%"].notna().sum() >= 1:
            # estava presente mas não variou o suficiente
            criteria_dropped["reliability"] = "constant_or_below_epsilon"
        else:
            criteria_dropped["reliability"] = "missing"

    if not crits:
        raise RuntimeError("Nenhum critério válido/variável encontrado depois de checagens. Verifique o CSV.")

    # matriz bruta
    M_raw = pd.DataFrame(crits)
    M_raw.insert(0, "alternative", alts["alternative"].values)
    M_raw.to_csv(os.path.join(args.outdir, "decision_matrix_raw.csv"), index=False)

    # normalização vetorial individual por critério (como no TOPSIS)
    M = pd.DataFrame({k: v.values for k, v in crits.items()}, index=alts.index).astype(float)
    M_norm = M.copy()
    for c in M_norm.columns:
        col = M_norm[c].values.astype(float)
        denom = np.sqrt(np.nansum(col ** 2))
        if denom == 0 or np.isnan(denom):
            M_norm[c] = 0.0
        else:
            M_norm[c] = col / denom
    M_norm.insert(0, "alternative", alts["alternative"].values)
    M_norm.to_csv(os.path.join(args.outdir, "decision_matrix_normalized.csv"), index=False)

    # perfis
    # repondera de acordo com critérios presentes
    profiles = {}
    for pname, w0 in DEFAULT_PROFILES.items():
        # garantir que cost esteja presente nos pesos se for forçado
        present = list(crits.keys())
        if args.force_cost and "cost" not in present and "cost" in w0:
            # só incluir se existirem ao menos dois valores não NaN
            if int(alts.get("LCOE_USD_per_kWh", pd.Series()).notna().sum()) >= 2:
                present.append("cost")
        w = reweight(w0, present_criteria=present)
        profiles[pname] = w

    # Métodos disponíveis
    available_methods = {"topsis", "vikor", "fuzzy_topsis"}
    methods = [m.strip().lower() for m in str(args.method).split(",") if m.strip()]
    if "all" in methods:
        methods = list(available_methods)
    methods = [m for m in methods if m in available_methods]
    if not methods:
        methods = ["topsis"]

    # scores por método/perfil
    method_better_high = {"topsis": True, "fuzzy_topsis": True, "vikor": False}
    scores_all = []
    for pname, w in profiles.items():
        df_vals = pd.DataFrame(crits)
        for method in methods:
            if method == "topsis":
                s = topsis(df_vals, directions, w)
            elif method == "vikor":
                s = vikor(df_vals, directions, w, v=args.vikor_v)
            elif method == "fuzzy_topsis":
                s = fuzzy_topsis(df_vals, directions, w, fuzziness=args.fuzzy_delta)
            else:
                continue
            tmp = pd.DataFrame({
                "alternative": alts["alternative"],
                "profile": pname,
                "method": method,
                "score": s.values
            })
            scores_all.append(tmp)

    scores = pd.concat(scores_all, ignore_index=True)
    scores.to_csv(os.path.join(args.outdir, "mcdm_scores.csv"), index=False)
    # compatibilidade com antigo TOPSIS
    if "topsis" in methods:
        scores[scores["method"]=="topsis"].drop(columns="method").to_csv(
            os.path.join(args.outdir, "topsis_scores.csv"), index=False
        )

    # leaderboards
    winners = {}
    for method in methods:
        winners[method] = {}
        for pname in profiles:
            sc = scores[(scores["profile"] == pname) & (scores["method"] == method)]
            asc = not method_better_high.get(method, True)
            sc = sc.sort_values("score", ascending=asc).reset_index(drop=True)
            sc_rank = sc.copy()
            sc_rank["rank"] = np.arange(1, len(sc_rank) + 1)

            sc_rank.to_csv(os.path.join(args.outdir, f"leaderboard_{pname}_{method}.csv"), index=False)
            sc_rank[["alternative", "rank"]].to_csv(os.path.join(args.outdir, f"leaderboard_rank_{pname}_{method}.csv"), index=False)

            winners[method][pname] = sc_rank.loc[0, "alternative"] if not sc_rank.empty else None

        # relatório compacto por método
        rep = scores[scores["method"] == method].pivot(index="alternative", columns="profile", values="score").reset_index()
        rep.to_csv(os.path.join(args.outdir, f"report_{method}.csv"), index=False)

    # winners summary
    with open(os.path.join(args.outdir, "winners_summary.json"), "w", encoding="utf-8") as f:
        json.dump(winners, f, indent=2, ensure_ascii=False)

    # contexto
    context = {
        "source_csv": os.path.abspath(args.csv),
        "filters": {
            "region": args.region,
            "load_profile": args.load_profile,
            "percent_load": args.percent_load
        },
        "columns_used": cols,
        "present_columns": {k: v for k, v in cols.items() if v is not None},
        "criteria_included": criteria_included,
        "criteria_dropped": criteria_dropped,
        "emissions_route_count": emissions_route_count,
        "emissions_config": {
            "kgco2_per_kwh": eco.kgco2_per_kwh,
            "kgco2_per_litre": eco.kgco2_per_litre,
            "diesel_price_per_litre": eco.diesel_price_per_litre
        },
        "renewables_weights": {"w_pv": rw.w_pv, "w_batt": rw.w_batt},
        "aggregate_renewables": args.aggregate_renewables,
        "directions": directions,
        "profiles": profiles,
        "methods": methods,
        "vikor_v": args.vikor_v,
        "fuzzy_delta": args.fuzzy_delta,
    }
    with open(os.path.join(args.outdir, "context_options.json"), "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=False)

    # sensibilidade simples (opcional)
    if args.sensitivity:
        sens = {}
        base_crits = list(crits.keys())
        for pname, base_w in profiles.items():
            grids = grid_perturb(base_w, step=args.sens_step)
            counts = {}
            for w in grids:
                sc = topsis(pd.DataFrame(crits), directions, w)
                alt = alts.loc[np.argmax(sc.values), "alternative"]
                counts[alt] = counts.get(alt, 0) + 1
            sens[pname] = counts
        with open(os.path.join(args.outdir, "sensitivity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sens, f, indent=2, ensure_ascii=False)

    # resumo de diagnóstico no console
    print("\n[DIAGNÓSTICO] critérios incluídos:", criteria_included)
    print("[DIAGNÓSTICO] direções:", directions)
    # pesos por perfil após reponderação
    print("[DIAGNÓSTICO] pesos por perfil (após reponderação):")
    for pname, w in profiles.items():
        print(f"  - {pname}: {w}")
    predominant_route = None
    if emissions_route_count:
        predominant_route = max(emissions_route_count.items(), key=lambda x: x[1])[0]
    print("[DIAGNÓSTICO] rota CO2 predominante:", predominant_route)

    print("[OK] MCDM gerado em:", os.path.abspath(args.outdir))
    print("Vencedores por perfil:", winners)


def parse_args():
    p = argparse.ArgumentParser(description="MCDM (TOPSIS, VIKOR, Fuzzy TOPSIS) a partir do CSV do REopt.")
    p.add_argument("--csv", required=True, help="Caminho para reopt_ALL_blocks_vX_Y.csv")
    p.add_argument("--outdir", required=True, help="Pasta de saída")
    # filtros de contexto (se existirem no CSV)
    p.add_argument("--region", default=None, help="Filtro por região (substring)")
    p.add_argument("--load_profile", default=None, help="Filtro por load profile (substring)")
    p.add_argument("--percent_load", type=float, default=None, help="Filtro por Percent of load served (±0.5)")
    p.add_argument("--method", default="topsis", help="Método(s): topsis, vikor, fuzzy_topsis ou all (separar por vírgula)")
    p.add_argument("--vikor_v", type=float, default=0.5, help="Parâmetro v do VIKOR (0..1)")
    p.add_argument("--fuzzy_delta", type=float, default=0.05, help="Fator de fuzziness para o fuzzy TOPSIS (+/- delta*valor)")
    # emissões
    p.add_argument("--kgco2_per_kwh", type=float, default=0.42)
    p.add_argument("--kgco2_per_litre", type=float, default=2.68)
    p.add_argument("--diesel_price_per_litre", type=float, default=None,
                   help="Se informado, permite derivar litros a partir de Fuel Cost.")
    p.add_argument("--force_cost", action="store_true", default=False,
                   help="Força inclusão do critério cost mesmo se variação for pequena (requer >=2 valores não-NaN).")
    p.add_argument("--min_variation_epsilon", type=float, default=1e-6,
                   help="Epsilon mínimo (max-min) para considerar um critério variável.")
    p.add_argument("--reliability_csv", default=None, help="CSV auxiliar com valores de confiabilidade por alternativa.")
    p.add_argument("--reliability_key", default="alternative", help="Nome da coluna no CSV de confiabilidade que identifica a alternativa.")
    p.add_argument("--reliability_value", default="Reliability_%", help="Nome da coluna no CSV com o valor de confiabilidade.")
    # renováveis
    p.add_argument("--aggregate_renewables", action="store_true", default=True,
                   help="Se ativo, usa índice RES_Storage_Index (PV+Battery).")
    p.add_argument("--w_pv", type=float, default=0.7)
    p.add_argument("--w_batt", type=float, default=0.3)
    # sensibilidade
    p.add_argument("--sensitivity", action="store_true", help="Executa sensibilidade nos pesos dos perfis.")
    p.add_argument("--sens_step", type=float, default=0.1, help="Passo da variação de pesos na sensibilidade.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args)
