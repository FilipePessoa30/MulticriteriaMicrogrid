# reopt_extract_v3_8.py
# ------------------------------------------------------------
# Extrai blocos regionais do arquivo:
#   "1 - Microgrid_ReOpt_LCOE_Results_Explorer.xlsm"
# e grava reopt_ALL_blocks_v3_8.csv com colunas padronizadas.
#
# Principais ajustes desta versão:
# - Mantém o esquema de leitura dos blocos regionais (PV, bateria,
#   gerador, custos, etc).
# - Quando NÃO encontra LCOE dentro do bloco, busca os valores na aba
#   "LCOE_breakdown", combinando com "process_inputs".
# - Tabela de LCOE é ligada ao df principal por:
#   Region_Sheet, Discount_Rate_%, Percent_Load_Target,
#   Cost_PVBatt, Cost_Diesel, Diesel_Price_Code, System.
# - Se, no futuro, houver LCOE dentro dos blocos regionais, esses
#   valores continuam tendo prioridade (só preenche NaN).
#
# Execução:
#   python reopt_extract_v3_8.py --file "1 - Microgrid_ReOpt_LCOE_Results_Explorer.xlsm"
#   (opcional) --out "reopt_ALL_blocks_v3_8.csv"
# ------------------------------------------------------------

import argparse
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd

REGION_PREFIXES = ["Accra_", "Lodwar_", "Lusaka_"]
PLS_ALLOWED = (100, 95, 90, 85, 80)
DEFAULT_ANNUAL_KWH = 19711.0
DEFAULT_ANALYSIS_YEARS = 20

# ---------- utils básicos ----------

def is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and pd.notna(x)

def parse_number(v):
    """Converte strings com vírgula/ponto decimal para float."""
    if is_number(v):
        return float(v)
    if v is None:
        return np.nan
    s = str(v).strip().replace("\u00A0", " ").replace(" ", "")
    if s == "" or s.lower() in ("nan", "none"):
        return np.nan
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    elif "." in s and "," in s:
        # último símbolo conta como decimal; o outro é milhar
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")
        else:
            s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def _ci(s: str) -> str:
    """Normaliza para comparação: minúsculas, sem acentos, só [a-z0-9 ] e espaços únicos."""
    if s is None:
        return ""
    s = str(s).lower()
    table = str.maketrans("áàãâäéèêëíìîïóòõôöúùûüç", "aaaaaeeeeiiiiooooouuuuc")
    s = s.translate(table)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def map_system(code: str) -> str:
    c = (code or "").lower()
    if c == "pbd": return "PV+battery+diesel"
    if c == "pb":  return "PV+battery"
    if c in ("d", "do", "dieselonly", "diesel_only", "diesel-only"):
        return "Diesel only"
    return (code or "").upper()

def infer_system(pv_kw, gen_kw, batt_kw, batt_kwh):
    pv  = float(pv_kw)  if is_number(pv_kw)  else 0.0
    gen = float(gen_kw) if is_number(gen_kw) else 0.0
    bk  = float(batt_kw)   if is_number(batt_kw)   else 0.0
    bkwh= float(batt_kwh)  if is_number(batt_kwh)  else 0.0
    if pv <= 0 and bk <= 0 and bkwh <= 0 and gen > 0: return "Diesel only"
    if pv > 0 and gen > 0:  return "PV+battery+diesel"
    if pv > 0 and gen <= 0: return "PV+battery"
    return None

# PRIORIDADE: pbd | pb | d ...
BLOCK_RE = re.compile(
    r"^TMY_(?:(low|med|high)PVBatt_)?(low|med|high)Diesel_(32|36|44)_(pbd|pb|d|do|dieselonly|diesel_only|diesel\-only)_(business|residential|nrel|average)$",
    re.IGNORECASE
)

# Rótulos esperados na coluna A
METRIC_LABELS = [
    "PV kW",
    "Battery kW",
    "Battery kWh",
    "Generator kW",
    "PV capital cost",
    "PV O&M",
    "Battery capital cost",
    "Battery O&M",
    "Diesel capital cost",
    "Diesel O&M (kW)",
    "Diesel O&M (kWh)",
    "Fuel cost",
    "REopt LCC",
    # LCOE não existe nos blocos desta planilha,
    # mas deixamos a estrutura para compatibilidade.
    "LCOE ($/kWh)",
]

# Alias/assinaturas tolerantes (normalizadas via _ci)
LABEL_ALIASES = {
    "PV kW": [
        "pv kw", "pv capacity kw", "solar kw"
    ],
    "Battery kW": [
        "battery kw", "storage kw", "batt kw"
    ],
    "Battery kWh": [
        "battery kwh", "battery capacity kwh", "storage kwh", "batt kwh"
    ],
    "Generator kW": [
        "generator kw", "diesel kw", "genset kw"
    ],
    "PV capital cost": [
        "pv capital cost", "pv capex"
    ],
    "PV O&M": [
        "pv o m", "pv om", "pv o and m"
    ],
    "Battery capital cost": [
        "battery capital cost", "battery capex", "storage capex"
    ],
    "Battery O&M": [
        "battery o m", "battery om", "battery o and m", "storage om"
    ],
    "Diesel capital cost": [
        "diesel capital cost", "gen capital cost", "generator capital cost", "genset capex"
    ],
    "Diesel O&M (kW)": [
        "diesel o m kw", "diesel om kw", "genset om kw", "generator om kw"
    ],
    "Diesel O&M (kWh)": [
        "diesel o m kwh", "diesel om kwh", "genset om kwh", "generator om kwh"
    ],
    "Fuel cost": [
        "fuel cost", "diesel cost", "generator fuel cost", "fuel usd"
    ],
    "REopt LCC": [
        "reopt lcc", "lcc", "life cycle cost", "lifecycle cost", "net present cost", "npc"
    ],
    "LCOE ($/kWh)": [
        "lcoe", "lcoe usd kwh", "lcoe per kwh",
        "levelized cost of energy", "levelised cost of energy"
    ],
}

# ---------- helpers de estrutura (abas regionais) ----------

def list_region_sheets(xls: pd.ExcelFile):
    names = xls.sheet_names
    region = []
    for pref in REGION_PREFIXES:
        region.extend([s for s in names if s.startswith(pref)])
    generic = [s for s in names
               if re.match(r"^(Accra|Lodwar|Lusaka)_\d+$", s, flags=re.I)
               and s not in region]
    region.extend(generic)
    out, seen = [], set()
    for s in region:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def parse_discount_rate_from_sheet(sheet_name: str):
    m = re.search(r"_(\d+)$", sheet_name)
    return int(m.group(1)) if m else None

def find_block_headers(df: pd.DataFrame):
    headers = []
    R, C = df.shape
    for r in range(R):
        for c in range(C):
            v = df.iat[r, c]
            if isinstance(v, str) and v.startswith("TMY_"):
                m = BLOCK_RE.match(v.strip())
                if not m:
                    continue
                pvb_cost_opt, diesel_cost, diesel_price, sys_code, profile = m.groups()
                headers.append({
                    "row": r, "col": c,
                    "pvbatt_cost": (pvb_cost_opt or "").lower(),
                    "diesel_cost": diesel_cost.lower(),
                    "diesel_price": diesel_price,
                    "sys_code": sys_code.lower(),
                    "profile": profile.lower(),
                    "header_text": v.strip(),
                })
    headers.sort(key=lambda x: (x["row"], x["col"]))
    return headers

def group_end_row(headers, idx, last_row):
    cur_row = headers[idx]["row"]
    for j in range(idx + 1, len(headers)):
        if headers[j]["row"] > cur_row:
            return headers[j]["row"] - 1
    return last_row

def find_pls_cols(df: pd.DataFrame, header_row: int, header_col: int):
    mapping = {}
    for off in range(0, 5):
        r = header_row + off
        if r >= df.shape[0]:
            break
        for k in range(1, 40):
            c = header_col + k
            if c >= df.shape[1]:
                break
            v = df.iat[r, c]
            try:
                iv = int(round(float(v)))
            except Exception:
                continue
            if iv in PLS_ALLOWED and iv not in mapping:
                mapping[iv] = c
        if len(mapping) == len(PLS_ALLOWED):
            break
    return mapping

def _label_match_norm(s: str) -> str:
    return _ci(s)

def build_label_index(df: pd.DataFrame, start_row: int, end_row: int):
    """
    Constrói índice de rótulos tolerante:
    - Procura na Coluna A (índice 0) entre start_row..end_row
    - Faz matching por aliases normalizados
    Retorna: dict {label_canonico -> row_index}
    """
    idx = {}
    r0, r1 = max(0, start_row), min(df.shape[0] - 1, end_row)

    # mapa alias -> label canônico
    alias_norm_to_canon = {}
    for canon, aliases in LABEL_ALIASES.items():
        for a in aliases + [canon]:
            alias_norm_to_canon[_label_match_norm(a)] = canon

    seen_canon = set()

    for r in range(r0, r1 + 1):
        cell = df.iat[r, 0]
        if not isinstance(cell, str):
            continue
        norm = _label_match_norm(cell)
        if not norm:
            continue
        # substring match
        for alias_norm, canon in alias_norm_to_canon.items():
            if canon in seen_canon:
                continue
            if alias_norm and alias_norm in norm:
                idx[canon] = r
                seen_canon.add(canon)
    return idx

def present_worth_factor(rate_pct: float, years: int) -> float:
    """(1 - (1+r)^-n) / r ; se r=0, retorna n."""
    try:
        r = float(rate_pct) / 100.0
        n = int(years)
    except Exception:
        return np.nan
    if n <= 0:
        return np.nan
    if abs(r) < 1e-12:
        return float(n)
    return (1 - (1 + r) ** (-n)) / r

def compute_lcoe_from_lcc(lcc_value, annual_kwh: float, discount_rate_pct: float, years: int, load_pct: float):
    """
    LCOE aproximado a partir do LCC do bloco:
    LCOE = LCC / (consumo_anual_kWh * PWF * (PLS/100)).
    """
    if not is_number(lcc_value):
        return np.nan
    if annual_kwh is None or annual_kwh <= 0:
        return np.nan
    if load_pct is None or load_pct <= 0:
        return np.nan

    pwf = present_worth_factor(discount_rate_pct, years)
    if not is_number(pwf) or pwf <= 0:
        return np.nan

    served_kwh = float(annual_kwh) * (float(load_pct) / 100.0)
    if served_kwh <= 0:
        return np.nan
    return float(lcc_value) / (served_kwh * pwf)

# ---------- helpers específicos para o LCOE_breakdown ----------

def _parse_cost_level(val):
    """Converte 'High', 'Medium', 'Low' -> 'high'|'med'|'low'."""
    s = _ci(val)
    if "low" in s:
        return "low"
    if "medium" in s or "med" in s:
        return "med"
    if "high" in s:
        return "high"
    return None

def _diesel_price_code_from_value(v):
    """Mapeia valor em $/litro (3.2, 3.6, 4.4) para código '32'/'36'/'44'."""
    try:
        f = float(v)
    except Exception:
        return None
    candidates = {3.2: "32", 3.6: "36", 4.4: "44"}
    best_code, best_diff = None, float("inf")
    for val, code in candidates.items():
        diff = abs(f - val)
        if diff < best_diff:
            best_diff = diff
            best_code = code
    return best_code

def _norm_system_from_label(val):
    """Normaliza rótulos de sistema vindos da LCOE_breakdown."""
    s = _ci(val)
    if "diesel" in s and "pv" not in s and "battery" not in s:
        return "Diesel only"
    if "pv" in s and "battery" in s and "diesel" not in s:
        return "PV+battery"
    if "pv" in s and "battery" in s and "diesel" in s:
        return "PV+battery+diesel"
    return None

def _build_lcoe_lookup_from_breakdown(xls_path: Path) -> pd.DataFrame:
    """
    Lê a aba LCOE_breakdown e process_inputs e monta uma tabela:

      Region_Sheet, Discount_Rate_%, Percent_Load_Target,
      Cost_PVBatt, Cost_Diesel, Diesel_Price_Code, System,
      LCOE_$_per_kWh

    São 6 linhas neste arquivo: 3 sistemas (Diesel only, PV+batt, PBD)
    para o caso base (uma folha, ex: Lodwar_10) e 3 para o caso comparação
    (ex: Lusaka_20).
    """
    try:
        xls = pd.ExcelFile(xls_path)
        dfb = pd.read_excel(xls, sheet_name="LCOE_breakdown", header=None)
        dfi = pd.read_excel(xls, sheet_name="process_inputs", header=None)
    except Exception:
        return pd.DataFrame()

    # Linha com "sheet name" na process_inputs
    sheet_row = None
    for r in range(dfi.shape[0]):
        if _ci(dfi.iat[r, 0]) == "sheet name":
            sheet_row = r
            break
    if sheet_row is None:
        return pd.DataFrame()

    base_sheet = dfi.iat[sheet_row, 1]
    comp_sheet = dfi.iat[sheet_row, 3]

    # Linha "Percent of load served" na LCOE_breakdown
    pls_row = None
    for r in range(dfb.shape[0]):
        v = dfb.iat[r, 1]
        if isinstance(v, str) and "percent of load served" in v.lower():
            pls_row = r
            break
    if pls_row is None:
        return pd.DataFrame()

    pls_base = parse_number(dfb.iat[pls_row, 2])
    pls_comp = parse_number(dfb.iat[pls_row, 4])

    # Linhas de custos e preço do diesel
    pv_row = diesel_row = fuel_row = None
    for r in range(dfb.shape[0]):
        v = dfb.iat[r, 1]
        if not isinstance(v, str):
            continue
        s = v.lower()
        if "pv/battery costs" in s:
            pv_row = r
        elif "diesel generator costs" in s:
            diesel_row = r
        elif "diesel fuel price" in s:
            fuel_row = r

    pv_cost_base  = _parse_cost_level(dfb.iat[pv_row, 2]) if pv_row is not None else None
    pv_cost_comp  = _parse_cost_level(dfb.iat[pv_row, 4]) if pv_row is not None else None
    diesel_cost_base = _parse_cost_level(dfb.iat[diesel_row, 2]) if diesel_row is not None else None
    diesel_cost_comp = _parse_cost_level(dfb.iat[diesel_row, 4]) if diesel_row is not None else None
    diesel_price_base = _diesel_price_code_from_value(dfb.iat[fuel_row, 2]) if fuel_row is not None else None
    diesel_price_comp = _diesel_price_code_from_value(dfb.iat[fuel_row, 4]) if fuel_row is not None else None

    # Discount rate pela folha (ex: Lodwar_10 -> 10)
    def _disc_from_sheet(name):
        m = re.search(r"_(\d+)$", str(name))
        return int(m.group(1)) if m else None

    disc_base = _disc_from_sheet(base_sheet)
    disc_comp = _disc_from_sheet(comp_sheet)

    # Encontrar linha/coluna do rótulo LCOE
    lcoe_r = lcoe_c = None
    for r in range(dfb.shape[0]):
        for c in range(dfb.shape[1]):
            v = dfb.iat[r, c]
            if isinstance(v, str) and _ci(v) == "lcoe":
                lcoe_r, lcoe_c = r, c
                break
        if lcoe_r is not None:
            break
    if lcoe_r is None:
        return pd.DataFrame()

    # Colunas com números de LCOE nessa linha
    lcoe_cols = []
    for c in range(lcoe_c + 1, dfb.shape[1]):
        v = dfb.iat[lcoe_r, c]
        if is_number(v):
            lcoe_cols.append(c)
    if not lcoe_cols:
        return pd.DataFrame()

    # Linha com "RESULTS SUMMARY", "Base case" e "Comparison"
    rs_row = None
    base_label_col = comp_label_col = None
    for r in range(dfb.shape[0]):
        for c in range(dfb.shape[1]):
            v = dfb.iat[r, c]
            if isinstance(v, str) and "results summary" in v.lower():
                rs_row = r
                break
        if rs_row is not None:
            break
    if rs_row is not None:
        for c in range(dfb.shape[1]):
            v = dfb.iat[rs_row, c]
            if isinstance(v, str):
                s = v.strip().lower()
                if s == "base case":
                    base_label_col = c
                elif s == "comparison":
                    comp_label_col = c

    # Se conseguir localizar base/comparison, separamos as colunas
    base_cols, comp_cols = [], []
    if base_label_col is not None and comp_label_col is not None:
        for c in lcoe_cols:
            if base_label_col <= c < comp_label_col:
                base_cols.append(c)
            elif c >= comp_label_col:
                comp_cols.append(c)
    else:
        # fallback: divide ao meio
        half = len(lcoe_cols) // 2
        base_cols = lcoe_cols[:half]
        comp_cols = lcoe_cols[half:]

    # Procurar linha com nomes dos sistemas (Diesel only, PV+battery, etc)
    def _find_sys_row(dfb_local, lcoe_row, cols):
        def is_sys_label(val):
            if not isinstance(val, str):
                return False
            s = _ci(val)
            return ("diesel" in s) or ("battery" in s) or ("pv" in s)
        for rr in range(lcoe_row - 1, -1, -1):
            cnt = sum(1 for cc in cols if is_sys_label(dfb_local.iat[rr, cc]))
            if cnt >= 2:
                return rr
        return None

    sys_row = _find_sys_row(dfb, lcoe_r, lcoe_cols)
    if sys_row is None:
        return pd.DataFrame()

    records = []

    for scenario, cols, sheet, pls, pv_cost, diesel_cost, diesel_price, disc in [
        ("base", base_cols, base_sheet, pls_base, pv_cost_base, diesel_cost_base, diesel_price_base, disc_base),
        ("comp", comp_cols, comp_sheet, pls_comp, pv_cost_comp, diesel_cost_comp, diesel_price_comp, disc_comp),
    ]:
        if sheet is None or pls is None:
            continue
        for c in cols:
            sys_label = dfb.iat[sys_row, c]
            sys_norm = _norm_system_from_label(sys_label)
            if sys_norm is None:
                continue
            val = parse_number(dfb.iat[lcoe_r, c])
            records.append({
                "Region_Sheet": str(sheet),
                "Discount_Rate_%": int(disc) if disc is not None else None,
                "Percent_Load_Target": int(round(float(pls) * 100.0)),
                "Cost_PVBatt": pv_cost,
                "Cost_Diesel": (diesel_cost or None),
                "Diesel_Price_Code": diesel_price,
                "System": sys_norm,
                "LCOE_$_per_kWh": val,
            })

    df = pd.DataFrame(records)
    # normaliza costas para minúsculas (compatível com harvest_all_blocks)
    if not df.empty:
        df["Cost_PVBatt"] = df["Cost_PVBatt"].fillna("").astype(str).str.lower()
        df["Cost_Diesel"] = df["Cost_Diesel"].fillna("").astype(str).str.lower()
    return df

# ---------- extrator principal dos blocos ----------

def harvest_all_blocks(path: Path, annual_kwh: float, analysis_years: int):
    xls = pd.ExcelFile(path)
    region_sheets = list_region_sheets(xls)

    rows, logs = [], []

    for sheet in region_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        headers = find_block_headers(df)
        if not headers:
            continue

        disc_rate = parse_discount_rate_from_sheet(sheet)
        region_base = sheet.split("_")[0] if "_" in sheet else sheet

        for idx, h in enumerate(headers):
            start_row = h["row"] + 1
            end_row   = group_end_row(headers, idx, df.shape[0] - 1)

            pls_cols = find_pls_cols(df, h["row"], h["col"])
            if len(pls_cols) != len(PLS_ALLOWED):
                logs.append({
                    "Region_Sheet": sheet,
                    "Header": h["header_text"],
                    "Issue": f"PLS cols found={sorted(pls_cols.keys())}"
                })

            lbl_idx = build_label_index(df, start_row, end_row)

            def get_val(label_canonico, col):
                ridx = lbl_idx.get(label_canonico)
                if ridx is None or col >= df.shape[1]:
                    return np.nan
                return parse_number(df.iat[ridx, col])

            for pls, col in sorted(pls_cols.items(), reverse=True):  # 100..80
                pv_kw    = get_val("PV kW", col)
                batt_kw  = get_val("Battery kW", col)
                batt_kwh = get_val("Battery kWh", col)
                gen_kw   = get_val("Generator kW", col)

                fuel     = get_val("Fuel cost", col)
                lcc      = get_val("REopt LCC", col)

                pv_cap   = get_val("PV capital cost", col)
                pv_om    = get_val("PV O&M", col)
                bat_cap  = get_val("Battery capital cost", col)
                bat_om   = get_val("Battery O&M", col)
                d_cap    = get_val("Diesel capital cost", col)
                d_om_kw  = get_val("Diesel O&M (kW)", col)
                d_om_kwh = get_val("Diesel O&M (kWh)", col)

                # Em muitas planilhas isto não existe dentro do bloco.
                lcoe_val = get_val("LCOE ($/kWh)", col)

                system_raw = map_system(h["sys_code"])
                system_inf = infer_system(pv_kw, gen_kw, batt_kw, batt_kwh)
                system     = system_inf or system_raw

                # Normalizações para evitar contaminação residual
                if system == "Diesel only":
                    pv_kw, batt_kw, batt_kwh = 0.0, 0.0, 0.0
                elif system == "PV+battery":
                    gen_kw = 0.0

                computed_lcoe = False
                if not is_number(lcoe_val) and is_number(lcc):
                    lcoe_calc = compute_lcoe_from_lcc(
                        lcc_value=lcc,
                        annual_kwh=annual_kwh,
                        discount_rate_pct=disc_rate,
                        years=analysis_years,
                        load_pct=pls,
                    )
                    lcoe_val = lcoe_calc
                    computed_lcoe = is_number(lcoe_calc)

                row = {
                    "Region_Sheet": sheet,
                    "Region": region_base,
                    "Discount_Rate_%": disc_rate,
                    "Load_Profile": h["profile"],
                    "Cost_PVBatt": (h["pvbatt_cost"] or "").lower(),
                    "Cost_Diesel": h["diesel_cost"],
                    "Diesel_Price_Code": h["diesel_price"],
                    "System": system,
                    "System_raw": system_raw,
                    "Percent_Load_Target": pls,
                    "PV_kW": pv_kw,
                    "Battery_kW": batt_kw,
                    "Battery_kWh": batt_kwh,
                    "Generator_kW": gen_kw,
                    "Fuel_cost_$": fuel,
                    "REopt_LCC_$": lcc,
                    "PV_capex_$": pv_cap,
                    "PV_OandM_$": pv_om,
                    "Battery_capex_$": bat_cap,
                    "Battery_OandM_$": bat_om,
                    "Diesel_capex_$": d_cap,
                    "Diesel_OandM_kW_$": d_om_kw,
                    "Diesel_OandM_kWh_$": d_om_kwh,
                    "LCOE_$_per_kWh": lcoe_val,   # inicialmente pode ser NaN
                }
                rows.append(row)

                # Log por bloco
                miss = []
                if "LCOE ($/kWh)" not in lbl_idx:
                    miss.append("LCOE_row_missing_in_block")
                if computed_lcoe:
                    miss.append("LCOE_computed_from_LCC")
                logs.append({
                    "Region_Sheet": sheet,
                    "Header": h["header_text"],
                    "Block_rows": f"{start_row}-{end_row}",
                    "PLS": pls,
                    "System_raw": system_raw,
                    "System_final": system,
                    "Labels_found": sorted(lbl_idx.keys()),
                    "Notes": ",".join(miss) if miss else "",
                })

    df_all = pd.DataFrame(rows)
    df_log = pd.DataFrame(logs)

    if df_all.empty:
        return df_all, df_log

    # --------- preenche LCOE via aba LCOE_breakdown (fallback) ---------
    lcoe_lookup = _build_lcoe_lookup_from_breakdown(path)
    if not lcoe_lookup.empty:
        merge_keys = [
            "Region_Sheet",
            "Discount_Rate_%",
            "Percent_Load_Target",
            "Cost_PVBatt",
            "Cost_Diesel",
            "Diesel_Price_Code",
            "System",
        ]
        missing = [k for k in merge_keys if k not in df_all.columns]
        if not missing:
            df_all = df_all.merge(
                lcoe_lookup,
                how="left",
                on=merge_keys,
                suffixes=("", "_from_lcoe")
            )
            if "LCOE_$_per_kWh_from_lcoe" in df_all.columns:
                df_all["LCOE_$_per_kWh"] = df_all["LCOE_$_per_kWh"].where(
                    df_all["LCOE_$_per_kWh"].notna(),
                    df_all["LCOE_$_per_kWh_from_lcoe"]
                )
                df_all = df_all.drop(columns=["LCOE_$_per_kWh_from_lcoe"])

    # ordena e remove duplicatas
    order_sys = {"Diesel only": 0, "PV+battery": 1, "PV+battery+diesel": 2}
    df_all["__sys"] = df_all["System"].map(order_sys).fillna(9).astype(int)
    df_all = df_all.sort_values(
        ["Region", "Discount_Rate_%", "Load_Profile", "Percent_Load_Target", "__sys",
         "Cost_PVBatt", "Cost_Diesel", "Diesel_Price_Code"]
    ).drop(columns="__sys").drop_duplicates()

    return df_all, df_log

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True,
                    help="XLSM do REopt (1 - Microgrid_ReOpt_LCOE_Results_Explorer.xlsm)")
    ap.add_argument("--out", default="", help="CSV de saída (opcional)")
    ap.add_argument("--annual-kwh", type=float, default=DEFAULT_ANNUAL_KWH,
                    help="Consumo anual (kWh/ano) para calcular LCOE quando faltam valores (padrao: 19711).")
    ap.add_argument("--analysis-years", type=int, default=DEFAULT_ANALYSIS_YEARS,
                    help="Horizonte (anos) usado no fator-presente do LCOE (padrao: 20).")
    args = ap.parse_args()

    fpath = Path(args.file)
    if not fpath.exists():
        print("Arquivo não existe:", fpath)
        sys.exit(1)

    try:
        df, dlog = harvest_all_blocks(fpath, args.annual_kwh, args.analysis_years)
    except ImportError:
        print("Instale dependências: pip install pandas openpyxl numpy")
        raise

    if df.empty:
        print("Nenhum bloco encontrado.")
        sys.exit(2)

    out_csv = Path(args.out) if args.out else fpath.with_name("reopt_ALL_blocks_v3_8.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    log_csv = out_csv.with_name("reopt_MATCHES_log_v3_8.csv")
    dlog.to_csv(log_csv, index=False, encoding="utf-8")
    print("\nExtracao completa (v3.8)")
    print(f" - Linhas: {len(df)}")
    print(f" - CSV: {out_csv}")
    print(f" - Log: {log_csv}")
    print(f" • CSV: {out_csv}")
    print(f" • Log: {log_csv}")

    # checagens rápidas
    do  = df[df["System"]=="Diesel only"]
    pb  = df[df["System"]=="PV+battery"]
    pbd = df[df["System"]=="PV+battery+diesel"]

    print("\nChecagens de consistência:")
    print(" • Diesel only  -> PV=0 & Batt=0 & Gen>0:",
          int(((do["PV_kW"]==0) & (do["Battery_kW"]==0) & (do["Battery_kWh"]==0) & (do["Generator_kW"]>0)).sum()),
          "/", len(do))
    print(" • PV+battery   -> Gen=0:",
          int((pb["Generator_kW"]==0).sum()), "/", len(pb))
    print(" • PBD          -> PV>0 & Gen>0:",
          int(((pbd["PV_kW"]>0) & (pbd["Generator_kW"]>0)).sum()), "/", len(pbd))

    # cobertura do LCOE
    cov = int(df["LCOE_$_per_kWh"].notna().sum())
    print("\nCobertura LCOE ($/kWh):", f"{cov} de {len(df)} linhas com valor.")

    print("\nPrévia:")
    print(df.head(12))

if __name__ == "__main__":
    main()