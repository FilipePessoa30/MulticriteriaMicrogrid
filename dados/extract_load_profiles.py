# -*- coding: utf-8 -*-
"""
Extrai, do arquivo:
  2 - Microgrid_Load_Profile_Explorer.xlsx

Saídas (em --outdir):
  - household_total_low_income_24h.csv      (hour, value)
  - commercial_total_load_24h.csv           (hour, value)
  - village_profile_24h.csv                 (hour, total_load)
  - coincidence_factor_hourly_avg.csv       (hour, CF_hourly_avg) [opcional]
  - extract_load_profiles_log.txt           (log detalhado com posições/localização)

Robustez:
  - Busca a âncora (texto) em qualquer célula.
  - Extrai 24 valores em horizontal (mesma linha / linhas abaixo) ou vertical (abaixo),
    tolerando células vazias e colunas mescladas.
  - Fallback por varrimento global com heurística.
"""

import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd


# --------------------------- logging -----------------------------------------

def mk_logger(outdir):
    os.makedirs(outdir, exist_ok=True)
    logfile = os.path.join(outdir, "extract_load_profiles_log.txt")
    f = open(logfile, "w", encoding="utf-8")

    def log(msg):
        ts = datetime.now().strftime("[%Y-%m-%dT%H:%M:%S]")
        line = f"{ts} {msg}"
        print(line)
        f.write(line + "\n")
        f.flush()
    return log, f


# ------------------------ helpers --------------------------------------------

def norm_str(x):
    if x is None:
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = norm_str(x)
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    s = s.replace(" ", "").replace("\u00A0", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        if s.count(".") > 1:
            parts = s.split(".")
            s2 = "".join(parts[:-1]) + "." + parts[-1]
            try:
                return float(s2)
            except Exception:
                return np.nan
        return np.nan

def is_number(x):
    return np.isfinite(to_float(x))

def coerce_1d_clean(vals):
    """Converte lista de células para floats, remove NaN e retorna np.array."""
    arr = [to_float(v) for v in vals]
    arr = [v for v in arr if np.isfinite(v)]
    return np.array(arr, dtype=float)

# --------------------- busca por âncora e 24 valores -------------------------

def find_anchor(df, patterns):
    """Procura qualquer célula cujo texto case com um dos padrões."""
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            s = norm_str(df.iat[r, c])
            for pat in patterns:
                if pat.search(s):
                    return r, c, s
    return None, None, None

def try_horizontal_from(df, row, col_start, max_cols_ahead=80):
    """Na linha dada, varre à direita e coleta números (ignorando vazios), retorna 24 se possível."""
    vals = []
    coords = []
    for c in range(col_start, min(df.shape[1], col_start + max_cols_ahead)):
        v = df.iat[row, c]
        f = to_float(v)
        if np.isfinite(f):
            vals.append(f)
            coords.append((row, c))
        # tolera vazios; não quebra sequência
        if len(vals) >= 24:
            return np.array(vals[:24], dtype=float), coords[:24]
    return None, None

def try_vertical_from(df, row_start, col, max_rows_down=80):
    """Na coluna dada, varre para baixo e coleta números (ignorando vazios), retorna 24 se possível."""
    vals = []
    coords = []
    for r in range(row_start, min(df.shape[0], row_start + max_rows_down)):
        v = df.iat[r, col]
        f = to_float(v)
        if np.isfinite(f):
            vals.append(f)
            coords.append((r, col))
        if len(vals) >= 24:
            return np.array(vals[:24], dtype=float), coords[:24]
    return None, None

def extract_24h_near_anchor(df, anchor_r, anchor_c, log, label):
    """
    Estratégia:
      1) mesma linha, à direita da âncora
      2) linhas seguintes (até +10), à direita da âncora
      3) vertical: mesma coluna (e colunas seguintes até +5), abaixo da âncora
    """
    # 1) mesma linha
    v, coords = try_horizontal_from(df, anchor_r, anchor_c + 1, max_cols_ahead=120)
    if v is not None:
        log(f"  • {label}: 24 valores na linha {anchor_r} (coluna inicial {anchor_c+1}).")
        return v, coords

    # 2) linhas seguintes
    for dr in range(1, 11):
        r = anchor_r + dr
        if r >= df.shape[0]:
            break
        v, coords = try_horizontal_from(df, r, anchor_c + 1, max_cols_ahead=120)
        if v is not None:
            log(f"  • {label}: 24 valores na linha {r} (abaixo da âncora {anchor_r}, col {anchor_c+1}).")
            return v, coords

    # 3) vertical (mesma coluna e até +5 colunas à direita)
    for dc in range(0, 6):
        c = anchor_c + dc
        if c >= df.shape[1]:
            break
        v, coords = try_vertical_from(df, anchor_r + 1, c, max_rows_down=120)
        if v is not None:
            log(f"  • {label}: 24 valores em coluna {c} (abaixo da âncora {anchor_r},{anchor_c}).")
            return v, coords

    return None, None

def fallback_scan_anywhere(df, log, label):
    """
    Varrimento global (heurístico):
      - procura linha com >=24 números ao longo da linha (ignorando vazios), pega os primeiros 24
      - se falhar, procura coluna com >=24 números (ignorando vazios)
    """
    # horizontal
    best = None
    for r in range(df.shape[0]):
        cleaned = coerce_1d_clean(df.iloc[r, :].tolist())
        if cleaned.size >= 24:
            best = cleaned[:24]
            log(f"  • {label}: fallback horizontal (linha {r}).")
            return best
    # vertical
    for c in range(df.shape[1]):
        cleaned = coerce_1d_clean(df.iloc[:, c].tolist())
        if cleaned.size >= 24:
            best = cleaned[:24]
            log(f"  • {label}: fallback vertical (coluna {c}).")
            return best
    return None

# ------------------------ extratores principais -------------------------------

def extract_household_24h(xls_path, log):
    sheet = "HouseholdLoadProfiles"
    log(f"→ Residencial: procurando em '{sheet}' a âncora 'Total Low Income'...")
    df = pd.read_excel(xls_path, sheet_name=sheet, header=None, dtype=object, engine="openpyxl")

    pats = [
        re.compile(r"\bTotal\s*Low\s*Income\b", re.IGNORECASE),
        re.compile(r"\bLow\s*Income\b", re.IGNORECASE),
        re.compile(r"\bTotal\b.*\bLow\b", re.IGNORECASE),
    ]
    r, c, txt = find_anchor(df, pats)
    if r is None:
        raise RuntimeError("Não encontrei a âncora 'Total Low Income'.")

    log(f"  • Âncora encontrada em linha {r}, coluna {c}: '{txt}'")

    vals, coords = extract_24h_near_anchor(df, r, c, log, "Residencial")
    if vals is None:
        # fallback
        log("  • Residencial: tentando fallback por varrimento global...")
        vals = fallback_scan_anywhere(df, log, "Residencial")

    if vals is None or vals.size != 24:
        raise RuntimeError("Falha ao localizar 24 valores para o perfil residencial.")

    return vals

def extract_commercial_24h(xls_path, log):
    sheet = "CommercialLoadProfiles"
    log(f"→ Comercial: procurando em '{sheet}' a âncora 'Total Commercial Load'...")
    df = pd.read_excel(xls_path, sheet_name=sheet, header=None, dtype=object, engine="openpyxl")

    pats = [
        re.compile(r"\bTotal\s*Commercial\s*Load\b", re.IGNORECASE),
        re.compile(r"\bTotal\s*Commercial\b", re.IGNORECASE),
        re.compile(r"\bCommercial\s*Load\b", re.IGNORECASE),
    ]
    r, c, txt = find_anchor(df, pats)
    if r is None:
        raise RuntimeError("Não encontrei a âncora 'Total Commercial Load'.")

    log(f"  • Âncora encontrada em linha {r}, coluna {c}: '{txt}'")

    vals, coords = extract_24h_near_anchor(df, r, c, log, "Comercial")
    if vals is None:
        log("  • Comercial: tentando fallback por varrimento global...")
        vals = fallback_scan_anywhere(df, log, "Comercial")

    if vals is None or vals.size != 24:
        raise RuntimeError("Falha ao localizar 24 valores para o perfil comercial.")

    return vals


def extract_coincidence_factor_hourly_avg(xls_path, log, sheet="CoincidenceFactor_HouseholdLoad"):
    """
    Procura cabeçalhos:
      Hour (ou Hour of Day), Original Load Profile, Load Profile with 80% Coincidence Factor
    Lê todos os blocos possíveis (separados no Excel), normaliza hora para 1..24,
    e calcula média de CF = (80% / Original) por hora.
    """
    log(f"→ Coincidence Factor: varrendo '{sheet}'...")
    df = pd.read_excel(xls_path, sheet_name=sheet, header=None, dtype=object, engine="openpyxl")

    targ = {
        "hour": re.compile(r"^Hour(\s*of\s*Day)?$", re.IGNORECASE),
        "orig": re.compile(r"^Original\s*Load\s*Profile$", re.IGNORECASE),
        "cf80": re.compile(r"^Load\s*Profile\s*with\s*80%\s*Coincidence\s*Factor$", re.IGNORECASE),
    }

    header_rows = []
    for r in range(df.shape[0]):
        col_map = {}
        for c in range(df.shape[1]):
            s = norm_str(df.iat[r, c])
            if "hour" not in col_map and targ["hour"].search(s):
                col_map["hour"] = c
            if "orig" not in col_map and targ["orig"].search(s):
                col_map["orig"] = c
            if "cf80" not in col_map and targ["cf80"].search(s):
                col_map["cf80"] = c
        if set(col_map) == {"hour", "orig", "cf80"}:
            header_rows.append((r, col_map))

    if not header_rows:
        raise RuntimeError("Cabeçalhos Hour/Original/80% não encontrados.")

    rows_accum = []
    blocks = 0
    for hdr_row, cm in header_rows:
        hcol, ocol, ccol = cm["hour"], cm["orig"], cm["cf80"]
        r = hdr_row + 1
        got = 0
        while r < df.shape[0]:
            hv = df.iat[r, hcol]
            if hv is None or str(hv).strip() == "":
                # se duas vazias em sequência, considera fim do bloco
                nxt = r + 1
                if nxt >= df.shape[0]:
                    break
                hv2 = df.iat[nxt, hcol]
                if hv2 is None or str(hv2).strip() == "":
                    break
                r += 1
                continue

            try:
                h = int(float(str(hv).replace(",", ".")))
            except Exception:
                # não-numérico -> fim do bloco
                break

            if not (0 <= h <= 24 or 1 <= h <= 24):
                break

            o = to_float(df.iat[r, ocol])
            c = to_float(df.iat[r, ccol])
            rows_accum.append((h, o, c))
            r += 1
            got += 1

        if got > 0:
            blocks += 1

    if not rows_accum:
        raise RuntimeError("Não encontrei linhas de dados após os cabeçalhos.")

    tab = pd.DataFrame(rows_accum, columns=["hour_raw", "orig", "cf80"])

    def norm_hour(h):
        if 1 <= h <= 24:
            return h
        if 0 <= h <= 23:
            return h + 1
        return np.nan

    tab["hour"] = tab["hour_raw"].map(norm_hour)
    tab = tab.dropna(subset=["hour"]).copy()
    tab["hour"] = tab["hour"].astype(int)

    with np.errstate(divide="ignore", invalid="ignore"):
        cf = np.where(tab["orig"].to_numpy() != 0.0, tab["cf80"].to_numpy() / tab["orig"].to_numpy(), np.nan)
    tab["cf"] = cf

    cf_series = tab.groupby("hour", as_index=True)["cf"].mean()
    cf_series = cf_series.reindex(range(1, 25))
    cf_series.name = "CF_hourly_avg"

    counts = tab.groupby("hour")["cf"].count().reindex(range(1,25)).fillna(0).astype(int)
    missing = [int(h) for h in range(1,25) if counts.loc[h] == 0]
    log(f"  • CF: blocos lidos={blocks}; horas sem amostra={missing if missing else 'nenhuma'}")
    return cf_series


def build_village(house_24, comm_24, households, commercial_units):
    return households * np.asarray(house_24, dtype=float) + commercial_units * np.asarray(comm_24, dtype=float)

def save_24(vec, path, hour_base=1, value_col="value", fmt="%.6f"):
    hours = list(range(hour_base, hour_base + 24))
    out = pd.DataFrame({"hour": hours, value_col: vec})
    out.to_csv(path, index=False, float_format=fmt)


# ---------------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extrai perfis 24h e Coincidence Factor do arquivo de Load Profiles.")
    ap.add_argument("--excel", required=True, help="Caminho para '2 - Microgrid_Load_Profile_Explorer.xlsx'")
    ap.add_argument("--outdir", required=True, help="Diretório de saída")
    ap.add_argument("--households", type=float, default=100.0, help="Nº de residências para o agregado")
    ap.add_argument("--commercial_units", type=float, default=1.0, help="Nº de unidades comerciais")
    ap.add_argument("--hour_base", type=int, default=1, choices=[0,1], help="Rótulo 'hour': 1..24 (1) ou 0..23 (0)")
    args = ap.parse_args()

    xls = args.excel
    outdir = args.outdir
    hh = args.households
    cu = args.commercial_units
    hb = args.hour_base

    if not os.path.isfile(xls):
        raise FileNotFoundError(f"Arquivo não encontrado: {xls}")
    os.makedirs(outdir, exist_ok=True)

    log, flog = mk_logger(outdir)
    log("Início")
    log(f"Arquivo Excel: {xls}")

    try:
        # 1) Residencial
        house = extract_household_24h(xls, log)
        p_house = os.path.join(outdir, "household_total_low_income_24h.csv")
        save_24(house, p_house, hour_base=hb, value_col="value")
        log(f"✓ household_total_low_income_24h.csv salvo ({len(house)} valores).")

        # 2) Comercial
        comm = extract_commercial_24h(xls, log)
        p_comm = os.path.join(outdir, "commercial_total_load_24h.csv")
        save_24(comm, p_comm, hour_base=hb, value_col="value")
        log(f"✓ commercial_total_load_24h.csv salvo ({len(comm)} valores).")

        # 3) Coincidence Factor (opcional)
        try:
            cf_series = extract_coincidence_factor_hourly_avg(xls, log)
            cf_df = cf_series.reset_index()
            cf_df.columns = ["hour", "CF_hourly_avg"]
            if hb == 0:
                cf_df["hour"] = cf_df["hour"] - 1
            p_cf = os.path.join(outdir, "coincidence_factor_hourly_avg.csv")
            cf_df.to_csv(p_cf, index=False, float_format="%.6f")
            log("✓ coincidence_factor_hourly_avg.csv salvo.")
        except Exception as e:
            log(f"⚠ Coincidence Factor não extraído: {e}")

        # 4) Vila (agregado)
        village = build_village(house, comm, hh, cu)
        p_vil = os.path.join(outdir, "village_profile_24h.csv")
        save_24(village, p_vil, hour_base=hb, value_col="total_load")
        log("✓ village_profile_24h.csv salvo.")

        log("Concluído com sucesso.")

    except Exception as e:
        log(f"ERRO: {e}")
        raise
    finally:
        try:
            flog.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
