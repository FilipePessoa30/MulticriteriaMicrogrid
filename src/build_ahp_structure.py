"""
Construcao da hierarquia AHP usando os dados processados do REopt.

Niveis:
  - Nivel 1: Objetivo -> Escolher a melhor estrategia operacional da microgrid
  - Nivel 2: Criterios -> Economico, Ambiental, Tecnico, Social
  - Nivel 3: Alternativas -> C1 (Diesel-only), C2 (PV + Battery), C3 (Diesel + PV + Battery)

Saida principal: dicionario com a estrutura da hierarquia e uma tabela
resumida de metricas por alternativa (media dos cenarios filtrados).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


DATA_PATH = Path("dados_preprocessados/reopt_ALL_blocks_v3_8.csv")

# Perfis de decisao (pesos nos criterios de nivel 2)
PROFILES = {
    "economic": {"cost": 0.60, "emissions": 0.20, "reliability": 0.20, "social": 0.0},
    "sustainable": {"cost": 0.25, "emissions": 0.45, "reliability": 0.20, "social": 0.10},
    "resilient": {"cost": 0.30, "emissions": 0.10, "reliability": 0.50, "social": 0.10},
    "social": {"cost": 0.20, "emissions": 0.10, "reliability": 0.30, "social": 0.40},
}

# Definicao dos criterios conforme o README
CRITERIA_TREE = {
    "goal": "Escolher a melhor estrategia operacional da microgrid",
    "criteria": {
        "economic": {
            "label": "Economico",
            "metrics": [
                {"id": "lcoe", "label": "Custo nivelado de energia (LCOE)", "source": "LCOE_breakdown ou LCOE_$_per_kWh"},
                {"id": "tlcc", "label": "Custo total do ciclo de vida (TLCC)", "source": "PV cost + Battery cost + Diesel O&M + Fuel_cost"},
                {"id": "fuel_cost", "label": "Custo do combustivel", "source": "Fuel_cost_$"},
                {"id": "payback_npv", "label": "Payback / NPV", "source": "Derivado dos custos e receitas"},
            ],
        },
        "environmental": {
            "label": "Ambiental",
            "metrics": [
                {"id": "emissions", "label": "Emissoes totais de CO2", "source": "Fuel_cost_$ * 2.68 / preco_diesel"},
                {"id": "renewable_fraction", "label": "Fracao renovavel (%)", "source": "(energia PV + eolica) / total"},
                {"id": "fossil_consumption", "label": "Consumo de combustivel fossil", "source": "Geracao diesel / eficiencia media"},
            ],
        },
        "technical": {
            "label": "Tecnico / Operacional",
            "metrics": [
                {"id": "reliability", "label": "Confiabilidade energetica (%)", "source": "(energia suprida / demanda total) * 100"},
                {"id": "diesel_use", "label": "Uso do gerador diesel (%)", "source": "energia gerada por diesel / total"},
                {"id": "battery_efficiency", "label": "Eficiencia da bateria / autoconsumo PV", "source": "energia devolvida / total carregada"},
                {"id": "excess_energy", "label": "Energia excedente (%)", "source": "(geracao total - demanda util) / geracao total"},
            ],
        },
        "social": {
            "label": "Social / Contextual",
            "metrics": [
                {"id": "demand_served", "label": "Atendimento da demanda comunitaria (%)", "source": "perfis atendidos / total"},
                {"id": "energy_accessibility", "label": "Acessibilidade energetica", "source": "LCOE ajustado por renda media"},
                {"id": "local_jobs", "label": "Criacao de emprego local", "source": "escala qualitativa 1-5"},
            ],
        },
    },
    "alternatives": {
        "C1": {"label": "Diesel-only"},
        "C2": {"label": "PV + Battery"},
        "C3": {"label": "Diesel + PV + Battery (Hibrido)"},
    },
}


def parse_diesel_map(map_str: str | None) -> Dict[str, float]:
    if not map_str:
        return {}
    mapping = {}
    for pair in map_str.split(","):
        if ":" not in pair:
            continue
        region, price = pair.split(":", 1)
        try:
            mapping[region.strip()] = float(price.strip())
        except ValueError:
            continue
    return mapping


def _add_derived_columns(df: pd.DataFrame, diesel_price_per_liter: float, diesel_price_map: Dict[str, float]) -> pd.DataFrame:
    """Gera colunas derivadas com base nos dados disponiveis no CSV."""
    df = df.copy()

    def price_for_row(row) -> float:
        region = str(row.get("Region", "")).strip()
        return float(diesel_price_map.get(region, diesel_price_per_liter))

    # TLCC como soma dos custos de CAPEX/O&M + combustivel (usa colunas disponiveis)
    cost_cols = [
        "PV_capex_$",
        "PV_OandM_$",
        "Battery_capex_$",
        "Battery_OandM_$",
        "Diesel_capex_$",
        "Diesel_OandM_kW_$",
        "Diesel_OandM_kWh_$",
        "Fuel_cost_$",
    ]
    present_cost_cols = [c for c in cost_cols if c in df.columns]
    if present_cost_cols:
        df["TLCC_$"] = df[present_cost_cols].sum(axis=1, skipna=True)
    else:
        df["TLCC_$"] = np.nan

    # Emissoes e consumo fossil (litros) como aproximacao via custo do combustivel
    if "Fuel_cost_$" in df.columns:
        df["Emissions_kgCO2"] = df.apply(lambda r: r["Fuel_cost_$"] * 2.68 / price_for_row(r), axis=1)
        df["Fossil_fuel_liters"] = df.apply(lambda r: r["Fuel_cost_$"] / price_for_row(r), axis=1)
    else:
        df["Emissions_kgCO2"] = np.nan
        df["Fossil_fuel_liters"] = np.nan

    # Uso do diesel como participacao do TLCC
    if "Fuel_cost_$" in df.columns and "TLCC_$" in df.columns:
        df["Diesel_cost_share"] = df["Fuel_cost_$"] / (df["TLCC_$"] + 1e-9)
    else:
        df["Diesel_cost_share"] = np.nan

    return df


def _default_alternative_filters() -> Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    """
    Filtros simples para mapear os cenarios do REopt nas quatro alternativas AHP.

    - C1 Diesel-only: gerador a diesel isolado
    - C2 PV + Battery: sistemas solares com bateria, sem diesel
    - C3 Diesel + PV + Battery: sistema hibrido com apoio do diesel
    """

    return {
        "C1": lambda d: d[d["System"] == "Diesel only"],
        "C2": lambda d: d[
            d["System"].str.contains("PV+battery", na=False, regex=False)
            & ~d["System"].str.contains("diesel", na=False, case=False)
        ],
        "C3": lambda d: d[d["System"] == "PV+battery+diesel"],
    }


def aggregate_metrics_by_alternative(
    df: pd.DataFrame,
    diesel_price_per_liter: float,
    diesel_price_map: Dict[str, float] | None = None,
    alternative_filters: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """
    Aplica filtros de alternativas e retorna tabela com metricas agregadas (media).

    Apenas metricas presentes sao retornadas. Valores ausentes permanecem como NaN.
    """
    df = _add_derived_columns(df, diesel_price_per_liter, diesel_price_map or {})
    filters = alternative_filters or _default_alternative_filters()

    metric_columns: List[str] = []
    # economico
    if "LCOE_$_per_kWh" in df.columns:
        metric_columns.append("LCOE_$_per_kWh")
    if "TLCC_$" in df.columns:
        metric_columns.append("TLCC_$")
    if "Fuel_cost_$" in df.columns:
        metric_columns.append("Fuel_cost_$")
    # ambiental
    for col in ("Emissions_kgCO2", "Fossil_fuel_liters"):
        if col in df.columns:
            metric_columns.append(col)
    # tecnico
    if "Percent_Load_Target" in df.columns:
        metric_columns.append("Percent_Load_Target")
    if "Diesel_cost_share" in df.columns:
        metric_columns.append("Diesel_cost_share")

    alt_rows = {}
    for alt_id, filt in filters.items():
        subset = filt(df)
        if subset.empty:
            alt_rows[alt_id] = {c: np.nan for c in metric_columns}
            continue
        alt_rows[alt_id] = subset[metric_columns].mean(numeric_only=True).to_dict()

    table = pd.DataFrame.from_dict(alt_rows, orient="index")
    table.index.name = "Alternative"
    return table


def build_ahp_structure(
    df: pd.DataFrame,
    diesel_price_per_liter: float = 1.0,
    diesel_price_map: Dict[str, float] | None = None,
    alternative_filters: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
) -> Dict:
    """
    Monta dicionario com a hierarquia AHP e as metricas agregadas por alternativa.
    """
    metrics_table = aggregate_metrics_by_alternative(
        df,
        diesel_price_per_liter=diesel_price_per_liter,
        diesel_price_map=diesel_price_map,
        alternative_filters=alternative_filters,
    )
    structure = {
        "ahp": CRITERIA_TREE,
        "metrics_by_alternative": metrics_table.to_dict(orient="index"),
        "diesel_price_per_liter": diesel_price_per_liter,
        "source_csv": str(DATA_PATH),
        "profiles": PROFILES,
    }
    return structure


def save_json(structure: Dict, out_path: Path) -> None:
    out_path.write_text(json.dumps(structure, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Constroi estrutura AHP a partir do CSV processado do REopt.")
    parser.add_argument("--csv", type=Path, default=DATA_PATH, help="Caminho para dados_preprocessados/reopt_ALL_blocks_v3_8.csv")
    parser.add_argument("--diesel-price", type=float, default=1.0, help="Preco do diesel (USD/L) default")
    parser.add_argument("--diesel-map", type=str, help="Mapa Region:preco separados por virgula (ex: Accra:0.95,Lusaka:1.16)")
    parser.add_argument("--out", type=Path, default=Path("ahp_structure.json"), help="Arquivo JSON de saida")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    price_map = parse_diesel_map(args.diesel_map)
    structure = build_ahp_structure(df, diesel_price_per_liter=args.diesel_price, diesel_price_map=price_map)

    save_json(structure, args.out)
    # resumo rapido no console
    metrics_df = pd.DataFrame.from_dict(structure["metrics_by_alternative"], orient="index")
    print("Hierarquia AHP pronta.")
    print(f"- Objetivo: {CRITERIA_TREE['goal']}")
    print("- Alternativas:", ", ".join(f"{k}={v['label']}" for k, v in CRITERIA_TREE["alternatives"].items()))
    print("- Perfis:", ", ".join(PROFILES.keys()))
    print("\nMetricas agregadas (media):")
    print(metrics_df.round(4).fillna("NA"))
    print(f"\nJSON salvo em: {args.out}")


if __name__ == "__main__":
    main()
