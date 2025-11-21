# Projeto: Modelo Multicritério para Operação de Microgrids

## Visão geral

- **Objetivo:** apoiar decisões de operação de microgrids renováveis equilibrando custo, emissões e confiabilidade, comparando três métodos MCDM: TOPSIS (crisp), Fuzzy-TOPSIS e VIKOR.
- **Critérios (dados do REopt):** custo = LCOE (fallback LCC); emissões = proxy calculada a partir de Fuel_cost e preço do diesel (fator 2.68 kgCO₂/L, fallback Fuel_cost se faltar preço); confiabilidade = Percent_Load_Target.
- **Perfis de pesos:** econômico (0.6/0.2/0.2), sustentável (0.25/0.45/0.30), resiliente (0.3/0.1/0.6).
- **Fluxo:** extrair/ler CSV do REopt → aplicar TOPSIS/Fuzzy-TOPSIS/VIKOR → sensibilidade (pesos e v do VIKOR) → regra final (maioria → Borda → média de rank).
- **Limitação:** emissões são proxy; as alternativas são as do REopt (PV + bateria/diesel), não os cenários C1–C4 solares/eólicos/híbridos/rede.

## Scripts-chave

- `run_reopt_mcdm.py`: roda TOPSIS, Fuzzy-TOPSIS e VIKOR sobre `dados_preprocessados/reopt_ALL_blocks_v3_8.csv`. Saídas por perfil: `topsis_<perfil>.csv`, `fuzzy_topsis_<perfil>.csv`, `vikor_<perfil>.csv`, `all_ranks.csv`, `summary.json`.
  - Exemplo:
    ```powershell
    python run_reopt_mcdm.py `
      --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
      --out reopt_mcdm_results `
      --fuzziness 0.05 `
      --vikor_v 0.5
    ```
- `run_reopt_sensitivity.py`: sensibilidade para v = 0.3/0.5/0.7 (ajustável) e perfis. Gera `winners_sensitivity.csv` e `all_ranks_sensitivity.csv`.
  - Exemplo:
    ```powershell
    python run_reopt_sensitivity.py `
      --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
      --out reopt_mcdm_sensitivity `
      --vikor_v 0.3,0.5,0.7 `
      --fuzziness 0.05
    ```
- `finalize_profiles_from_sensitivity.py`: aplica regra final por perfil (maioria → Borda → média) usando `all_ranks_sensitivity.csv` (default v=0.5). Saída: `final_profile_decisions.csv`.
  - Exemplo:
    ```powershell
    python finalize_profiles_from_sensitivity.py `
      --ranks reopt_mcdm_sensitivity/all_ranks_sensitivity.csv `
      --vikor_v 0.5 `
      --out reopt_mcdm_sensitivity/final_profile_decisions.csv
    ```
- `build_mcdm_from_reopt.py`: extrator/normalizador original do REopt (mantido).

## Resultados obtidos (dados do REopt)

- **Econômico:** TOPSIS → Lodwar_PLS85_PV+battery; Fuzzy-TOPSIS → Lodwar_PLS100_PV+battery; VIKOR → Lusaka_PLS100_PV+battery.
- **Sustentável:** TOPSIS → Lodwar_PLS95_PV+battery; Fuzzy-TOPSIS → Lodwar_PLS100_PV+battery; VIKOR → Lodwar_PLS100_Diesel only (diverge).
- **Resiliente:** TOPSIS → Lodwar_PLS95_PV+battery; Fuzzy-TOPSIS → Lodwar_PLS100_PV+battery; VIKOR → Lusaka_PLS100_PV+battery.
- **Leitura:** PV+battery domina na maioria dos perfis/métodos; VIKOR diverge no sustentável por efeito da proxy de emissões e confiabilidade.

### Sensibilidade (v = 0.3/0.5/0.7)

- Econ: fuzzy-topsis = Lodwar_PLS100_PV+battery; vikor = Lusaka_PLS100_PV+battery (estável).
- Sustentável: fuzzy-topsis = Lodwar_PLS100_PV+battery; vikor = Diesel only (Lusaka v=0.3, Lodwar v=0.5, Accra v=0.7).
- Resiliente: fuzzy-topsis = Lodwar_PLS100_PV+battery; vikor = Lusaka_PLS100_PV+battery (estável).

### Decisão final por perfil (v=0.5)

Regra maioria → Borda → média, com os três métodos:

- Econômico: Final = **Lodwar_PLS100_PV+battery**.
- Sustentável: Final = **Lodwar_PLS100_PV+battery** (VIKOR divergiu para Diesel only, mas Borda consolidou PV+battery).
- Resiliente: Final = **Lodwar_PLS100_PV+battery**.
