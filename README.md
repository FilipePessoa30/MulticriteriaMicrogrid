# Projeto: Modelo Multicritério para Operação de Microgrids

## Metodologia (resumo)
- **Problema/Motivação:** apoiar decisões de operação de microgrids renováveis equilibrando custo, emissões e confiabilidade, de forma simples e reprodutível.
- **Objetivo:** comparar Fuzzy-TOPSIS e VIKOR (mais TOPSIS crisp como baseline), com preferências crisp/fuzzy e análise de sensibilidade.
- **Fluxo:** coleta/simulação → indicadores (custo, emissões, confiabilidade) → matriz de decisão (benefício/custo) → Fuzzy-TOPSIS + VIKOR → comparação de rankings → sensibilidade → regra de decisão final (maioria → Borda → custo → média de rank).
- **Cenários esperados:** C1 solar, C2 eólica, C3 híbrido, C4 rede (quando disponíveis nos dados) ou alternativas reais do REopt (PV+battery, PV+battery+diesel, Diesel only, etc).

## Scripts principais
- `run_reopt_mcdm.py`: aplica TOPSIS, Fuzzy-TOPSIS e VIKOR diretamente em `reopt_ALL_blocks_v3_8.csv` usando critérios: custo=LCOE (fallback LCC), emissões proxy=Fuel_cost_$, confiabilidade=PLS. Gera ranks por perfil (econômico/sustentável/resiliente) e `summary.json`.
- `build_mcdm_from_reopt.py`: extrai e normaliza blocos do REopt.
- (Opcional, se recriar) `evaluate_portfolio_mcdm.py`, `finalize_mcdm_results.py`, `audit_consensus.py`: consolidação/auditoria de múltiplas pastas `_PLS*`.

## Como rodar nos dados processados
```powershell
python run_reopt_mcdm.py `
  --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
  --out reopt_mcdm_results `
  --fuzziness 0.05 `
  --vikor_v 0.5
```
Saídas: `topsis_<perfil>.csv`, `fuzzy_topsis_<perfil>.csv`, `vikor_<perfil>.csv`, `all_ranks.csv`, `summary.json`.

## Resultados obtidos (dados do REopt)
Critérios: custo = LCOE (fallback LCC); emissões = calculadas de forma aproximada a partir de Fuel_cost$/L e Diesel_Price_Code (fator 2.68 kgCO₂/L; fallback Fuel_cost se faltar preço); confiabilidade = Percent_Load_Target. Pesos: econômico 0.6/0.2/0.2, sustentável 0.25/0.45/0.30, resiliente 0.3/0.1/0.6. Observação: emissões são proxy de custo de diesel; sem dados de consumo/geração a diesel, não há CO₂ real.

- **Econômico:** TOPSIS → Lodwar_PLS85_PV+battery; Fuzzy-TOPSIS → Lodwar_PLS100_PV+battery; VIKOR → Lusaka_PLS100_PV+battery.
- **Sustentável:** TOPSIS → Lodwar_PLS95_PV+battery; Fuzzy-TOPSIS → Lodwar_PLS100_PV+battery; VIKOR → Lodwar_PLS100_Diesel only (divergência).
- **Resiliente:** TOPSIS → Lodwar_PLS95_PV+battery; Fuzzy-TOPSIS → Lodwar_PLS100_PV+battery; VIKOR → Lusaka_PLS100_PV+battery.
- **Leitura:** PV+battery domina na maioria dos perfis/métodos; VIKOR diverge no sustentável, influenciado pela proxy de emissões e confiabilidade.

### Sensibilidade (v = 0.3/0.5/0.7 no VIKOR)
Comando: `python run_reopt_sensitivity.py --csv dados_preprocessados/reopt_ALL_blocks_v3_8.csv --out reopt_mcdm_sensitivity --vikor_v 0.3,0.5,0.7 --fuzziness 0.05`

Resumo dos vencedores (fuzzy_topsis vs vikor):
- **Econômico:** Fuzzy-TOPSIS = Lodwar_PLS100_PV+battery; VIKOR = Lusaka_PLS100_PV+battery (estável).
- **Sustentável:** Fuzzy-TOPSIS = Lodwar_PLS100_PV+battery; VIKOR = Diesel only (Lusaka v=0.3, Lodwar v=0.5, Accra v=0.7).
- **Resiliente:** Fuzzy-TOPSIS = Lodwar_PLS100_PV+battery; VIKOR = Lusaka_PLS100_PV+battery (estável).

### Decisão final por perfil (v=0.5, regra: maioria → Borda → média)
Gerado com `finalize_profiles_from_sensitivity.py` sobre `reopt_mcdm_sensitivity/all_ranks_sensitivity.csv`:
- Econômico: TOPSIS=Lodwar_PLS85_PV+battery | Fuzzy=Lodwar_PLS100_PV+battery | VIKOR=Lusaka_PLS100_PV+battery → **Final: Lodwar_PLS100_PV+battery (Borda)**.
- Sustentável: TOPSIS=Lodwar_PLS95_PV+battery | Fuzzy=Lodwar_PLS100_PV+battery | VIKOR=Lodwar_PLS100_Diesel only → **Final: Lodwar_PLS100_PV+battery (Borda)**.
- Resiliente: TOPSIS=Lodwar_PLS95_PV+battery | Fuzzy=Lodwar_PLS100_PV+battery | VIKOR=Lusaka_PLS100_PV+battery → **Final: Lodwar_PLS100_PV+battery (Borda)**.

**Nota:** Os resultados refletem as alternativas do REopt (PV + bateria/diesel) e emissões aproximadas; para cenários C1–C4 (solar/eólica/híbrido/rede) e CO₂ real, seria necessário gerar uma matriz específica com esses cenários e dados de consumo/geração a diesel.

## Observação sobre cenários
Os rankings e a decisão final refletem as alternativas do REopt (PV + bateria/diesel). Para cumprir totalmente os cenários C1–C4 (solar, eólica, híbrido, rede de apoio), é necessário gerar uma matriz específica com esses cenários (e emissões reais) e reaplicar os métodos. Enquanto isso, os resultados aqui se aplicam aos arranjos disponíveis no REopt.

## Próximos passos alinhados à metodologia
- Incluir cenários solares/eólicos/híbridos/rede reais (C1–C4) ou gerar um CSV com esses cenários e rerodar os métodos.
- Rodar sensibilidade de pesos (econ/sust/res) e do parâmetro `vikor_v` (ex.: 0.3/0.5/0.7) para registrar inversões de ranking.
- Se emissões reais estiverem disponíveis, substituir a proxy de Fuel_cost por CO₂ calculado.
- Documentar a regra de decisão final por perfil (maioria → Borda → custo → média de rank) e, se desejar, gerar tabela consolidada por perfil com essa regra aplicada nos três métodos.

## Notas
- Todos os scripts usam Python 3 + pandas/numpy; para plots/Excel, matplotlib/seaborn/xlsxwriter (se instalados).
- Se recriar a camada de auditoria/portfólio, repor os scripts `evaluate_portfolio_mcdm.py`, `finalize_mcdm_results.py`, etc., conforme necessário.
