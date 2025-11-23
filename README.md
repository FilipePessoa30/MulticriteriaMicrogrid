# Modelo multicriterio para operacao de microgrids renovaveis
_Titulo provisorio: **A Multi-Criteria Decision Model for Renewable Microgrid Operation: Integrating Economic, Environmental, Technical, and Social Perspectives through AHP and Comparative MCDM Methods**_

## Metodologia final
- Apoia decisoes de operacao de microgrids renovaveis, equilibrando custo, emissoes, confiabilidade e aspectos sociais.
- Integra dados reais de desempenho economico e tecnico com metodos multicriterio (MCDM), aplicando AHP/Fuzzy-AHP para calcular pesos e quatro metodos comparativos para ranquear alternativas.

### Etapas do modelo
| Etapa | Descricao | Ferramentas |
| --- | --- | --- |
| 1. Coleta e preparacao dos dados | Extracao das planilhas ReOpt LCOE Results e Load Profile Explorer, incluindo custos, consumo de combustivel, energia suprida e perfis de carga. | Excel / Python |
| 2. Estruturacao hierarquica (AHP) | Definicao da hierarquia de decisao: objetivo -> criterios (economico, ambiental, tecnico, social) -> alternativas (cenarios operacionais). | Python |
| 3. Calculo dos pesos via AHP / Fuzzy-AHP | Comparacoes par-a-par (escala Saaty 1-9 ou versao fuzzy). Verificacao de consistencia (CR < 0.1). | ahpy, scikit-fuzzy |
| 4. Normalizacao dos indicadores | Normalizacao min-max ou vetorial, distinguindo criterios de beneficio e de custo. | NumPy, Pandas |
| 5. Aplicacao dos metodos MCDM | Fuzzy-TOPSIS, VIKOR, COPRAS e MOORA para gerar rankings de alternativas sob diferentes perfis de decisao. | pymcdm, scikit-fuzzy |
| 6. Analise comparativa e sensibilidade | Convergencia de rankings (Spearman e Kendall tau), robustez dos pesos e parametros (v no VIKOR, fuzzificacao no TOPSIS). | Python / Matplotlib |

### Estrutura hierarquica AHP
- Nivel 1: Objetivo -> escolher a melhor estrategia operacional da microgrid.
- Nivel 2: Criterios -> economico, ambiental, tecnico, social.
- Nivel 3: Alternativas -> C1 (solar-dominante), C2 (eolica), C3 (hibrida), C4 (rede de apoio).

### Criterios de avaliacao
**Criterios economicos**
| Criterio | Metrica / Fonte | Fundamentacao |
| --- | --- | --- |
| Custo nivelado de energia (LCOE) | `LCOE_breakdown` | Indicador central de viabilidade economica (Bohra et al., 2021) |
| Custo total do ciclo de vida (TLCC) | Soma de `PV cost`, `Battery cost`, `Diesel O&M`, `Fuel_cost` | Mede investimento + operacao (Kiptoo et al., 2023) |
| Custo do combustivel | `Fuel_cost` | Reflete vulnerabilidade a precos fosseis (Panwar et al., 2017) |
| Payback / NPV | Derivado dos custos e receitas | Criterio de investimento (Kumar et al., 2018) |

**Criterios ambientais**
| Criterio | Calculo | Fundamentacao |
| --- | --- | --- |
| Emissoes totais de CO2 | `Fuel_cost * 2.68 / preco_diesel` | Fator padrao 2.68 kgCO2/L (Chaerul & Tompubolon, 2019) |
| Fracao renovavel (%) | `(energia PV + eolica) / total` | Indicador ambiental comum (Ghasemi et al., 2023) |
| Consumo de combustivel fossil (L ou kWh) | `Geracao diesel / eficiencia media` | (Bohra et al., 2021) |

**Criterios tecnicos / operacionais**
| Criterio | Calculo | Referencia |
| --- | --- | --- |
| Confiabilidade energetica (%) | `(energia suprida / demanda total) * 100` | (Pandey et al., 2021) |
| Uso do gerador diesel (%) | `energia gerada por diesel / total` | (Kiptoo et al., 2023) |
| Eficiencia da bateria / autoconsumo PV | `(energia devolvida / total carregada)` | (Chai et al., 2024) |
| Energia excedente (%) | `(geracao total - demanda util) / geracao total` | (Panwar et al., 2017) |

**Criterios sociais / contextuais**
| Criterio | Descricao | Referencia |
| --- | --- | --- |
| Atendimento da demanda comunitaria (%) | `(numero de perfis atendidos / total)` | (Kumar et al., 2017) |
| Acessibilidade energetica | `LCOE` ajustado por renda media | (Jamal et al., 2018) |
| Criacao de emprego local | Criterio qualitativo (escala 1-5) | (Bohra et al., 2019) |

### Perfis de decisao (pesos aplicados ao AHP)
| Perfil | Custo | Emissoes | Confiabilidade | Social | Aplicacao tipica |
| --- | --- | --- | --- | --- | --- |
| Economico | 0.6 | 0.2 | 0.2 | - | Minimizar custos totais |
| Sustentavel | 0.25 | 0.45 | 0.20 | 0.10 | Foco em emissoes e fracao renovavel |
| Resiliente | 0.3 | 0.1 | 0.5 | 0.1 | Priorizar confiabilidade e estabilidade |
| Social | 0.2 | 0.1 | 0.3 | 0.4 | Maximizar acesso e equidade |

### Metodos multicriterio aplicados
| Metodo | Tipo | Funcao | Vantagem |
| --- | --- | --- | --- |
| Fuzzy-TOPSIS | Fuzzy | Determinacao da similaridade com solucao ideal | Captura incertezas nas preferencias |
| VIKOR | Deterministico | Solucao de compromisso otima | Equilibrio entre criterios conflitantes |
| COPRAS | Deterministico | Avaliacao proporcional direta | Alta estabilidade e simplicidade |
| MOORA | Deterministico | Analise baseada em razao normalizada | Leve e eficiente para validacao cruzada |

### Analise de sensibilidade e validacao
- Comparacao dos rankings entre metodos (Spearman e Kendall tau).
- Variacao dos pesos AHP e dos parametros de cada metodo (v no VIKOR, niveis fuzzy no TOPSIS).
- Analise grafica de convergencia de rankings e robustez dos resultados.

### Ferramentas de implementacao
| Etapa | Ferramentas |
| --- | --- |
| AHP/Fuzzy-AHP | ahpy, scikit-fuzzy |
| MCDM (TOPSIS, VIKOR, COPRAS, MOORA) | pymcdm, numpy, pandas |
| Analise e visualizacao | matplotlib, plotly, streamlit |

### Conclusao metodologica
- O modelo utiliza AHP/Fuzzy-AHP para derivar pesos consistentes dos criterios, integrando-os aos metodos Fuzzy-TOPSIS, VIKOR, COPRAS e MOORA para analise comparativa.
- Essa combinacao garante transparencia na atribuicao dos pesos, consistencia matematica entre metodos, capacidade de tratar incertezas e robustez sob diferentes perfis decisorios.
- Resultado: modelo leve, reprodutivel e alinhado a literatura recente (2020-2025), adequado para estudos de decisao em microgrids renovaveis.
