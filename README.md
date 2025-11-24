# Modelo multicriterio para operacao de microgrids renovaveis
_Titulo provisorio: **A Multi-Criteria Decision Model for Renewable Microgrid Operation Using AHP and Metaheuristic Weight Optimization: Balancing Cost, Emissions, and Reliability**_

## Metodologia final (AHP + otimizacao de pesos)
- Integra AHP (pesos derivados da literatura) com otimizacao via PSO, GA, SA e metaheuristicas de vizinhanca (VNS, Tabu, ILS, hibrida).
- Compara alternativas Diesel-only (C1), PV + Battery (C2) e Diesel + PV + Battery (C3) por metodos MCDM (Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA).
- Avalia robustez, consistencia e melhoria de desempenho com varias metricas (robustez de ranking, regret, Pareto, estabilidade, CV).

## Etapas do modelo
| Etapa | Descricao | Ferramentas |
| --- | --- | --- |
| 1. Coleta e preparacao dos dados | Extracao de custos, emissoes e confiabilidade das planilhas ReOpt LCOE Results e Load Profile Explorer. | Excel / Python |
| 2. Estruturacao hierarquica (AHP) | Objetivo -> Criterios (Economico, Ambiental, Tecnico, Social) -> Alternativas (C1-C3). | Python |
| 3. Atribuicao inicial de pesos (literatura) | Pesos validados por Bohra (2021), Song (2022), Lu (2021), Rocha-Buelvas (2025). | Base AHP |
| 4. Otimizacao dos pesos via metaheuristicas | PSO, GA, SA; VNS, Tabu, ILS e hibrida VNS+Tabu. | Python (pymoo, deap, pyswarm) |
| 5. Normalizacao e aplicacao dos metodos MCDM | Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA com pesos originais e otimizados. | Python (pymcdm, numpy, scikit-fuzzy) |
| 6. Avaliacao de melhoria | Comparacao de rankings e consistencia antes/depois (Spearman, Kendall tau, CR AHP). | Python / Matplotlib |
| 7. Analise de sensibilidade e visualizacao | Convergencia entre rankings, pesos e perfis decisorios. | Python / Streamlit |

## Estrutura hierarquica AHP
- Nivel 1 (objetivo): escolher a melhor estrategia operacional da microgrid.
- Nivel 2 (criterios): Economico, Ambiental, Tecnico (Confiabilidade), Social.
- Nivel 3 (alternativas): C1 Diesel-only; C2 PV + Battery; C3 Diesel + PV + Battery.
- Pesos iniciais: derivados da literatura e ajustados por metaheuristicas para perfis (economico, sustentavel, resiliente, social).

## Criterios e pesos iniciais (literatura)
| Criterio | Descricao | Faixa na literatura | Peso inicial | Referencia |
| --- | --- | --- | --- | --- |
| Economico | Custo total e LCOE | 0.3-0.6 | 0.40 | Bohra et al., 2021 |
| Ambiental | Emissoes totais e fracao renovavel | 0.2-0.5 | 0.30 | Rocha-Buelvas, 2025 |
| Tecnico (Confiabilidade) | Energia suprida, eficiencia da bateria | 0.2-0.4 | 0.20 | Song et al., 2022 |
| Social | Acessibilidade e impacto local | 0.1-0.2 | 0.10 | Lu et al., 2021 |

## Otimizacao dos pesos (PSO, GA, SA)
| Algoritmo | Estrategia | Funcao objetivo | Referencia |
| --- | --- | --- | --- |
| PSO | Busca coletiva de particulas | Minimizar CR e maximizar estabilidade de ranking | Yu et al., 2020 |
| GA | Selecao e cruzamento | Minimizar diferenca entre ranking AHP e MCDM + CR | Nemova et al., 2024 |
| SA | Busca local com resfriamento | Minimizar inconsistencia residual e desvios entre perfis | Rad et al., 2024 |

Funcao objetivo (exemplo): Min f(W) = alpha * CR(W) + beta * (1 - rho(W)), com alpha = beta = 0.5; CR(W): razao de consistencia; rho(W): correlacao de Spearman entre rankings pre e pos otimizacao.

## Metaheuristicas de vizinhanca (pesos)
| Metaheuristica | Caracteristica | Aplicacao em pesos | Referencias |
| --- | --- | --- | --- |
| VNS (Variable Neighborhood Search) | Alterna entre vizinhancas sistematicamente | Ajuste progressivo de pesos ate convergencia | Rodrigues 2022; Garcia-Gonzalez 2006 |
| Tabu Search (TS) | Evita ciclos e otimos locais | Ajuste iterativo de pesos com memoria | Matijevic 2023 |
| Iterated Local Search (ILS) | Reaproveita solucoes locais com perturbacoes | Refinamento fino dos pesos do AHP | Stutzle 2001 |
| Hybrid VNS + TS / VND | Combina vizinhancas adaptativas e memoria | Ajuste simultaneo multi-criterio de pesos | Wei 2019 |

## Metodos multicriterio aplicados
| Metodo | Tipo | Funcao | Vantagem |
| --- | --- | --- | --- |
| Fuzzy-TOPSIS | Fuzzy | Similaridade com solucao ideal | Captura incerteza nas preferencias |
| VIKOR | Deterministico | Solucao de compromisso | Equilibrio entre criterios conflitantes |
| COPRAS | Deterministico | Avaliacao proporcional direta | Estabilidade e simplicidade |
| MOORA | Deterministico | Razao normalizada | Leve e valida cruzada |

## Metricas de avaliacao (robustez/qualidade)
| Metrica | Usada em MCDM? | Base conceitual | Referencias |
| --- | --- | --- | --- |
| Robustez de ranking | Sim | Consistencia entre metodos | McPhail 2018; Fernandez 2001 |
| Regret / Utility | Sim | Minimiza arrependimento | Ning & You 2018; Groetzner 2021 |
| Dominancia / Pareto | Sim | Eficiencia multiobjetivo | Saif 2014; Lagaros 2007 |
| Consistencia (CR AHP) | Sim | Validacao de pesos AHP | Mufazzal 2021 |
| Estabilidade a ruido / cenario | Sim | Robustez sob incerteza | Kim 2015; Li 2018 |
| Validacao cruzada temporal | Sim | Generalizacao de decisao | Ning & You 2018 |

## Ferramentas e implementacao
| Componente | Ferramenta |
| --- | --- |
| AHP/Fuzzy-AHP | ahpy, scikit-fuzzy |
| Otimizacao de pesos | pyswarm, deap, simanneal |
| MCDM (TOPSIS, VIKOR, COPRAS, MOORA) | pymcdm, numpy, pandas |
| Analise de sensibilidade e graficos | matplotlib, plotly, streamlit |

## Scripts e exemplos
- `build_ahp_structure.py`: monta a hierarquia AHP e agrega metricas por alternativa a partir do CSV consolidado.
- `apply_mcdm_profiles.py`: aplica Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA para cada perfil (economico, sustentavel, resiliente, social) e gera CSV/JSON em `ahp_mcdm_results/`.
  ```powershell
  python apply_mcdm_profiles.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --out-dir ahp_mcdm_results
  ```
- `optimize_weights.py`: PSO/GA/SA (8 configs cada), executa `--runs` vezes (ex.: 30) e salva em `weight_optimization/`.
  ```powershell
  python optimize_weights.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --runs 30 `
    --seed 123 `
    --out-dir weight_optimization
  ```
- `optimize_weights_neighborhood.py`: VNS, Tabu, ILS e hibrido VNS+Tabu, executa `--runs` vezes (ex.: 30) e salva em `neighborhood_results/`.
  ```powershell
  python optimize_weights_neighborhood.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --runs 30 `
    --seed 321 `
    --out-dir neighborhood_results
  ```
- `evaluate_mcdm_quality.py`: avalia pesos (baseline + geneticos + vizinhanca) com robustez, regret, Pareto, estabilidade e CV. `--auto` varre `weight_optimization/`, `neighborhood_results/`, `summary_global.csv` e `runs_*.csv`.
  ```powershell
  python evaluate_mcdm_quality.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --auto `
    --out eval_quality_all.csv
  ```

## Resultados de otimizacao (regret_mean menor = melhor)
- Baseline: regret = 0.4377734737 (w: 0.40/0.30/0.20/0.10).
- Melhores vistos por metaheuristica (execucoes `runs_*.csv`):
  - VNS: regret = 0.1214687459 com w_cost=0.3206608381, w_emissions=0.0000000000, w_reliability=0.1560573560, w_social=0.5232818060.
  - Tabu Search: regret = 0.1348949419 com w_cost=0.5763681853, w_emissions=0.0048817464, w_reliability=0.2765960002, w_social=0.1421540680.
  - ILS: regret = 0.2004044126 com w_cost=0.3074530348, w_emissions=0.0181989677, w_reliability=0.2041180049, w_social=0.4702299925.
  - Hybrid VNS+Tabu: regret = 0.2343004034 com w_cost=0.3337491064, w_emissions=0.0312942730, w_reliability=0.1355886034, w_social=0.4993680173.

## Conclusao
- Parte de pesos validados (AHP), otimiza via PSO/GA/SA e tambem por metaheuristicas de vizinhanca, e compara em quatro metodos MCDM.
- Garante consistencia, robustez e transparencia ao escolher entre Diesel-only, PV + Battery e Hibrido.
- Combina fundamentacao teorica e otimizacao computacional para decisao multicriterio em microgrids renovaveis.
