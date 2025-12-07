# Modelo multicriterio para operacao de microgrids renovaveis
_Titulo provisorio: **A Multi-Criteria Decision Model for Renewable Microgrid Operation Using AHP and Metaheuristic Weight Optimization: Balancing Economic, Environmental, Technical, and Social Criteria**_

## Metodologia final (AHP + otimizacao de pesos)
- Integra AHP (pesos da literatura) com otimizacao via PSO, GA, SA e vizinhancas (VNS, Tabu, ILS, hibrida).
- Compara C1 Diesel-only, C2 PV + Battery, C3 Diesel + PV + Battery por Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA.
- Avalia robustez, consistencia e melhoria com metricas de ranking, regret, Pareto, estabilidade e CV.

## Etapas do modelo
| Etapa | Descricao | Ferramentas |
| --- | --- | --- |
| 1. Coleta e preparacao | Custos, emissoes, confiabilidade (ReOpt LCOE Results, Load Profile Explorer). | Excel / Python |
| 2. AHP hierarquico | Objetivo -> Criterios (Econ, Ambient, Tec, Social) -> Alternativas (C1-C3). | Python |
| 3. Pesos iniciais (literatura) | Bohra 2021; Song 2022; Lu 2021; Rocha-Buelvas 2025. | Base AHP |
| 4. Otimizacao de pesos | PSO/GA/SA + VNS/Tabu/ILS/Hybrid. | pymoo, deap, pyswarm |
| 5. Aplicacao MCDM | Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA. | pymcdm, numpy, scikit-fuzzy |
| 6. Avaliacao de melhoria | Spearman, Kendall, CR AHP. | Python / Matplotlib |
| 7. Sensibilidade/visualizacao | Convergencia de rankings, pesos, perfis. | Python / Streamlit |

## Estrutura AHP
- Nivel 1: escolher a melhor estrategia operacional da microgrid.
- Nivel 2: criterios Economico, Ambiental, Tecnico (Confiabilidade), Social.
- Nivel 3: alternativas C1 Diesel-only; C2 PV + Battery; C3 Diesel + PV + Battery.

## Criterios e pesos iniciais (literatura)
| Criterio | Descricao | Peso representativo | Referencia/Contexto |
| --- | --- | --- | --- |
| Economico | Custo total, CoE, NPC | 0.6923 | Bohra et al., 2021 (valor fixo para cenarios com foco economico) |
| Ambiental | Fracao renovavel, pegada de carbono, impacto ecossistema | 0.3328 | Rocha-Buelvas, 2025 (maior peso encontrado no estudo, FAHP SC4 A4) |
| Tecnico (Confiabilidade) | Confiabilidade, economia, tecnologia, protecao ambiental | 0.3492 | Song et al., 2022 (peso abrangente DEMATEL-AHP/CRITIC-EWM) |
| Social | Aceitacao social, empregos, obstaculos locais | 0.3351 | Rocha-Buelvas, 2025 (maior peso encontrado, AHP SC1 A5) |

### Validacao do AHP
| Metodo | Nome completo | Uso tipico | Referencia |
| --- | --- | --- | --- |
| AHP | Analytic Hierarchy Process | Derivacao de pesos de criterios (base de hibridos) | Ogrodnik, 2019 |

## Precos de diesel por regiao (planilha "1 - Microgrid_ReOpt_LCOE_Results_Explorer.xlsm")
| Pais / Regiao | USD/gal | USD/L | Observacao |
| --- | --- | --- | --- |
| Tanzania (Lodwar) | 3.20 | ~0.85 | 0.85 = 3.20 / 3.785 |
| Gana (Accra) | 3.60 | ~0.95 | 0.95 = 3.60 / 3.785 |
| Zambia (Lusaka) | 4.40 | ~1.16 | 1.16 = 4.40 / 3.785 |
- Conversao usada: USD/L = USD/gal / 3.785 (ex.: 3.60 / 3.785 ≈ 0.95 USD/L).
- Mapa passado aos scripts via `--diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85"`; fator de emissao: 2.67 kg CO₂/L.

## Meta-heuristicas geneticas
| Meta-heuristica | Objetivo Especifico para Validacao | Referencia |
| --- | --- | --- |
| PSO-SA Hibrido | Refinar matrizes AHP inconsistentes, minimizando a Inconsistencia (lambda max - N) e a Discrepancia (DI). | Sarani Rad et al., 2024 |
| HC, SA, PSO | Encontrar os valores otimos de preferencias de Objetos Caracteristicos (COs) para um modelo MCDM (COMET). | Kizielewicz & Salabun, 2020 |
| ABC (Artificial Bee Colony) | Otimizar e aprender os pesos dos criterios (criteria weights), que sao usados como input para o metodo MCDM VIKOR, minimizando o custo de classificacao de inventario (Funcao Objetivo). | Cherif & Ladhari, 2016 |

Funcao objetivo exemplo: Min f(W) = alpha * CR(W) + beta * (1 - rho(W)), alpha = beta = 0.5; CR(W) = razao de consistencia; rho(W) = Spearman pre vs pos.

## Metaheuristicas de vizinhanca
| Meta-heuristica (Tipo de Vizinhança) | Objetivo Especifico para Validacao | Referencia |
| --- | --- | --- |
| Hill-Climbing (HC), Simulated Annealing (SA) | Usados para encontrar os valores otimos de preferencias de Objetos Caracteristicos (COs), otimizando os parametros do modelo MCDM (COMET) e minimizando o erro de preferencia absoluto. | Kizielewicz & Salabun, 2020 |
| Otimizacao Local Iterativa (Metodo das Decisoes Herdadas) | Solucionar problemas complexos de Otimizacao Multicriterio Nao-Linear (MCO) via varias otimizacoes locais iniciadas em pontos na vizinhanca da decisao anterior, usando funcao escalarizante. | Lotova et al., 2019 |
| ILS (Iterative Local Search) | Otimizar uma funcao objetiva escalarizada que combina dois criterios com coeficiente de ponderacao (mu) em problema bi-criterio, validando busca em vizinhanca para solucoes de compromisso com pesos MCDM. | Aqil & Allali, 2021 |
| Tabu Search Multi-Vizinhanca (MTS) | Aplicar tabu search com quatro vizinhancas restritas para otimizar alocacao multicriterio com multiplos orcamentos (MMCP), maximizando lucro total. | Liu & Pan, 2024 |
| WILB (Weighted Iterated Local Branching) | Busca local iterativa guiada por restricoes de vizinhanca (local branching) para otimizacao binaria ponderada, direcionando a busca por probabilidade de flip das variaveis. | Rodrigues et al., 2022 |

## Parametros MCDM (referencias)
| Parametro | Valor | Interpretacao | Referencias |
| --- | --- | --- | --- |
| Fuzziness | 0.05 | Nivel de incerteza baixo/moderado em MCDM fuzzy | Otay & Kahraman, 2022; Peng et al., 2020 |
| VIKOR v | 0.5 | Solucao de compromisso balanceada (maioritario = minoritario) | Kahraman & Otay, 2021; Sivaraja et al., 2020 |

## Metodos MCDM
| Metodo | Tipo | Entrada | Vantagem principal | Aplicabilidade | Referencia |
| --- | --- | --- | --- | --- | --- |
| Fuzzy-TOPSIS | Fuzzy | Matriz fuzzy + pesos linguisticos | Trata incerteza | Julgamentos subjetivos | Zamani-Sabzi et al., 2016 |
| VIKOR | Deterministico | Valores normalizados + v | Solucao de compromisso | Trade-offs custo x emissao x confiabilidade | Salabun et al., 2020 |
| COPRAS | Deterministico | Matriz normalizada proporcional | Estavel e simples | Validacao e robustez | Radulescu & Radulescu, 2024 |
| MOORA | Deterministico | Razao normalizada (beneficio/custo) | Simples e eficiente | Referencia base e consistencia | Singh & Pathak, 2024 |

## Metricas de avaliacao
| Metrica | Usada em MCDM? | Base | Referencias |
| --- | --- | --- | --- |
| Robustez de ranking | Sim | Consistencia entre metodos | McPhail 2018; Fernandez 2001 |
| Regret / Utility | Sim | Minimiza arrependimento | Ning & You 2018; Groetzner 2021 |
| Dominancia / Pareto | Sim | Eficiencia multiobjetivo | Saif 2014; Lagaros 2007 |
| Consistencia (CR AHP) | Sim | Validacao de pesos AHP | Mufazzal 2021 |
| Estabilidade a ruido / cenario | Sim | Robustez a incerteza | Kim 2015; Li 2018 |
| Validacao cruzada temporal | Sim | Generalizacao de decisao | Ning & You 2018 |

## Ferramentas
| Componente | Ferramenta |
| --- | --- |
| AHP/Fuzzy-AHP | ahpy, scikit-fuzzy |
| Otimizacao de pesos | pyswarm, deap, simanneal |
| MCDM | pymcdm, numpy, pandas |
| Sensibilidade/graficos | matplotlib, plotly, streamlit |

## Scripts e exemplos
- `build_ahp_structure.py`: agrega metricas por alternativa.
- `apply_mcdm_profiles.py`: roda MCDM por perfil baseline.
  ```powershell
  python apply_mcdm_profiles.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --out-dir ahp_mcdm_results
  ```
- `optimize_weights.py`: PSO/GA/SA, 8 configs cada, multiplas execucoes.
  ```powershell
  python optimize_weights.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --runs 30 `
    --seed 123 `
    --out-dir weight_optimization
  ```
- `optimize_weights_neighborhood.py`: VNS, Tabu, ILS, Hybrid.
  ```powershell
  python optimize_weights_neighborhood.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --runs 30 `
    --seed 321 `
    --out-dir neighborhood_results
  ```
- `evaluate_mcdm_quality.py`: avalia pesos (baseline + geneticos + vizinhanca); `--auto` varre `weight_optimization/`, `neighborhood_results/`, `summary_global.csv` e `runs_*.csv`.
  ```powershell
  python evaluate_mcdm_quality.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --auto `
    --out eval_quality_all.csv
  ```
- `apply_optimized_weights.py`: aplica MCDM com os melhores pesos (VNS/Tabu/ILS/Hybrid).
  ```powershell
  python apply_optimized_weights.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.05 `
    --vikor-v 0.5 `
    --out-dir optimized_mcdm_results
  ```

## Resultados de otimizacao (regret)
- Baseline: regret = 0.4163999763 (w: 0.6923/0.3328/0.3492/0.3351).
- Melhores por metaheuristica (execucoes `runs_*.csv`):
  - VNS: 0.1214687459 (0.3206608381 / 0.0000000000 / 0.1560573560 / 0.5232818060).
  - Tabu: 0.1348949419 (0.5763681853 / 0.0048817464 / 0.2765960002 / 0.1421540680).
  - ILS: 0.2004044126 (0.3074530348 / 0.0181989677 / 0.2041180049 / 0.4702299925).
  - Hybrid: 0.2343004034 (0.3337491064 / 0.0312942730 / 0.1355886034 / 0.4993680173).

## Resultados MCDM (baseline x otimizados)
- Baseline (perfis):
  - Economico: C2 liderou Fuzzy-TOPSIS, COPRAS e MOORA; VIKOR elegeu C3 (C2 em 2o).
  - Sustentavel, Resiliente, Social: C2 em 1o nos quatro metodos.
- Otimizados (VNS_best, Tabu_best, ILS_best, Hybrid_best): C2 em 1o nos quatro metodos (Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA) em todos os conjuntos.
- Arquivos: baseline em `ahp_mcdm_results/`, otimizados em `optimized_mcdm_results/`.

## Conclusao
- Parte de pesos validados (AHP), otimiza via PSO/GA/SA e vizinhanca, e compara em quatro metodos MCDM.
- Vencedor consistente: C2 (PV + Battery) na maioria dos casos; unica divergencia: baseline economico no VIKOR (C3).
- Modelo transparente, robusto e reprodutivel para decisao multicriterio em microgrids renovaveis.
