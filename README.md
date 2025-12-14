# Modelo multicriterio para operacao de microgrids renovaveis

_Titulo provisorio: **A Multi-Criteria Decision Model for Renewable Microgrid Operation Using AHP and Metaheuristic Weight Optimization: Balancing Economic, Environmental, Technical, and Social Criteria**_

## Metodologia final (AHP + otimizacao de pesos)

- Integra AHP (pesos da literatura) com otimizacao via PSO, GA, SA e vizinhancas (VNS, Tabu, ILS, hibrida).
- Compara C1 Diesel-only, C2 PV + Battery, C3 Diesel + PV + Battery por Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA.
- Avalia robustez, consistencia e melhoria com metricas de ranking, regret, Pareto, estabilidade e CV.

## Funcoes objetivo ativas (3)
- entropy: peso maior para criterios com maior dispersao das metricas.
- critic: pondera variancia e penaliza criterios colineares (redundancia).
- bayes: combina prior BASE_WEIGHTS com pesos objetivos (fusao literatura + dados).

## Etapas do modelo

| Etapa                                              | Descricao                                                                     | Ferramentas                 |
| -------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------- |
| 1. Coleta e preparacao                             | Custos, emissoes, confiabilidade (ReOpt LCOE Results, Load Profile Explorer). | Excel / Python              |
| 2. AHP hierarquico                                 | Objetivo -> Criterios (Econ, Ambient, Tec, Social) -> Alternativas (C1-C3).   | Python                      |
| 3. Pesos iniciais (literatura)                     | Bohra 2021; Song 2022; Lu 2021; Rocha-Buelvas 2025.                           | Base AHP                    |
| 4. Otimizacao de pesos (metaheuristicas geneticas) | ABC; HC/SA/PSO (COMET); PSO-SA hibrido.                                       | numpy, pandas               |
| 5. Otimizacao de pesos (vizinhanca)                | I2PLS, MTS, WILB, ILS, LBH.                                                   | numpy, pandas               |
| 6. Aplicacao MCDM                                  | Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA.                                           | pymcdm, numpy, scikit-fuzzy |
| 7. Avaliacao de melhoria                           | Spearman, Kendall, CR AHP; testes t contra baseline.                          | Python / matplotlib, scipy  |
| 8. Sensibilidade/visualizacao                      | Convergencia de rankings, pesos, perfis.                                      | Python / Streamlit          |

## Estrutura AHP

- Nivel 1: escolher a melhor estrategia operacional da microgrid.
- Nivel 2: criterios Economico, Ambiental, Tecnico (Confiabilidade), Social.
- Nivel 3: alternativas C1 Diesel-only; C2 PV + Battery; C3 Diesel + PV + Battery.

## Criterios e pesos iniciais (literatura)

| Criterio                 | Descricao                                                | Peso representativo | Referencia/Contexto                                                |
| ------------------------ | -------------------------------------------------------- | ------------------- | ------------------------------------------------------------------ |
| Economico                | Custo total, CoE, NPC                                    | 0.6923              | Bohra et al., 2021 (valor fixo para cenarios com foco economico)   |
| Ambiental                | Fracao renovavel, pegada de carbono, impacto ecossistema | 0.3328              | Rocha-Buelvas, 2025 (maior peso encontrado no estudo, FAHP SC4 A4) |
| Tecnico (Confiabilidade) | Confiabilidade, economia, tecnologia, protecao ambiental | 0.3492              | Song et al., 2022 (peso abrangente DEMATEL-AHP/CRITIC-EWM)         |
| Social                   | Aceitacao social, empregos, obstaculos locais            | 0.3351              | Rocha-Buelvas, 2025 (maior peso encontrado, AHP SC1 A5)            |

### Validacao do AHP

| Metodo | Nome completo              | Uso tipico                                         | Referencia     |
| ------ | -------------------------- | -------------------------------------------------- | -------------- |
| AHP    | Analytic Hierarchy Process | Derivacao de pesos de criterios (base de hibridos) | Ogrodnik, 2019 |

## Precos de diesel por regiao (planilha "1 - Microgrid_ReOpt_LCOE_Results_Explorer.xlsm")

| Pais / Regiao     | USD/gal | USD/L | Observacao          |
| ----------------- | ------- | ----- | ------------------- |
| Tanzania (Lodwar) | 3.20    | ~0.85 | 0.85 = 3.20 / 3.785 |
| Gana (Accra)      | 3.60    | ~0.95 | 0.95 = 3.60 / 3.785 |
| Zambia (Lusaka)   | 4.40    | ~1.16 | 1.16 = 4.40 / 3.785 |

- Conversao usada: USD/L = USD/gal / 3.785 (ex.: 3.60 / 3.785 ≈ 0.95 USD/L).
- Mapa passado aos scripts via `--diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85"`; fator de emissao: 2.67 kg CO₂/L.

## Meta-heuristicas geneticas

| Meta-heuristica             | Objetivo Especifico para Validacao                                                                                                                                                       | Referencia                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| PSO-SA Hibrido              | Refinar matrizes AHP inconsistentes, minimizando a Inconsistencia (lambda max - N) e a Discrepancia (DI).                                                                                | Sarani Rad et al., 2024     |
| HC, SA, PSO                 | Encontrar os valores otimos de preferencias de Objetos Caracteristicos (COs) para um modelo MCDM (COMET).                                                                                | Kizielewicz & Salabun, 2020 |
| ABC (Artificial Bee Colony) | Otimizar e aprender os pesos dos criterios (criteria weights), que sao usados como input para o metodo MCDM VIKOR, minimizando o custo de classificacao de inventario (Funcao Objetivo). | Cherif & Ladhari, 2016      |

Funcao objetivo exemplo: Min f(W) = alpha _ CR(W) + beta _ (1 - rho(W)), alpha = beta = 0.5; CR(W) = razao de consistencia; rho(W) = Spearman pre vs pos.

## Metaheuristicas de vizinhanca

| Meta-heuristica (Tipo de Vizinhança)                      | Objetivo Especifico para Validacao                                                                                                                                                                          | Referencia                  |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Hill-Climbing (HC), Simulated Annealing (SA)              | Usados para encontrar os valores otimos de preferencias de Objetos Caracteristicos (COs), otimizando os parametros do modelo MCDM (COMET) e minimizando o erro de preferencia absoluto.                     | Kizielewicz & Salabun, 2020 |
| Otimizacao Local Iterativa (Metodo das Decisoes Herdadas) | Solucionar problemas complexos de Otimizacao Multicriterio Nao-Linear (MCO) via varias otimizacoes locais iniciadas em pontos na vizinhanca da decisao anterior, usando funcao escalarizante.               | Lotova et al., 2019         |
| ILS (Iterative Local Search)                              | Otimizar uma funcao objetiva escalarizada que combina dois criterios com coeficiente de ponderacao (mu) em problema bi-criterio, validando busca em vizinhanca para solucoes de compromisso com pesos MCDM. | Aqil & Allali, 2021         |
| Tabu Search Multi-Vizinhanca (MTS)                        | Aplicar tabu search com quatro vizinhancas restritas para otimizar alocacao multicriterio com multiplos orcamentos (MMCP), maximizando lucro total.                                                         | Liu & Pan, 2024             |
| WILB (Weighted Iterated Local Branching)                  | Busca local iterativa guiada por restricoes de vizinhanca (local branching) para otimizacao binaria ponderada, direcionando a busca por probabilidade de flip das variaveis.                                | Rodrigues et al., 2022      |

## Parametros MCDM (referencias)

| Parametro                                                | Valor | Interpretacao                                                                                                                                                   | Referencias                              |
| -------------------------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Parametro de Defuzzificacao (Largura/Nivel de Incerteza) | 0.01  | Numero pequeno (tau) usado no denominador de termos em funcoes de Score Relativo (RSF) para defuzzificacao de pesos, representando estabilidade/margem de erro. | Otay & Kahraman, 2022                    |
| VIKOR v ou J                                             | 0.5   | Solucao de compromisso balanceada: pesa utilidade de grupo maxima (J) e pesar individual minimo (1-J).                                                          | Peng et al., 2020; Otay & Kahraman, 2022 |

## Metricas de calibracao de pesos e funcoes objetivo em MCDM (ativas)

| Abordagem / Metodo | Principio central (funcao objetivo implicita)                                                      | Metrica de calibracao / otimizacao                                       | Titulo do artigo                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Entropia           | Dar maior peso a criterios com maior dispersao; peso w_j inversamente proporcional a entropia e_j. | e_j dos valores normalizados; w_j = (1 - e_j) / sum(1 - e_j).            | Integration of objective weighting methods for criteria and MCDM methods: application in material selection |
| CRITIC             | Combinar variancia (sigma_j) e nao-redundancia (1 - r_ij) para definir importancia.                | C_j = sigma_j * (1 - sum r_ij); pesos normalizados de C_j.               | Integration of objective weighting methods for criteria and MCDM methods: application in material selection |
| Bayes / IDOCRIW    | Integrar pesos de metodos distintos como variaveis aleatorias (media geometrica ponderada).        | alpha_j = sum(omega_j * W_j) / sum(omega_j * W_j).                       | The Recalculation of the Weights of Criteria in MCDM Methods Using the Bayes Approach                       |

### Como escolher o objetivo na linha de comando
- Cada execucao usa apenas uma funcao objetivo via `--objective {entropy|critic|bayes}`.
- Para testar todas, rode o script tres vezes mudando o valor de `--objective`.

## Validacao da abordagem lexicografica

| Conceito Implementado     | Validacao na Literatura                                                                                                                            | Fonte                 |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------- |
| **Prioridade Preemptiva** | Define a _Lexicographic GP_ como a minimizacao sequencial de desvios baseada em prioridades, onde trade-offs finitos nao existem entre niveis.     | Romero (2001)         |
| **Tupla de Objetivos**    | "Lexicographic Orders": Valida matematicamente o uso de vetores ordenados como regra de decisao preferencial sobre a soma escalar.                 | Fishburn (1974)       |
| **Motivacao da Mudanca**  | Resolve a dificuldade de definir "pesos" numericos apropriados para objetivos conflitantes e nao comensuraveis.                                    | Nishad & Singh (2015) |
| **Relevancia Pratica**    | A abordagem lexicografica é a variante mais utilizada na literatura (aprox. 75% das aplicacoes) devido à sua clareza na modelagem de preferencias. | Romero (2001)         |

## Metodos MCDM

| Metodo       | Tipo           | Entrada                             | Vantagem principal     | Aplicabilidade                              | Referencia                  |
| ------------ | -------------- | ----------------------------------- | ---------------------- | ------------------------------------------- | --------------------------- |
| Fuzzy-TOPSIS | Fuzzy          | Matriz fuzzy + pesos linguisticos   | Trata incerteza        | Julgamentos subjetivos                      | Zamani-Sabzi et al., 2016   |
| VIKOR        | Deterministico | Valores normalizados + v            | Solucao de compromisso | Trade-offs custo x emissao x confiabilidade | Salabun et al., 2020        |
| COPRAS       | Deterministico | Matriz normalizada proporcional     | Estavel e simples      | Validacao e robustez                        | Radulescu & Radulescu, 2024 |
| MOORA        | Deterministico | Razao normalizada (beneficio/custo) | Simples e eficiente    | Referencia base e consistencia              | Singh & Pathak, 2024        |

## Metricas de avaliacao (implementadas)

| Metrica                        | Descricao no evaluate_mcdm_quality.py                         | Referencias                     |
| ------------------------------ | -------------------------------------------------------------- | ------------------------------- |
| Robustez de ranking            | Spearman medio entre ranks (TOPSIS/VIKOR/COPRAS/MOORA)        | McPhail 2018; Fernandez 2001    |
| Regret / Utility               | Regret por metodo: TOPSIS, VIKOR, COPRAS, MOORA               | Ning & You 2018; Groetzner 2022 |
| Dominancia / Pareto            | Checa se o vencedor TOPSIS esta na fronteira Pareto           | Saif 2014; Lagaros 2007         |
| Consistencia (proxy)           | |sum(w) - 1| como proxy de consistencia                       | Mufazzal 2021 (CR como referencia) |
| Estabilidade a ruido / cenario | Win-rate e variacao de rank com ruido 2%                      | Kim 2015; Li 2018               |
| Validacao cruzada (ruido)      | Segunda rodada de ruido 5% para robustez adicional            | Ning & You 2018                 |

## Ferramentas

| Componente             | Ferramenta                    |
| ---------------------- | ----------------------------- |
| AHP                    | ahpy                          |
| Otimizacao de pesos    | pyswarm, deap, simanneal      |
| MCDM                   | pymcdm, numpy, pandas         |
| Sensibilidade/graficos | matplotlib, plotly, streamlit |

## Scripts e exemplos (ordem do fluxo)

- (Etapa 1) `extract_load_profiles.py`: extrai e agrupa perfis de carga brutos em CSV de perfis de consumo por bloco/horario.
  ```powershell
  python extract_load_profiles.py `
    --input-dir dados\load_profiles_raw `
    --out dados_preprocessados\load_profiles_extracted.csv
  ```
- (Etapa 1) `extrair_tudo_reopt.py`: baixa/extrai resultados ReOpt (LCOE, custos, emissoes) para todos os cenarios, gerando CSV consolidado por bloco/regiao.
  ```powershell
  python extrair_tudo_reopt.py `
    --input-dir dados\reopt_raw `
    --out dados_preprocessados\reopt_ALL_blocks_v3_8.csv
  ```
- (Etapa 2/3) `build_ahp_structure.py`: agrega metricas por alternativa e constroi a estrutura AHP (baseline com pesos da literatura).
- (Etapa 5) `apply_mcdm_profiles.py`: roda MCDM por perfil baseline.
  ```powershell
  python apply_mcdm_profiles.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.01 `
    --vikor-v 0.5 `
    --out-dir ahp_mcdm_results
  ```
- (Etapa 4) `optimize_weights.py`: metaheuristicas geneticas (ABC; HC/SA/PSO; PSO-SA hibrido), 8 configs cada, multiplas execucoes.
  - Use `--objective {entropy|critic|bayes}`; cada execucao usa um objetivo. Para varrer todos, rode 3 vezes mudando o valor.
  ```powershell
  python optimize_weights.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.01 `
    --vikor-v 0.5 `
    --objective entropy `
    --runs 30 `
    --seed 2024 `
    --out-dir weight_optimization
  ```
- (Etapa 4) `optimize_weights_neighborhood.py`: metaheuristicas de vizinhanca (I2PLS, MTS, WILB, ILS, LBH), 8 configs cada, multiplas execucoes.
  - Use `--objective {entropy|critic|bayes}`; cada execucao usa um objetivo. Para varrer todos, rode 3 vezes mudando o valor.
  ```powershell
  python optimize_weights_neighborhood.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.01 `
    --vikor-v 0.5 `
    --objective entropy `
    --runs 30 `
    --seed 2024 `
    --out-dir neighborhood_results
  ```
- (Etapa 6) `evaluate_mcdm_quality.py`: avalia pesos (baseline + geneticos + vizinhanca); `--auto` varre `weight_optimization/`, `neighborhood_results/`, `summary_global.csv` e `runs_*.csv`.
  ```powershell
  python evaluate_mcdm_quality.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.01 `
    --vikor-v 0.5 `
    --auto `
    --out eval_quality_all.csv
  ```
- (Etapa 5/8) `apply_optimized_weights.py`: aplica MCDM com os melhores pesos (geneticos/vizinhanca) e gera perfis otimizados.
  ```powershell
  python apply_optimized_weights.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.01 `
    --vikor-v 0.5 `
    --out-dir optimized_mcdm_results
  ```
## Top 5 Genetic (from bayes objective runs)
| Label                       | Objective | cost    | emissions | reliability | social   |
| --------------------------- | --------- | ------- | --------- | ----------- | -------- |
| bayes_HC_SA_PSO_kiz_sa_1    | bayes     | 0.4630  | 0.2254    | 0.1684      | 0.1428   |
| bayes_ABC_abc_7             | bayes     | 0.4628  | 0.2288    | 0.1658      | 0.1426   |
| bayes_HC_SA_PSO_kiz_sa_3    | bayes     | 0.4632  | 0.2279    | 0.1701      | 0.1388   |
| bayes_ABC_abc_8             | bayes     | 0.4628  | 0.2408    | 0.1543      | 0.1421   |
| bayes_ABC_abc_3             | bayes     | 0.4628  | 0.2218    | 0.1420      | 0.1733   |

## Top 5 Neighborhood (ranked by evaluation score; with objective)
| Label          | Objective | cost    | emissions | reliability | social   |
| -------------- | --------- | ------- | --------- | ----------- | -------- |
| runs_lbh_3_22  | critic    | 0.1679  | 0.0114    | 0.8206      | 0.0000   |
| runs_mts_2_17  | entropy   | 0.1825  | 0.0100    | 0.4417      | 0.3658   |
| runs_lbh_8_15  | entropy   | 0.1811  | 0.0086    | 0.3340      | 0.4763   |
| runs_i2pls_7_13| entropy   | 0.1826  | 0.0090    | 0.3128      | 0.4956   |
| runs_lbh_8_23  | entropy   | 0.1908  | 0.0155    | 0.7936      | 0.0000   |

## Scripts (flow)
- extract_load_profiles.py — prepare load profiles.
- extrair_tudo_reopt.py — fetch/aggregate ReOpt results.
- build_ahp_structure.py — aggregate metrics and build AHP baseline.
- apply_mcdm_profiles.py — run baseline MCDM.
- optimize_weights.py — genetic metaheuristics.
- optimize_weights_neighborhood.py — neighborhood metaheuristics.
- evaluate_mcdm_quality.py — evaluate all weights (auto or explicit).
- apply_optimized_weights.py — apply best/selected weights and save MCDM outputs.

## Conclusion (results)
- Genetic (bayes objective) top weights cluster around ~0.46 cost, ~0.22 emissions, with balanced reliability/social, always yielding C2 as TOPSIS winner.
- Neighborhood bests (critic/entropy) include more extreme allocations (e.g., heavy reliability or social) but also converge on C2.
- All optimized sets agree on C2 as the preferred alternative; differences reflect how trade-offs are distributed across reliability vs. social vs. emissions.
