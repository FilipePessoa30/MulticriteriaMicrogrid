# Modelo multicriterio para operacao de microgrids renovaveis
_Titulo provisorio: **A Multi-Criteria Decision Model for Renewable Microgrid Operation Using AHP and Metaheuristic Weight Optimization: Balancing Cost, Emissions, and Reliability**_

## Metodologia final (AHP + otimizacao de pesos)
- Integra AHP (pesos derivados da literatura) com otimizacao via PSO, GA e SA.
- Aplica quatro metodos MCDM (Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA) para comparar alternativas Diesel-only (C1), PV + Battery (C2) e Diesel + PV + Battery (C3).
- Avalia robustez, consistencia e melhoria de desempenho nos rankings.

### Etapas do modelo
| Etapa | Descricao | Ferramentas |
| --- | --- | --- |
| 1. Coleta e preparacao dos dados | Extracao de custos, emissoes e confiabilidade das planilhas ReOpt LCOE Results e Load Profile Explorer. | Excel / Python |
| 2. Estruturacao hierarquica (AHP) | Objetivo -> Criterios (Economico, Ambiental, Tecnico, Social) -> Alternativas (C1-C3). | Python |
| 3. Atribuicao inicial de pesos (literatura) | Pesos validados por Bohra (2021), Song (2022), Lu (2021), Rocha-Buelvas (2025). | Base AHP |
| 4. Otimizacao dos pesos via metaheuristicas | PSO, GA e SA para ajustar pesos, minimizando inconsistencia e maximizando estabilidade de ranking. | Python (pymoo, deap, pyswarm) |
| 5. Normalizacao e aplicacao dos metodos MCDM | Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA com pesos originais e otimizados. | Python (pymcdm, numpy, scikit-fuzzy) |
| 6. Avaliacao de melhoria | Comparacao de rankings e consistencia antes/depois (Spearman, Kendall tau, CR AHP). | Python / Matplotlib |
| 7. Analise de sensibilidade e visualizacao | Convergencia entre rankings, pesos e perfis decisorios. | Python / Streamlit |

### Estrutura hierarquica AHP
- Nivel 1 (objetivo): Escolher a melhor estrategia operacional da microgrid.
- Nivel 2 (criterios): Economico, Ambiental, Tecnico (Confiabilidade), Social.
- Nivel 3 (alternativas): C1 Diesel-only; C2 PV + Battery; C3 Diesel + PV + Battery.
- Pesos iniciais: derivados da literatura e ajustados por metaheuristicas para perfis (economico, sustentavel, resiliente, social).

### Criterios e pesos iniciais (literatura)
| Criterio | Descricao | Faixa de pesos na literatura | Peso inicial adotado | Referencia |
| --- | --- | --- | --- | --- |
| Economico | Custo total e LCOE | 0.3-0.6 | 0.40 | Bohra et al., 2021 |
| Ambiental | Emissoes totais e fracao renovavel | 0.2-0.5 | 0.30 | Rocha-Buelvas, 2025 |
| Tecnico (Confiabilidade) | Energia suprida, eficiencia da bateria | 0.2-0.4 | 0.20 | Song et al., 2022 |
| Social | Acessibilidade e impacto local | 0.1-0.2 | 0.10 | Lu et al., 2021 |

### Otimizacao dos pesos (PSO, GA, SA)
| Algoritmo | Estrategia | Funcao objetivo | Referencia |
| --- | --- | --- | --- |
| PSO (Particle Swarm Optimization) | Busca coletiva de particulas | Minimizar CR e maximizar estabilidade de ranking | Yu et al., 2020 |
| GA (Genetic Algorithm) | Selecao e cruzamento | Minimizar diferenca entre ranking AHP e MCDM + CR | Nemova et al., 2024 |
| SA (Simulated Annealing) | Busca local com resfriamento | Minimizar inconsistencia residual e desvios entre perfis | Rad et al., 2024 |

Funcao objetivo (exemplo):
- Min f(W) = alpha * CR(W) + beta * (1 - rho(W)), com alpha = beta = 0.5
- CR(W): razao de consistencia do AHP; rho(W): correlacao de Spearman entre rankings pre e pos otimizacao.

### Metodos multicriterio aplicados
| Metodo | Tipo | Funcao | Vantagem |
| --- | --- | --- | --- |
| Fuzzy-TOPSIS | Fuzzy | Similaridade com solucao ideal | Captura incerteza nas preferencias |
| VIKOR | Deterministico | Solucao de compromisso | Equilibrio entre criterios conflitantes |
| COPRAS | Deterministico | Avaliacao proporcional direta | Estabilidade e simplicidade |
| MOORA | Deterministico | Razao normalizada | Leve e valida cruzada |

### Avaliacao de melhoria
| Metrica | Interpretacao |
| --- | --- |
| Delta CR | Reducao na razao de inconsistencia (AHP) |
| Delta rho (Spearman) | Aumento da correlacao entre rankings (robustez) |
| Delta J (Jaccard) | Similaridade entre rankings antes/depois |
| Delta Score (MCDM) | Melhoria media na pontuacao das alternativas |

### Ferramentas e implementacao
| Componente | Ferramenta |
| --- | --- |
| AHP/Fuzzy-AHP | ahpy, scikit-fuzzy |
| Otimizacao de pesos | pyswarm, deap, simanneal |
| MCDM (TOPSIS, VIKOR, COPRAS, MOORA) | pymcdm, numpy, pandas |
| Analise de sensibilidade e graficos | matplotlib, plotly, streamlit |

### Scripts
- `build_ahp_structure.py`: monta a hierarquia AHP e agrega metricas por alternativa a partir do CSV consolidado.
- `apply_mcdm_profiles.py`: aplica Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA para cada perfil (economico, sustentavel, resiliente, social) e gera CSV/JSON em `ahp_mcdm_results/`.
  - Exemplo:
    ```powershell
    python apply_mcdm_profiles.py `
      --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
      --diesel-price 1.2 `
      --fuzziness 0.05 `
      --vikor-v 0.5 `
      --out-dir ahp_mcdm_results
    ```

### Conclusao
- Parte de pesos validados (AHP), otimiza via PSO/GA/SA e compara em quatro metodos MCDM.
- Garante consistencia, robustez e transparencia ao escolher entre Diesel-only, PV + Battery e Hibrido.
- Combina fundamentacao teorica e otimizacao computacional para decisao multicriterio em microgrids renovaveis.
