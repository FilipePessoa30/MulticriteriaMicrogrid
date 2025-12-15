# Multicriteria Model for Renewable Microgrid Operation

_Provisional title: **A Multi-Criteria Decision Model for Renewable Microgrid Operation Using AHP and Metaheuristic Weight Optimization: Balancing Economic, Environmental, Technical, and Social Criteria**_

## Final Methodology (AHP + weight optimization)

- AHP baseline weights (literature) for four criteria, optimized via two blocks: genetic (ABC, HC/SA/PSO, PSO-SA) and neighborhood (I2PLS, MTS, WILB, ILS, LBH).
- Runs three objective functions (entropy, critic, bayes) over C1 Diesel-only, C2 PV + Battery, C3 Diesel + PV + Battery, using Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA.
- Evaluates ranking robustness, regret per method, Pareto dominance, consistency proxy, and stability under noise (2%/5%).

## Active objective functions (3)

- entropy: higher weight to criteria with greater dispersion.
- critic: variance plus redundancy penalty (correlation).
- bayes: combines prior BASE_WEIGHTS with objective weights (literature + data).

## Model steps

| Step                                  | Description                                                                | Tools                       |
| ------------------------------------- | -------------------------------------------------------------------------- | --------------------------- |
| 1. Data collection & prep             | Costs, emissions, reliability (ReOpt LCOE Results, Load Profile Explorer). | Excel / Python              |
| 2. Hierarchical AHP                   | Goal -> Criteria (Econ, Env, Tech, Social) -> Alternatives (C1-C3).        | Python                      |
| 3. Initial weights (literature)       | Bohra 2021; Song 2022; Lu 2021; Rocha-Buelvas 2025.                        | AHP base                    |
| 4. Weight optimization (genetic)      | ABC; HC/SA/PSO (COMET); hybrid PSO-SA.                                     | numpy, pandas               |
| 5. Weight optimization (neighborhood) | I2PLS, MTS, WILB, ILS, LBH.                                                | numpy, pandas               |
| 6. MCDM application                   | Fuzzy-TOPSIS, VIKOR, COPRAS, MOORA.                                        | pymcdm, numpy, scikit-fuzzy |
| 7. Improvement evaluation             | Spearman, Kendall, AHP CR; t-tests vs baseline.                            | Python / matplotlib, scipy  |
| 8. Sensitivity/visualization          | Convergence of rankings, weights, profiles.                                | Python / Streamlit          |

## AHP structure

- Level 1: choose the best microgrid operational strategy.
- Level 2: criteria Economic, Environmental, Technical (Reliability), Social.
- Level 3: alternatives C1 Diesel-only; C2 PV + Battery; C3 Diesel + PV + Battery.

## Criteria and initial weights (literature)

| Criterion               | Description                                                | Representative weight | Reference/Context                                             |
| ----------------------- | ---------------------------------------------------------- | --------------------- | ------------------------------------------------------------- |
| Economic                | Total cost, CoE, NPC                                       | 0.6923                | Bohra et al., 2021 (fixed value for economic-focus scenarios) |
| Environmental           | Renewable fraction, carbon footprint, ecosystem impact     | 0.3328                | Rocha-Buelvas, 2025 (highest weight in study, FAHP SC4 A4)    |
| Technical (Reliability) | Reliability, economy, technology, environmental protection | 0.3492                | Song et al., 2022 (broad weight DEMATEL-AHP/CRITIC-EWM)       |
| Social                  | Social acceptance, jobs, local barriers                    | 0.3351                | Rocha-Buelvas, 2025 (highest weight, AHP SC1 A5)              |

### AHP validation

| Method | Full name                  | Typical use                                        | Reference      |
| ------ | -------------------------- | -------------------------------------------------- | -------------- |
| AHP    | Analytic Hierarchy Process | Derivation of criteria weights (basis for hybrids) | Ogrodnik, 2019 |

## Diesel prices by region (“1 - Microgrid_ReOpt_LCOE_Results_Explorer.xlsm”)

| Country/Region    | USD/gal | USD/L | Note         |
| ----------------- | ------- | ----- | ------------ |
| Tanzania (Lodwar) | 3.20    | ~0.85 | 3.20 / 3.785 |
| Ghana (Accra)     | 3.60    | ~0.95 | 3.60 / 3.785 |
| Zambia (Lusaka)   | 4.40    | ~1.16 | 4.40 / 3.785 |

- Conversion: USD/L = USD/gal / 3.785 (e.g., 3.60 / 3.785 ≈ 0.95 USD/L).
- Passed to scripts via `--diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85"`; emission factor: 2.67 kg CO2/L.

## Genetic metaheuristics

| Metaheuristic               | Specific validation objective                                                                                          | Reference                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| PSO-SA Hybrid               | Refine inconsistent AHP matrices, minimizing Inconsistency (lambda max - N) and Discrepancy (DI).                      | Sarani Rad et al., 2024     |
| HC, SA, PSO                 | Find optimal preferences of Characteristic Objects (COs) for an MCDM model (COMET).                                    | Kizielewicz & Salabun, 2020 |
| ABC (Artificial Bee Colony) | Optimize/learn criteria weights used as input to VIKOR, minimizing inventory classification cost (Objective Function). | Cherif & Ladhari, 2016      |

Example objective: Min f(W) = alpha _ CR(W) + beta _ (1 - rho(W)), alpha = beta = 0.5; CR(W) = consistency ratio; rho(W) = Spearman pre vs post.

## Neighborhood metaheuristics

| Metaheuristic (Neighborhood type)                  | Specific validation objective                                                                                                | Reference                   |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Hill-Climbing (HC), Simulated Annealing (SA)       | Optimize CO preferences (COMET), minimizing absolute preference error.                                                       | Kizielewicz & Salabun, 2020 |
| Iterative Local Optimization (Inherited Decisions) | Solve complex nonlinear MCO via multiple local searches starting from prior-neighborhood points, using scalarizing function. | Lotova et al., 2019         |
| ILS (Iterative Local Search)                       | Optimize scalarized bi-criterion objective with weight mu; neighborhood search for compromise MCDM weights.                  | Aqil & Allali, 2021         |
| Tabu Search Multi-Neighborhood (MTS)               | Tabu search with four restricted neighborhoods for multi-budget allocation, maximizing total profit.                         | Liu & Pan, 2024             |
| WILB (Weighted Iterated Local Branching)           | Iterated local branching for binary weighted optimization, guided by variable-flip probability.                              | Rodrigues et al., 2022      |

## MCDM parameters (references)

| Parameter                                     | Value | Interpretation                                                                 | References                               |
| --------------------------------------------- | ----- | ------------------------------------------------------------------------------ | ---------------------------------------- |
| Defuzzification parameter (uncertainty width) | 0.01  | Small tau in denominators of Relative Score functions (RSF); stability/margin. | Otay & Kahraman, 2022                    |
| VIKOR v or J                                  | 0.5   | Balanced compromise: group utility vs individual regret.                       | Peng et al., 2020; Otay & Kahraman, 2022 |

## Objective weighting methods in MCDM (active)

| Method          | Principle                                                                               | Calibration/optimization metric                           | Article                                                                                                     |
| --------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Entropy         | Higher weight to higher dispersion; w_j inversely proportional to entropy e_j.          | e_j of normalized values; w_j = (1 - e_j) / sum(1 - e_j). | Integration of objective weighting methods for criteria and MCDM methods: application in material selection |
| CRITIC          | Combine variance (sigma_j) and non-redundancy (1 - r_ij).                               | C_j = sigma_j \* (1 - sum r_ij); normalized.              | Same as above                                                                                               |
| Bayes / IDOCRIW | Integrate weights from different methods as random variables (weighted geometric mean). | alpha*j = sum(omega_j * W*j) / sum(omega_j * W_j).        | The Recalculation of the Weights of Criteria in MCDM Methods Using the Bayes Approach                       |

### How to choose objective on the CLI

- Each run uses one objective via `--objective {entropy|critic|bayes}`.
- To test all, run three times changing `--objective`.

## Lexicographic approach validation

| Implemented concept     | Literature validation                                                                                                   | Source                |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------------- | :-------------------- |
| **Preemptive priority** | Defines Lexicographic GP as sequential minimization of deviations by priority, with no finite trade-offs across levels. | Romero (2001)         |
| **Objective tuple**     | “Lexicographic Orders”: mathematically validates ordered vectors as decision rule over scalar sums.                     | Fishburn (1974)       |
| **Change motivation**   | Solves difficulty of defining numeric weights for conflicting, non-commensurable objectives.                            | Nishad & Singh (2015) |
| **Practical relevance** | Lexicographic approach is most used in literature (~75% of applications) due to clarity in preference modeling.         | Romero (2001)         |

## MCDM methods

| Method       | Type          | Input                             | Main advantage      | Applicability                              | Reference                   |
| ------------ | ------------- | --------------------------------- | ------------------- | ------------------------------------------ | --------------------------- |
| Fuzzy-TOPSIS | Fuzzy         | Fuzzy matrix + linguistic weights | Handles uncertainty | Subjective judgments                       | Zamani-Sabzi et al., 2016   |
| VIKOR        | Deterministic | Normalized values + v             | Compromise solution | Cost vs emission vs reliability trade-offs | Salabun et al., 2020        |
| COPRAS       | Deterministic | Proportionally normalized matrix  | Stable and simple   | Validation/robustness                      | Radulescu & Radulescu, 2024 |
| MOORA        | Deterministic | Normalized ratio (benefit/cost)   | Simple, efficient   | Reference base & consistency               | Singh & Pathak, 2024        |

## Evaluation metrics (implemented)

| Metric                        | Description                                          | References                      |
| ----------------------------- | ---------------------------------------------------- | ------------------------------- |
| Ranking robustness            | Mean Spearman across TOPSIS/VIKOR/COPRAS/MOORA ranks | McPhail 2018; Fernandez 2001    |
| Regret / Utility              | Regret per method: TOPSIS, VIKOR, COPRAS, MOORA      | Ning & You 2018; Groetzner 2022 |
| Dominance / Pareto            | Whether TOPSIS winner is on the Pareto front         | Saif 2014; Lagaros 2007         |
| Consistency (proxy)           | \|sum(w) - 1\| as a consistency proxy                | Mufazzal 2021 (CR reference)    |
| Stability to noise / scenario | Win rate & rank change with 2% noise                 | Kim 2015; Li 2018               |
| Cross-validated noise         | Second noise round at 5% for robustness              | Ning & You 2018                 |

## Tools

| Component           | Tool                          |
| ------------------- | ----------------------------- |
| AHP                 | ahpy                          |
| Weight optimization | pyswarm, deap, simanneal      |
| MCDM                | pymcdm, numpy, pandas         |
| Sensitivity/plots   | matplotlib, plotly, streamlit |

## Scripts and examples (flow)

- (Step 1) `extract_load_profiles.py`: extract and group raw load profiles to CSV by block/time.
  ```powershell
  python extract_load_profiles.py `
    --input-dir dados\load_profiles_raw `
    --out dados_preprocessados\load_profiles_extracted.csv
  ```
- (Step 1) `extrair_tudo_reopt.py`: fetch/extract ReOpt results (LCOE, costs, emissions) for all scenarios, creating consolidated CSV by block/region.
  ```powershell
  python extrair_tudo_reopt.py `
    --input-dir dados\reopt_raw `
    --out dados_preprocessados\reopt_ALL_blocks_v3_8.csv
  ```
- (Step 2/3) `build_ahp_structure.py`: aggregate metrics per alternative and build AHP structure (baseline weights).
- (Step 5) `apply_mcdm_profiles.py`: run MCDM for baseline profile.
  ```powershell
  python apply_mcdm_profiles.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.01 `
    --vikor-v 0.5 `
    --out-dir ahp_mcdm_results
  ```
- (Step 4) `optimize_weights.py`: genetic metaheuristics (ABC; HC/SA/PSO; hybrid PSO-SA), 8 configs each, multiple runs.
  - Use `--objective {entropy|critic|bayes}`; one objective per run. To sweep all, run 3 times changing the value.
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
- (Step 4) `optimize_weights_neighborhood.py`: neighborhood metaheuristics (I2PLS, MTS, WILB, ILS, LBH), 8 configs each, multiple runs.
  - Use `--objective {entropy|critic|bayes}`; one objective per run. To sweep all, run 3 times changing the value.
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
- (Step 6) `evaluate_mcdm_quality.py`: evaluate weights (baseline + genetic + neighborhood); `--auto` scans `weight_optimization/`, `neighborhood_results/`, `summary_global.csv`, and `runs_*.csv`.
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
- (Step 5/8) `apply_optimized_weights.py`: apply MCDM with best/selected weights (genetic/neighborhood) and generate optimized profiles.
  ```powershell
  python apply_optimized_weights.py `
    --csv dados_preprocessados\reopt_ALL_blocks_v3_8.csv `
    --diesel-price 1.2 `
    --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
    --fuzziness 0.01 `
    --vikor-v 0.5 `
    --auto `
    --weights optimized_mcdm_results/selected_weights_top.csv `
    --out-dir optimized_mcdm_results
  ```

## Results overview

- All optimized weight sets (genetic and neighborhood, across entropy/critic/bayes) converge on C2 as the winner in all four MCDM methods.
- Genetic (bayes objective) weights cluster around ~0.46 cost and ~0.22 emissions, with balanced reliability/social.
- Neighborhood (critic/entropy) explore more extreme trade-offs (e.g., very high reliability or high social) and also point to C2.
- Regret drops versus baseline; best neighborhood configs reach ~0.12?0.20 regret, and genetic bayes configs also beat the baseline.

## Top 5 Genetic (from bayes objective runs)

| Label                    | Objective | cost   | emissions | reliability | social |
| ------------------------ | --------- | ------ | --------- | ----------- | ------ |
| bayes_HC_SA_PSO_kiz_sa_1 | bayes     | 0.4630 | 0.2254    | 0.1684      | 0.1428 |
| bayes_ABC_abc_7          | bayes     | 0.4628 | 0.2288    | 0.1658      | 0.1426 |
| bayes_HC_SA_PSO_kiz_sa_3 | bayes     | 0.4632 | 0.2279    | 0.1701      | 0.1388 |
| bayes_ABC_abc_8          | bayes     | 0.4628 | 0.2408    | 0.1543      | 0.1421 |
| bayes_ABC_abc_3          | bayes     | 0.4628 | 0.2218    | 0.1420      | 0.1733 |

## Top 5 Neighborhood (ranked by evaluation score; with objective)

| Label           | Objective | cost   | emissions | reliability | social |
| --------------- | --------- | ------ | --------- | ----------- | ------ |
| runs_lbh_3_22   | critic    | 0.1679 | 0.0114    | 0.8206      | 0.0000 |
| runs_mts_2_17   | entropy   | 0.1825 | 0.0100    | 0.4417      | 0.3658 |
| runs_lbh_8_15   | entropy   | 0.1811 | 0.0086    | 0.3340      | 0.4763 |
| runs_i2pls_7_13 | entropy   | 0.1826 | 0.0090    | 0.3128      | 0.4956 |
| runs_lbh_8_23   | entropy   | 0.1908 | 0.0155    | 0.7936      | 0.0000 |

## Script flow

- extract_load_profiles.py — prepare load profiles.
- extrair_tudo_reopt.py — fetch/aggregate ReOpt results.
- build_ahp_structure.py — aggregate metrics and build AHP baseline.
- apply_mcdm_profiles.py — run baseline MCDM.
- optimize_weights.py — genetic metaheuristics.
- optimize_weights_neighborhood.py — neighborhood metaheuristics.
- evaluate_mcdm_quality.py — evaluate all weights (auto or explicit).
- apply_optimized_weights.py — apply best/selected weights and save MCDM outputs.

## Conclusion

- Genetic (bayes objective) top weights cluster around ~0.46 cost, ~0.22 emissions, with balanced reliability/social; all pick C2 (TOPSIS winner).
- Neighborhood bests (critic/entropy) include more extreme allocations (e.g., heavy reliability or heavy social) but also converge on C2.
- Overall, all optimized sets agree on C2 as the preferred alternative; differences reflect how trade-offs are distributed across reliability vs social vs emissions.
