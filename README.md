# Multi-Criteria Decision Model for Renewable Microgrid Operation

A multi-criteria decision model using AHP and metaheuristic weight optimization to balance economic, environmental, technical, and social criteria in renewable microgrid operation.

## Overview

This research applies AHP (Analytic Hierarchy Process) with metaheuristic weight optimization to evaluate three microgrid alternatives across four criteria. The model uses genetic algorithms (ABC, HC, SA, PSO) and neighborhood search methods (I2PLS, MTS, ILS, LBH, and a WILB-inspired approach) with three objective functions (entropy, CRITIC, Bayes) to optimize criteria weights for four MCDM methods: Fuzzy-TOPSIS, VIKOR, COPRAS, and MOORA.

## Project Structure

```
MulticriteriaMicrogrid/
├── src/                  # Source code
│   ├── build_ahp_structure.py          # Build AHP hierarchy
│   ├── apply_mcdm_profiles.py          # Apply MCDM methods
│   ├── optimize_weights.py             # Genetic optimization
│   ├── optimize_weights_neighborhood.py # Neighborhood optimization
│   ├── evaluate_mcdm_quality.py        # Quality evaluation
│   └── generate_*.py                   # Visualization scripts
├── data/
│   ├── raw/              # REopt simulations (Excel)
│   └── processed/        # Preprocessed CSV files
├── config/               # Configuration files
│   └── ahp_structure.json
└── results/              # Generated outputs
    ├── mcdm/             # Basic MCDM results
    ├── optimized/        # Optimized MCDM results
    ├── optimization/     # Weight optimization results
    └── figures/          # Visualizations
```

## Alternatives

- **C1**: Diesel-only microgrid
- **C2**: PV + Battery (100% renewable)
- **C3**: Hybrid (PV + Battery + Diesel)

## Decision Criteria & Literature-Based Weights

| Criterion     | Description                   | Weight | Reference           |
| ------------- | ----------------------------- | ------ | ------------------- |
| Economic      | LCOE, NPC, total costs        | 0.6923 | Bohra et al., 2021  |
| Environmental | Emissions, renewable fraction | 0.3328 | Rocha-Buelvas, 2025 |
| Technical     | Reliability, technology       | 0.3492 | Song et al., 2022   |
| Social        | Acceptance, jobs, barriers    | 0.3351 | Rocha-Buelvas, 2025 |

## Methodology

### 1. Weight Optimization Methods

**Genetic Metaheuristics:**

- ABC (Artificial Bee Colony) - Cherif & Ladhari, 2016
- HC/SA/PSO (Hill Climbing, Simulated Annealing, Particle Swarm) - Kizielewicz & Salabun, 2020
- PSO-SA Hybrid - Sarani Rad et al., 2024

**Neighborhood Search:**

- I2PLS - Lotova et al., 2019 (adapted two-phase search for continuous weight space)
- MTS, ILS - Aqil & Allali (2021), Liu & Pan (2024)
- LBH - Aqil & Allali (2021), Liu & Pan (2024) (adaptive radius for continuous space)
- WILB-inspired - Adapted local branching with variable groups (continuous space adaptation)

### 2. Objective Functions

| Method  | Principle                                                            | Reference                                  |
| ------- | -------------------------------------------------------------------- | ------------------------------------------ |
| Entropy | Higher weight to criteria with greater dispersion                    | Integration of objective weighting methods |
| CRITIC  | Combines variance and non-redundancy (correlation penalty)           | Same as above                              |
| Bayes   | Integrates prior weights with objective weights (weighted geom mean) | Recalculation of weights using Bayes       |

### 3. MCDM Methods

| Method       | Type          | Advantage           | Reference                   |
| ------------ | ------------- | ------------------- | --------------------------- |
| Fuzzy-TOPSIS | Fuzzy         | Handles uncertainty | Zamani-Sabzi et al., 2016   |
| VIKOR        | Deterministic | Compromise solution | Salabun et al., 2020        |
| COPRAS       | Deterministic | Stable and simple   | Radulescu & Radulescu, 2024 |
| MOORA        | Deterministic | Efficient           | Singh & Pathak, 2024        |

**Parameters:**

- Defuzzification: τ = 0.01 (Otay & Kahraman, 2022)
- VIKOR compromise: v = 0.5 (Peng et al., 2020)

### 4. Evaluation Metrics

- **Ranking robustness**: Spearman correlation across methods
- **Regret/Utility**: Per-method regret analysis
- **Pareto dominance**: Front analysis (cost vs emissions)
- **Consistency**: Weight sum proxy
- **Stability**: Noise sensitivity (2%, 5%)

References: McPhail 2018, Fernandez 2001, Ning & You 2018, Kim 2015

## Usage

### 1. Data Preparation

```powershell
# Extract load profiles
python data/raw/dados/extract_load_profiles.py `
  --input-dir data/raw/dados/load_profiles_raw `
  --out data/processed/dados_preprocessados/load_profiles.csv

# Extract REopt results
python data/raw/dados/extrair_tudo_reopt.py `
  --input-dir data/raw/dados/reopt_raw `
  --out data/processed/dados_preprocessados/reopt_ALL_blocks_v3_8.csv
```

### 2. Build AHP Structure

```powershell
python src/build_ahp_structure.py `
  --csv data/processed/dados_preprocessados/reopt_ALL_blocks_v3_8.csv `
  --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
  --out config/ahp_structure.json
```

**Diesel prices** (USD/L): Accra: 0.95, Lusaka: 1.16, Lodwar: 0.85 (from REopt Explorer)

### 3. Apply Baseline MCDM

```powershell
python src/apply_mcdm_profiles.py `
  --csv data/processed/dados_preprocessados/reopt_ALL_blocks_v3_8.csv `
  --diesel-price 1.2 `
  --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
  --fuzziness 0.01 `
  --vikor-v 0.5 `
  --out-dir results/mcdm
```

### 4. Optimize Weights

**Genetic algorithms:**

```powershell
python src/optimize_weights.py `
  --csv data/processed/dados_preprocessados/reopt_ALL_blocks_v3_8.csv `
  --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
  --objective {entropy|critic|bayes} `
  --runs 30 `
  --out-dir results/optimization/weight_optimization
```

**Neighborhood search:**

```powershell
python src/optimize_weights_neighborhood.py `
  --csv data/processed/dados_preprocessados/reopt_ALL_blocks_v3_8.csv `
  --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
  --objective {entropy|critic|bayes} `
  --runs 30 `
  --out-dir results/optimization/neighborhood_results
```

### 5. Evaluate Quality

```powershell
python src/evaluate_mcdm_quality.py `
  --csv data/processed/dados_preprocessados/reopt_ALL_blocks_v3_8.csv `
  --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
  --auto `
  --out config/eval_quality_all.csv
```

### 6. Apply Optimized Weights

```powershell
python src/apply_optimized_weights.py `
  --csv data/processed/dados_preprocessados/reopt_ALL_blocks_v3_8.csv `
  --diesel-map "Accra:0.95,Lusaka:1.16,Lodwar:0.85" `
  --auto `
  --out-dir results/optimized
```

### 7. Generate Visualizations

```powershell
# Ranking comparison
python src/generate_ranking_plot.py

# Pareto front (cost vs emissions)
python src/generate_pareto_front.py
```

## Results Summary

All optimized weight sets converge on **C2 (PV + Battery)** as the winner across all four MCDM methods.

### Top 5 Genetic (Bayes Objective)

| Configuration            | Cost   | Emissions | Reliability | Social |
| ------------------------ | ------ | --------- | ----------- | ------ |
| bayes_HC_SA_PSO_kiz_sa_1 | 0.4630 | 0.2254    | 0.1684      | 0.1428 |
| bayes_ABC_abc_7          | 0.4628 | 0.2288    | 0.1658      | 0.1426 |
| bayes_HC_SA_PSO_kiz_sa_3 | 0.4632 | 0.2279    | 0.1701      | 0.1388 |
| bayes_ABC_abc_8          | 0.4628 | 0.2408    | 0.1543      | 0.1421 |
| bayes_ABC_abc_3          | 0.4628 | 0.2218    | 0.1420      | 0.1733 |

**Pattern**: Genetic (Bayes) weights cluster around ~46% cost, ~22% emissions, with balanced reliability/social.

### Top 5 Neighborhood (Entropy/CRITIC)

| Configuration   | Cost   | Emissions | Reliability | Social |
| --------------- | ------ | --------- | ----------- | ------ |
| runs_lbh_3_22   | 0.1679 | 0.0114    | 0.8206      | 0.0000 |
| runs_mts_2_17   | 0.1825 | 0.0100    | 0.4417      | 0.3658 |
| runs_lbh_8_15   | 0.1811 | 0.0086    | 0.3340      | 0.4763 |
| runs_i2pls_7_13 | 0.1826 | 0.0090    | 0.3128      | 0.4956 |
| runs_lbh_8_23   | 0.1908 | 0.0155    | 0.7936      | 0.0000 |

**Pattern**: Neighborhood methods explore more extreme allocations (very high reliability or social) but also converge on C2.

### Key Findings

- **Consistency**: All optimization approaches agree on C2 as the preferred alternative
- **Trade-offs**: C2 offers zero emissions at 62% higher cost vs C3 (Hybrid)
- **Regret**: Optimized weights reduce regret vs baseline (0.12–0.20 for best configs)
- **Pareto front**: Only C2 and C3 are non-dominated; C1 is dominated by C3

## Dependencies

```
numpy, pandas, matplotlib, plotly
pymcdm, scikit-fuzzy
pyswarm, deap, simanneal
ahpy, streamlit
```

## References

### AHP & MCDM Methods

- Bohra et al. (2021) - Economic criteria weights
- Song et al. (2022) - Technical criteria (DEMATEL-AHP/CRITIC-EWM)
- Rocha-Buelvas (2025) - Environmental and social criteria (FAHP)
- Ogrodnik (2019) - AHP validation

### Optimization Methods

- Cherif & Ladhari (2016) - ABC for MCDM weight optimization
- Kizielewicz & Salabun (2020) - HC/SA/PSO for COMET
- Sarani Rad et al. (2024) - PSO-SA hybrid for AHP matrices
- Lotova et al. (2019), Aqil & Allali (2021), Liu & Pan (2024) - Neighborhood search (I2PLS, ILS, MTS)
- Rodrigues et al. (2022) - Local Branching concept (adapted for continuous weight space)

### MCDM Applications

- Zamani-Sabzi et al. (2016) - Fuzzy-TOPSIS
- Salabun et al. (2020) - VIKOR
- Radulescu & Radulescu (2024) - COPRAS
- Singh & Pathak (2024) - MOORA
- Otay & Kahraman (2022), Peng et al. (2020) - MCDM parameters

### Evaluation & Validation

- McPhail (2018), Fernandez (2001) - Ranking robustness
- Ning & You (2018), Groetzner (2022) - Regret analysis
- Saif (2014), Lagaros (2007) - Pareto dominance
- Kim (2015), Li (2018) - Stability analysis
- Mufazzal (2021) - Consistency metrics
- Fishburn (1974), Romero (2001), Nishad & Singh (2015) - Lexicographic approaches

## License

MIT License - See LICENSE file for details.

## Authors

**Filipe Pessoa**

- GitHub: [@FilipePessoa30](https://github.com/FilipePessoa30)
- Repository: [MulticriteriaMicrogrid](https://github.com/FilipePessoa30/MulticriteriaMicrogrid)
