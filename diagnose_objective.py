"""
Script de diagnóstico para investigar colisões na função objetivo.
Verifica:
1. Dimensionalidade de metrics_df
2. Valores reais de weight_refs calculados
3. Quantidade de colisões de objetivo quando se amostram vetores aleatórios
"""

from pathlib import Path
import numpy as np
import pandas as pd
from build_ahp_structure import (
    DATA_PATH,
    aggregate_metrics_by_alternative,
    parse_diesel_map,
)
from apply_mcdm_profiles import compute_profile_results


BASE_WEIGHTS = {"cost": 0.6923, "emissions": 0.3328, "reliability": 0.3492, "social": 0.3351}


def normalize_weights(w: np.ndarray) -> dict:
    w = np.clip(w, 0, None)
    total = w.sum()
    if total == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / total
    keys = ["cost", "emissions", "reliability", "social"]
    return dict(zip(keys, w.tolist()))


def detect_directions_simple(columns):
    directions = {}
    for col in columns:
        low = col.lower()
        if any(k in low for k in ["cost", "lcoe", "emission", "fossil", "tlcc"]):
            directions[col] = "min"
        elif "diesel_cost_share" in low:
            directions[col] = "min"
        elif "percent_load_target" in low:
            directions[col] = "max"
        else:
            directions[col] = "max"
    return directions


def compute_entropy_weights(metrics_df: pd.DataFrame) -> dict:
    data = metrics_df.to_numpy(dtype=float)
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = (col_max - col_min)
    denom[denom == 0] = 1.0
    norm = (data - col_min) / denom
    prob = norm / np.clip(norm.sum(axis=0, keepdims=True), 1e-12, None)
    prob[prob <= 0] = 1e-12
    m = data.shape[0]
    k = 1.0 / np.log(m) if m > 1 else 0.0
    entropy = -k * np.sum(prob * np.log(prob), axis=0)
    dj = 1 - entropy
    weights = dj / np.clip(dj.sum(), 1e-12, None)
    return normalize_weights(weights)


def compute_critic_weights(metrics_df: pd.DataFrame) -> dict:
    data = metrics_df.to_numpy(dtype=float)
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = (col_max - col_min)
    denom[denom == 0] = 1.0
    norm = (data - col_min) / denom
    std = np.std(norm, axis=0, ddof=0)
    corr = np.corrcoef(norm, rowvar=False)
    if np.isnan(corr).any():
        corr = np.nan_to_num(corr, nan=0.0)
    c_info = std * (1 - np.mean(corr, axis=0))
    weights = c_info / np.clip(c_info.sum(), 1e-12, None)
    return normalize_weights(weights)


def compute_merec_weights(metrics_df: pd.DataFrame) -> dict:
    data = metrics_df.to_numpy(dtype=float)
    m, n = data.shape
    if m == 0 or n == 0:
        return normalize_weights(np.ones(4))
    directions = detect_directions_simple(metrics_df.columns.tolist())
    norm = np.zeros_like(data, dtype=float)
    for j, col in enumerate(metrics_df.columns):
        col_data = data[:, j]
        if directions.get(col, "max") == "max":
            minv = np.nanmin(col_data)
            denom = np.nanmax(col_data) - minv
            denom = denom if denom != 0 else 1.0
            norm[:, j] = (col_data - minv) / denom
        else:
            maxv = np.nanmax(col_data)
            denom = maxv - np.nanmin(col_data)
            denom = denom if denom != 0 else 1.0
            norm[:, j] = (maxv - col_data) / denom
    norm = np.clip(norm, 1e-12, None)
    S = np.log(1 + np.sum(np.abs(np.log(norm)), axis=1))
    E = np.zeros(n)
    for j in range(n):
        temp = np.delete(norm, j, axis=1)
        S_prime = np.log(1 + np.sum(np.abs(np.log(temp)), axis=1))
        E[j] = np.sum(np.abs(S - S_prime))
    weights = E / np.clip(E.sum(), 1e-12, None)
    return normalize_weights(weights)


def compute_lopcow_weights(metrics_df: pd.DataFrame) -> dict:
    data = metrics_df.to_numpy(dtype=float)
    m, n = data.shape
    if m == 0 or n == 0:
        return normalize_weights(np.ones(4))
    col_min = np.nanmin(data, axis=0)
    col_max = np.nanmax(data, axis=0)
    denom = (col_max - col_min)
    denom[denom == 0] = 1.0
    norm = (data - col_min) / denom
    std = np.std(norm, axis=0, ddof=0)
    pv = std * np.log1p(denom)
    weights = pv / np.clip(pv.sum(), 1e-12, None)
    return normalize_weights(weights)


def compute_mean_weights(metrics_df: pd.DataFrame) -> dict:
    n = metrics_df.shape[1]
    if n == 0:
        return normalize_weights(np.ones(4))
    w = np.ones(n) / n
    return normalize_weights(w)


def compute_bayes_weights(base_w: dict, obj_w: dict) -> dict:
    keys = list(base_w.keys())
    base_vec = np.array([base_w[k] for k in keys], dtype=float)
    obj_vec = np.array([obj_w.get(k, 0.0) for k in keys], dtype=float)
    combined = base_vec * obj_vec
    if combined.sum() == 0:
        combined = base_vec
    combined = combined / combined.sum()
    return dict(zip(keys, combined.tolist()))


def main():
    print("=" * 80)
    print("DIAGNÓSTICO DA FUNÇÃO OBJETIVO")
    print("=" * 80)

    # 1. Carregar dados
    csv_path = Path("dados_preprocessados/reopt_ALL_blocks_v3_8.csv")
    diesel_price = 1.2
    diesel_map = {"Accra": 0.95, "Lusaka": 1.16, "Lodwar": 0.85}
    
    df_raw = pd.read_csv(csv_path)
    metrics_df = aggregate_metrics_by_alternative(
        df_raw, diesel_price_per_liter=diesel_price, diesel_price_map=diesel_map
    )
    metrics_df = metrics_df.dropna(axis=1, how="all")

    print("\n1. DIMENSIONALIDADE DE metrics_df:")
    print(f"   Shape: {metrics_df.shape}")
    print(f"   Colunas ({len(metrics_df.columns)}):")
    for i, col in enumerate(metrics_df.columns):
        print(f"      {i}: {col}")

    # 2. Calcular weight_refs
    print("\n2. WEIGHT_REFS CALCULADOS:")
    entropy_w = compute_entropy_weights(metrics_df)
    critic_w = compute_critic_weights(metrics_df)
    merec_w = compute_merec_weights(metrics_df)
    lopcow_w = compute_lopcow_weights(metrics_df)
    mean_w = compute_mean_weights(metrics_df)
    bayes_w = compute_bayes_weights(BASE_WEIGHTS, entropy_w)
    
    weight_refs = {
        "entropy": entropy_w,
        "critic": critic_w,
        "merec": merec_w,
        "lopcow": lopcow_w,
        "mean": mean_w,
        "bayes": bayes_w,
    }
    
    for name, weights in weight_refs.items():
        print(f"\n   {name}:")
        for key, val in weights.items():
            print(f"      {key:15s}: {val:.10f}")

    # 3. Verificar redundâncias
    print("\n3. VERIFICAÇÃO DE REDUNDÂNCIAS:")
    weight_vecs = {name: list(w.values()) for name, w in weight_refs.items()}
    
    duplicates = {}
    for name1, vec1 in weight_vecs.items():
        for name2, vec2 in weight_vecs.items():
            if name1 < name2:  # evitar comparações duplicadas
                diff = np.array(vec1) - np.array(vec2)
                max_diff = np.max(np.abs(diff))
                if max_diff < 1e-6:
                    print(f"   ⚠ {name1} ≈ {name2} (max diff: {max_diff:.2e})")
                    duplicates[f"{name1}-{name2}"] = max_diff

    if not duplicates:
        print("   ✓ Nenhuma redundância detectada entre weight_refs")

    # 4. Teste de colisão de objetivo
    print("\n4. TESTE DE COLISÃO DE OBJETIVO:")
    print("   Amostrando 1000 vetores aleatórios e calculando objetivo...")
    
    rng = np.random.default_rng(42)
    objectives = []
    samples = []
    
    for _ in range(1000):
        vec = normalize_weights(rng.dirichlet(np.ones(4)))
        samples.append(vec)
        # Calcular objetivo contra entropy_w (exemplo)
        obj = float(np.mean(np.abs(np.array(list(vec.values())) - np.array(list(entropy_w.values())))))
        objectives.append(obj)
    
    objectives = np.array(objectives)
    unique_objs = len(np.unique(np.round(objectives, 10)))
    collision_rate = (1000 - unique_objs) / 1000 * 100
    
    print(f"\n   Total de amostras: 1000")
    print(f"   Objetivos únicos (10 casas decimais): {unique_objs}")
    print(f"   Taxa de colisão: {collision_rate:.2f}%")
    print(f"   Min objetivo: {objectives.min():.10f}")
    print(f"   Max objetivo: {objectives.max():.10f}")
    print(f"   Média objetivo: {objectives.mean():.10f}")
    print(f"   Desvio padrão: {objectives.std():.10f}")
    
    # Distribuição de objetivos
    hist, bins = np.histogram(objectives, bins=50)
    print(f"\n   Histograma de distribuição (50 bins):")
    print(f"   Max frequência em um bin: {hist.max()}")
    print(f"   Min frequência em um bin: {hist.min()}")
    
    # 5. Análise de sensibilidade
    print("\n5. ANÁLISE DE SENSIBILIDADE (perturbações pequenas):")
    base_vec = normalize_weights(rng.dirichlet(np.ones(4)))
    print(f"\n   Vetor base: {base_vec}")
    obj_base = float(np.mean(np.abs(np.array(list(base_vec.values())) - np.array(list(entropy_w.values())))))
    print(f"   Objetivo base: {obj_base:.10f}")
    
    perturbations = []
    for _ in range(100):
        noise = rng.normal(0, 0.001, size=4)
        perturbed = base_vec.copy()
        perturbed_vec = np.array(list(perturbed.values())) + noise
        perturbed_vec = np.clip(perturbed_vec, 0, None)
        perturbed_vec = perturbed_vec / perturbed_vec.sum()
        perturbed_dict = dict(zip(["cost", "emissions", "reliability", "social"], perturbed_vec))
        obj_pert = float(np.mean(np.abs(perturbed_vec - np.array(list(entropy_w.values())))))
        perturbations.append((noise, obj_pert, obj_pert - obj_base))
    
    deltas = np.array([p[2] for p in perturbations])
    print(f"\n   Perturbações com σ=0.001 (100 amostras):")
    print(f"   Mudanças no objetivo:")
    print(f"      Min: {deltas.min():.10f}")
    print(f"      Max: {deltas.max():.10f}")
    print(f"      Média |Δ|: {np.mean(np.abs(deltas)):.10f}")
    print(f"      % onde Δ = 0 (12 casas decimais): {(np.abs(deltas) < 1e-12).sum() / 100 * 100:.1f}%")

    print("\n" + "=" * 80)
    print("FIM DO DIAGNÓSTICO")
    print("=" * 80)


if __name__ == "__main__":
    main()
