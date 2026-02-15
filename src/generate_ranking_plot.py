#!/usr/bin/env python3
"""
Gera ESPECIFICAMENTE o gráfico de Ranking Comparison usando dados reais
FOCO: Replicar exatamente o layout da imagem fornecida
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# Configuração de estilo para ficar idêntico ao original
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def generate_ranking_comparison():
    """Gera Ranking Comparison EXATAMENTE como a imagem usando dados reais."""
    
    print("📊 Coletando dados reais dos CSVs...")
    
    # Ler os dados reais dos CSVs
    result_files = list(Path("optimized_mcdm_results").glob("mcdm_*_weights.csv"))
    
    if not result_files:
        print("❌ Nenhum arquivo de resultado encontrado!")
        return
    
    methods = ['TOPSIS', 'VIKOR', 'COPRAS', 'MOORA']
    all_c1_scores = {method: [] for method in methods}
    all_c2_scores = {method: [] for method in methods}
    all_c3_scores = {method: [] for method in methods}
    
    # Coletar scores reais de múltiplos arquivos
    files_processed = 0
    for file in result_files:
        try:
            df = pd.read_csv(file)
            if len(df) >= 3:  # Garantir C1, C2, C3
                # Extrair scores (não ranks) para cada método
                c1_row = df[df['Alternative'] == 'C1'].iloc[0]
                c2_row = df[df['Alternative'] == 'C2'].iloc[0] 
                c3_row = df[df['Alternative'] == 'C3'].iloc[0]
                
                # TOPSIS: score direto (maior é melhor)
                all_c1_scores['TOPSIS'].append(c1_row['fuzzy_topsis_score'])
                all_c2_scores['TOPSIS'].append(c2_row['fuzzy_topsis_score'])
                all_c3_scores['TOPSIS'].append(c3_row['fuzzy_topsis_score'])
                
                # VIKOR: inverter score (menor é melhor no VIKOR)
                all_c1_scores['VIKOR'].append(1.0 - c1_row['vikor_score'])
                all_c2_scores['VIKOR'].append(1.0 - c2_row['vikor_score'])
                all_c3_scores['VIKOR'].append(1.0 - c3_row['vikor_score'])
                
                # COPRAS: score direto (maior é melhor)
                all_c1_scores['COPRAS'].append(c1_row['copras_score'])
                all_c2_scores['COPRAS'].append(c2_row['copras_score'])
                all_c3_scores['COPRAS'].append(c3_row['copras_score'])
                
                # MOORA: score direto (maior é melhor) 
                all_c1_scores['MOORA'].append(c1_row['moora_score'])
                all_c2_scores['MOORA'].append(c2_row['moora_score'])
                all_c3_scores['MOORA'].append(c3_row['moora_score'])
                
                files_processed += 1
                
        except Exception as e:
            print(f"⚠️  Erro processando {file}: {e}")
            continue
    
    print(f"✅ Processados {files_processed} arquivos de resultado")
    
    # Calcular médias reais
    c1_scores = [np.mean(all_c1_scores[method]) for method in methods]
    c2_scores = [np.mean(all_c2_scores[method]) for method in methods]  
    c3_scores = [np.mean(all_c3_scores[method]) for method in methods]
    
    print(f"📈 Scores médios:")
    for i, method in enumerate(methods):
        print(f"   {method}: C1={c1_scores[i]:.3f}, C2={c2_scores[i]:.3f}, C3={c3_scores[i]:.3f}")
    
    # Normalizar scores para [0,1] para visualização consistente
    all_scores = c1_scores + c2_scores + c3_scores
    min_score, max_score = min(all_scores), max(all_scores)
    
    def normalize(scores):
        if max_score == min_score:
            return [0.5] * len(scores)  # Fallback se todos iguais
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    c1_scores_norm = normalize(c1_scores)
    c2_scores_norm = normalize(c2_scores) 
    c3_scores_norm = normalize(c3_scores)
    
    print(f"📊 Scores normalizados:")
    for i, method in enumerate(methods):
        print(f"   {method}: C1={c1_scores_norm[i]:.3f}, C2={c2_scores_norm[i]:.3f}, C3={c3_scores_norm[i]:.3f}")
    
    # ========== PLOT EXATAMENTE COMO A IMAGEM ==========
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(methods))
    width = 0.25
    
    # Cores EXATAS da imagem
    color_c1 = '#FFD700'  # Amarelo (C1 - Diesel)
    color_c2 = '#FF6B00'  # Laranja (C2 - PV + Battery)
    color_c3 = '#CD5C5C'  # Vermelho (C3 - Hybrid)
    
    # Barras agrupadas
    bars1 = ax.bar(x - width, c1_scores_norm, width, label='C1 - Diesel', 
                   color=color_c1, alpha=0.9, edgecolor='none')
    bars2 = ax.bar(x, c2_scores_norm, width, label='C2 - PV + Battery', 
                   color=color_c2, alpha=0.9, edgecolor='none')  
    bars3 = ax.bar(x + width, c3_scores_norm, width, label='C3 - Hybrid', 
                   color=color_c3, alpha=0.9, edgecolor='none')
    
    # Labels e formatação EXATOS da imagem  
    ax.set_ylabel('Normalized Score', fontsize=14, fontweight='normal')
    ax.set_title('Ranking Comparison by MCDM Method', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    
    # Limites exatos
    ax.set_ylim(0, 1.0)
    
    # Grid horizontal igual à imagem
    ax.grid(True, axis='y', alpha=0.4, linestyle='-', color='lightgray', linewidth=0.8)
    ax.set_axisbelow(True)  # Grid atrás das barras
    
    # LEGENDA À DIREITA (fora do plot) - EXATO como na imagem
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, frameon=True)
    
    # Formatação dos ticks
    ax.tick_params(labelsize=12)
    ax.tick_params(axis='x', which='major', pad=8)
    ax.tick_params(axis='y', which='major', pad=5)
    
    # Remover spines superiores e direitas para ficar clean
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Ajustar layout para acomodar legenda externa
    plt.tight_layout()
    
    # Salvar com alta qualidade
    plt.savefig('ranking_comparison_final.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ Gerado: ranking_comparison_final.png")
    print("🎯 Gráfico criado com dados reais no layout exato da imagem!")

if __name__ == "__main__":
    print("🚀 Gerando Ranking Comparison com dados reais...")
    generate_ranking_comparison()
    print("✅ Concluído!")