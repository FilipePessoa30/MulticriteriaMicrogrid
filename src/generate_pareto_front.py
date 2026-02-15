import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Ler dados do ahp_structure.json
with open(PROJECT_ROOT / 'config' / 'ahp_structure.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

metrics = data['metrics_by_alternative']

# Extrair valores
# NOTA: Precisamos normalizar emissões por kWh gerado
# Vamos calcular kWh_ano assumindo Percent_Load_Target ~90% e uma carga base

# Para fronteira de Pareto, vamos usar valores agregados diretos
points = {}

for alt, vals in metrics.items():
    lcoe = vals['LCOE_$_per_kWh']
    
    # Calcular emissões por kWh
    # Total de emissões / kWh total gerado
    # Estimando: se TLCC considera toda a vida útil (20 anos?)
    # e Percent_Load_Target ~90%, precisamos estimar kWh_total
    
    # Alternativa simples: usar Fuel_cost + LCOE para estimar
    total_lcc = vals['TLCC_$']
    emissions_total = vals['Emissions_kgCO2']
    
    # Estimar kWh_total a partir do LCC e LCOE
    # LCC = LCOE * kWh_total (aproximado)
    if lcoe > 0:
        kwh_total_estimated = total_lcc / lcoe
        emissions_per_kwh = emissions_total / kwh_total_estimated if kwh_total_estimated > 0 else 0
    else:
        emissions_per_kwh = 0
    
    points[alt] = (lcoe, emissions_per_kwh)
    print(f"{alt}: LCOE={lcoe:.4f} USD/kWh, Emissions={emissions_per_kwh:.4f} kgCO2/kWh")

print("\n" + "=" * 80)
print("VALORES NORMALIZADOS PARA FRONTEIRA DE PARETO")
print("=" * 80)

for alt, (lcoe, em) in points.items():
    label = metrics[alt]
    print(f"{alt}: LCOE={lcoe:.4f} USD/kWh, Emissions={em:.4f} kgCO2/kWh")

# Verificar dominância
def is_dominated(point_a, point_b):
    """Retorna True se point_a é dominado por point_b (minimização em ambos objetivos)"""
    return (point_b[0] <= point_a[0] and point_b[1] <= point_a[1]) and \
           (point_b[0] < point_a[0] or point_b[1] < point_a[1])

print("\n" + "=" * 80)
print("ANÁLISE DE DOMINÂNCIA")
print("=" * 80)

pareto_front = []
for label, point in points.items():
    dominated = False
    for other_label, other_point in points.items():
        if label != other_label and is_dominated(point, other_point):
            dominated = True
            print(f"❌ {label} é dominado por {other_label}")
            break
    if not dominated:
        pareto_front.append((label, point))
        print(f"✅ {label} está na fronteira de Pareto")

# Ordenar fronteira por emissões
pareto_front_sorted = sorted(pareto_front, key=lambda x: x[1][1])

print(f"\n✅ Fronteira de Pareto: {len(pareto_front)} pontos")
for label, point in pareto_front_sorted:
    print(f"   {label}: ({point[0]:.4f}, {point[1]:.4f})")

# ========== PLOTAR ==========
print("\n📊 Gerando gráfico...")

fig, ax = plt.subplots(figsize=(10, 7))

# Ordenar APENAS os pontos da FRONTEIRA DE PARETO por emissões
pareto_x = [p[1][0] for p in pareto_front_sorted]
pareto_y = [p[1][1] for p in pareto_front_sorted]

# Linha conectando APENAS os pontos da fronteira de Pareto (C2 e C3)
ax.plot(pareto_x, pareto_y, '--', color='#d62728', alpha=0.8, linewidth=2.5, 
        label='Pareto Front', zorder=2)

# Plotar APENAS os pontos da fronteira de Pareto com destaque
for i, (label_pf, point_pf) in enumerate(pareto_front_sorted):
    ax.plot(point_pf[0], point_pf[1], 'o', color='#d62728', markersize=12, 
            markeredgewidth=2, markeredgecolor='darkred', zorder=3, 
            label='Non-dominated' if i == 0 else '')

# Plotar pontos dominados (C1) com estilo diferente
dominated = [label for label in points.keys() if label not in [p[0] for p in pareto_front]]
for i, label in enumerate(dominated):
    point = points[label]
    ax.plot(point[0], point[1], 'x', color='gray', markersize=12, 
            markeredgewidth=2.5, alpha=0.6, zorder=1, 
            label='Dominated' if i == 0 else '')

# Labels para todos os pontos (bem próximos)
for label, point in points.items():
    if 'C3' in label:
        ax.annotate(label, (point[0], point[1]), xytext=(5, 5), textcoords='offset points',
                   fontsize=13, ha='left', va='bottom', fontweight='bold')
    elif 'C2' in label:
        ax.annotate(label, (point[0], point[1]), xytext=(5, 5), textcoords='offset points',
                   fontsize=13, ha='left', va='bottom', fontweight='bold')
    elif 'C1' in label:
        ax.annotate(label, (point[0], point[1]), xytext=(-10, 0), textcoords='offset points',
                   fontsize=13, ha='right', va='center', fontweight='bold', color='gray')

ax.set_xlabel('LCOE (USD/kWh)', fontsize=14)
ax.set_ylabel('Emissions (kgCO2/kWh)', fontsize=14)
ax.set_title('Pareto Front: Cost vs Emissions', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)

# Legenda sem duplicatas
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=11)

plt.tight_layout()
output_path = PROJECT_ROOT / 'results' / 'figures' / 'pareto_front.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfico salvo: {output_path}")
print(f"✅ Fronteira de Pareto: {len(pareto_front)} pontos (C2, C3)")
print("   C1 é dominado por C3 (mostrado em cinza)")
