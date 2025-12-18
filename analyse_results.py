import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_theme(style="whitegrid")

df = pd.read_csv("results/MASTER_SUMMARY.csv")

if not os.path.exists("visualization/analysis"):
    os.makedirs("visualization/analysis")

print("Gerando gráficos...")

# GRÁFICO 1: Pareto dos Parâmetros (Custo vs Veículos)
# qual combinação chegou mais perto da origem (0,0)
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    data=df, 
    x="Best_Cost", 
    y="Best_Vehicles", 
    hue="Mutation",
    size="Population",
    style="Crossover",
    sizes=(50, 200),
    palette="viridis"
)
plt.title("Comparação de Desempenho por Parâmetros")
plt.xlabel("Distância Total (Custo)")
plt.ylabel("Número de Veículos")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("visualization/analysis/1_scatter_performance.png")
plt.close()

# GRÁFICO 2: Impacto da Mutação e População no Custo
g = sns.catplot(
    data=df, kind="bar",
    x="Population", y="Best_Cost", hue="Mutation", col="Crossover",
    alpha=.8, height=7, aspect=1.2, palette="magma"
)
g.despine(left=True)
g.set_axis_labels("Tamanho da População", "Melhor Custo Encontrado")
g.fig.suptitle("Impacto dos Parâmetros no Custo Final", y=1.02)
plt.savefig("visualization/analysis/2_barplot_impact.png")
plt.close()

# GRÁFICO 3: Mapa de Calor (Heatmap) - Qual a melhor combinação
pivot_table = df.pivot_table(
    values='Best_Cost', 
    index='Mutation', 
    columns=['Population', 'Crossover']
)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu_r")
plt.title("Heatmap de Custo Final")
plt.tight_layout()
plt.savefig("visualization/analysis/3_heatmap_optimization.png")
plt.close()

print("Gráficos gerados em visualization/analysis/")
