import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 7)

INPUT_FILE = "results/MASTER_SUMMARY_EVOLUTION_v2.csv"
OUTPUT_DIR = "visualization/analysis_evolution"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"dados de: {INPUT_FILE}")

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print("Arquivo CSV não encontrado")
    exit()

# GRÁFICO 1: Evolução do CUSTO (Distância)
plt.figure()
ax = sns.lineplot(
    data=df, 
    x="Generations", 
    y="Best_Cost", 
    hue="Config_ID", 
    style="Config_ID", 
    markers=True, 
    dashes=False,
    linewidth=2.5,
    palette="viridis"
)

plt.title("Evolução do Custo (Distância) por Geração", fontsize=18, fontweight='bold')
plt.xlabel("Número de Gerações")
plt.ylabel("Melhor Custo Encontrado")
plt.legend(title="Configuração", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "1_evolucao_custo.png")
plt.savefig(output_path, dpi=300)
print(f"gráfico salvo {output_path}")
plt.close()

# GRÁFICO 2: Evolução do NÚMERO DE VEÍCULOS
plt.figure()

ax = sns.lineplot(
    data=df, 
    x="Generations", 
    y="Best_Vehicles", 
    hue="Config_ID", 
    style="Config_ID",
    markers=True, 
    dashes=False,
    linewidth=2.5,
    palette="magma"
)

ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.title("Redução da Frota (Veículos) por Geração", fontsize=18, fontweight='bold')
plt.xlabel("Número de Gerações")
plt.ylabel("Número de Veículos")
plt.legend(title="Configuração", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "2_evolucao_veiculos.png")
plt.savefig(output_path, dpi=300)
print(f"gráfico salvo: {output_path}")
plt.close()

# GRÁFICO 3: Visão Geral (FacetGrid)
df_melted = df.melt(
    id_vars=["Generations", "Config_ID"], 
    value_vars=["Best_Cost", "Best_Vehicles"],
    var_name="Metrica", 
    value_name="Valor"
)

df_melted["Metrica"] = df_melted["Metrica"].replace({
    "Best_Cost": "Custo (Distância)",
    "Best_Vehicles": "Qtd. Veículos"
})

g = sns.relplot(
    data=df_melted,
    x="Generations", y="Valor",
    hue="Config_ID", style="Config_ID",
    col="Metrica", 
    kind="line", 
    markers=True, dashes=False,
    facet_kws={'sharey': False, 'sharex': True},
    height=6, aspect=1.1,
    palette="deep"
)

g.fig.suptitle("Análise Comparativa: Custo vs Frota", y=1.05, fontsize=20, fontweight='bold')
g.set_axis_labels("Gerações", "Valor da Métrica")
g._legend.set_title("Configuração")

output_path = os.path.join(OUTPUT_DIR, "3_visao_comparativa.png")
plt.savefig(output_path, dpi=300)
print(f"gráfico salvo: {output_path}")
plt.close()

print("gráficos foram gerados na pasta: visualization/analysis_evolution")

###### TIME ANALYSIS ######
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 7)

OUTPUT_DIR = "visualization/analysis_time"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"dados de: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print("CSV não encontrado.")
    exit()

# GRÁFICO 1: Curva de Eficiência (Trade-off Custo x Tempo)
plt.figure()

sns.lineplot(
    data=df,
    x="Execution_Time_s",
    y="Best_Cost",
    hue="Config_ID",
    style="Config_ID",
    markers=True,
    dashes=False,
    sort=False,
    linewidth=2.5,
    palette="viridis"
)

final_points = df.sort_values("Generations").groupby("Config_ID").tail(1)

for _, row in final_points.iterrows():
    plt.text(
        row['Execution_Time_s'], 
        row['Best_Cost'], 
        f" {int(row['Generations'])}gen", 
        fontsize=10, 
        va='bottom'
    )

plt.title("Curva de Eficiência: Qualidade da Solução vs. Tempo Gasto", fontsize=18, fontweight='bold')
plt.xlabel("Tempo de Execução (segundos)")
plt.ylabel("Melhor Custo (Menor é Melhor)")
plt.legend(title="Configuração")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "1_eficiencia_tempo_custo.png")
plt.savefig(output_path, dpi=300)
print(f"gráfico salvo: {output_path}")
plt.close()

# GRÁFICO 2: Escalabilidade (Tempo por Geração)
plt.figure()

sns.lineplot(
    data=df,
    x="Generations",
    y="Execution_Time_s",
    hue="Config_ID",
    style="Config_ID",
    markers=True,
    dashes=False,
    linewidth=2.5,
    palette="rocket"
)

plt.title("Escalabilidade Computacional: Custo de Tempo por Configuração", fontsize=18, fontweight='bold')
plt.xlabel("Número de Gerações")
plt.ylabel("Tempo Total Acumulado (s)")
plt.legend(title="Configuração")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "2_escalabilidade.png")
plt.savefig(output_path, dpi=300)
print(f"gráfico salvo: {output_path}")
plt.close()

print(f"análise de tempo concluída em: {OUTPUT_DIR}")
