import subprocess
import os
import pandas as pd
import itertools
from tqdm import tqdm

POP_SIZES = [100, 300, 500]
CROSS_PROBS = [0.7, 0.85, 0.9]
MUT_PROBS = [0.01, 0.05, 0.1]
NUM_GEN = 200
INSTANCE_NAME = "Input_Data" 

summary_data = []
combinations = list(itertools.product(POP_SIZES, CROSS_PROBS, MUT_PROBS))

print(f"iniciando Grid Search com {len(combinations)} combinações...")

for pop, cross, mut in tqdm(combinations):
    cmd = [
        "python3", "runAlgorithm.py",
        "--popSize", str(pop),
        "--crossProb", str(cross),
        "--mutProb", str(mut),
        "--numGen", str(NUM_GEN)
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    expected_filename = f"{INSTANCE_NAME}_pop{pop}_crossProb{cross}_mutProb{mut}_numGen{NUM_GEN}.csv"
    filepath = os.path.join("results", expected_filename)
    
    if not os.path.exists(filepath):
        print(f"\n⚠️ Arquivo não encontrado: {expected_filename}")
        print("Verifique se o 'INSTANCE_NAME' na linha 13 deste script bate com o seus arquivos.")
        continue

    try:
        df = pd.read_csv(filepath)
        if df.empty: continue

        last_gen = df.iloc[-1]
        
        val_str = str(last_gen['min']).replace('[', '').replace(']', '')
        parts = val_str.replace(',', ' ').split()
        
        best_vehicles = float(parts[0])
        best_cost = float(parts[1])
        
        summary_data.append({
            'Population': pop,
            'Crossover': cross,
            'Mutation': mut,
            'Best_Vehicles': best_vehicles,
            'Best_Cost': best_cost
        })
        
    except Exception as e:
        print(f"Erro ao ler {filepath}: {e}")

if summary_data:
    master_df = pd.DataFrame(summary_data)
    master_df.to_csv("results/MASTER_SUMMARY.csv", index=False)
    print("resumo salvo com sucesso!")
else:
    print("Nenhum dado foi coletado.")
