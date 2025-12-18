import subprocess
import os
import pandas as pd
import time
from tqdm import tqdm

NUM_GEN_LIST = [100, 200, 300, 400, 500]
INSTANCE_NAME = "Input_Data"

tests_configs = [
    {"id": "Config 1", "pop": 100, "cross": 0.85, "mut": 0.01},
    {"id": "Config 2", "pop": 200, "cross": 0.9,  "mut": 0.1},
    {"id": "Config 3", "pop": 300, "cross": 0.9,  "mut": 0.01}
]

summary_data = []

for config in tests_configs:
    pop = config["pop"]
    cross = config["cross"]
    mut = config["mut"]
    config_id = config["id"]
    
    print(f"\n{config_id}: Pop={pop}, Cross={cross}, Mut={mut}")

    for gen in tqdm(NUM_GEN_LIST):
        cmd = [
            "python3", "runAlgorithm.py",
            "--popSize", str(pop),
            "--crossProb", str(cross),
            "--mutProb", str(mut),
            "--numGen", str(gen)
        ]
        
        start_time = time.time()
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        expected_filename = f"{INSTANCE_NAME}_pop{pop}_crossProb{cross}_mutProb{mut}_numGen{gen}.csv"
        filepath = os.path.join("results", expected_filename)
        
        if not os.path.exists(filepath):
            print(f"não encontrado: {expected_filename}")
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
                'Config_ID': config_id,
                'Population': pop,
                'Crossover': cross,
                'Mutation': mut,
                "Generations": gen,
                'Best_Vehicles': best_vehicles,
                'Best_Cost': best_cost,
                'Execution_Time_s': round(execution_time, 2)
            })
            
        except Exception as e:
            print(f"Erro ao ler {filepath}: {e}")

if summary_data:
    master_df = pd.DataFrame(summary_data)
    
    output_csv = "results/MASTER_SUMMARY_EVOLUTION_v2.csv"
    master_df.to_csv(output_csv, index=False)
    
    print(f"salvo com sucesso em: {output_csv}")

else:
    print("nenhum dado foi coletado.")
