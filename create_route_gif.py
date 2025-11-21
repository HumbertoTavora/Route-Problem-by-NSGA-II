#!/usr/bin/env python3
"""
create_route_gif.py - Cria GIF animado mostrando as rotas dos veículos

Uso:
python create_route_gif.py \
  --instance_json data/json/Input_Data.json \
  --results_file results/Input_Data_pop400_crossProb0.85_mutProb0.05_numGen200.csv \
  --output_gif route_animation.gif
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from PIL import Image
import sys
import importlib.util


def load_instance(instance_json):
    """Carrega instância JSON"""
    with open(instance_json) as f:
        return json.load(f)


def get_functions_from_runAlgorithm():
    """Importa funções de runAlgorithm.py"""
    import sys
    spec = importlib.util.spec_from_file_location("runAlgorithm", "runAlgorithm.py")
    runAlgorithm_module = importlib.util.module_from_spec(spec)
    sys.modules["runAlgorithm"] = runAlgorithm_module
    spec.loader.exec_module(runAlgorithm_module)
    
    return {
        'getNumVehiclesRequired': runAlgorithm_module.getNumVehiclesRequired,
        'getRouteCost': runAlgorithm_module.getRouteCost,
        'routeToSubroute': runAlgorithm_module.routeToSubroute,
    }


def get_best_solution(csv_file):
    """Extrai melhor solução do CSV (última geração)"""
    df = pd.read_csv(csv_file)
    last_row = df.iloc[-1]
    
    # Extrair best_one (representação da solução)
    best_one_str = str(last_row['best_one']).strip('[]')
    best_solution = [int(x.strip().strip(',')) for x in best_one_str.split(',') if x.strip()]
    
    return best_solution


def create_route_animation_frames(instance, best_solution, funcs, 
                                 output_dir='./route_frames'):
    """Cria frames da animação das rotas"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)
    
    # Extrair funções
    routeToSubroute = funcs['routeToSubroute']
    
    # Converter solução em rotas
    rotas = routeToSubroute(best_solution, instance)
    
    print(f"📍 {len(rotas)} rotas identificadas")
    
    # Coordenadas dos clientes
    distance_matrix = np.array(instance['distance_matrix'])
    n_customers = instance['Number_of_customers']
    
    # Para visualização, usar coordenadas simuladas (baseadas em distância euclidiana)
    # Fazer MDS (Multidimensional Scaling) para converter distâncias em coordenadas 2D
    from scipy.spatial.distance import squareform
    from sklearn.manifold import MDS
    
    # Converter matriz de distância em array condensado
    distances_condensed = squareform(distance_matrix)
    
    # Aplicar MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distance_matrix)
    
    print("📍 Coordenadas 2D geradas via MDS")
    
    # Cores para cada rota
    colors = plt.cm.tab10(np.linspace(0, 1, len(rotas)))
    
    # Criar frames: um frame por cliente visitado
    frame_idx = 0
    total_customers = n_customers + len(rotas)  # +1 para cada retorno ao depósito
    
    for step in range(total_customers + 1):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plotar depósito (cliente 0)
        ax.scatter(coordinates[0, 0], coordinates[0, 1], s=500, c='red', 
                  marker='s', edgecolors='black', linewidth=2, label='Depósito', zorder=5)
        ax.text(coordinates[0, 0], coordinates[0, 1] - 2, 'Depósito', 
               ha='center', fontsize=10, fontweight='bold')
        
        # Plotar clientes
        for cid in range(1, n_customers + 1):
            ax.scatter(coordinates[cid, 0], coordinates[cid, 1], s=300, c='lightblue',
                      edgecolors='black', linewidth=1, zorder=3)
            ax.text(coordinates[cid, 0], coordinates[cid, 1], str(cid), 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Animar rotas até o passo atual
        customers_visited = 0
        
        for rota_idx, rota in enumerate(rotas):
            # rota é uma sequência [0, c1, c2, ..., cN, 0]
            
            # Determinar quantos clientes desta rota foram visitados
            customers_in_route = [c for c in rota if c != 0]
            num_customers_in_route = len(customers_in_route)
            
            if step <= customers_visited:
                # Esta rota ainda não começou
                continue
            
            # Número de clientes visitados nesta rota no passo atual
            local_step = min(step - customers_visited, num_customers_in_route + 1)
            
            # Desenhar segmentos da rota
            for i in range(local_step):
                if i < len(rota) - 1:
                    from_id = rota[i]
                    to_id = rota[i + 1]
                    
                    from_coord = coordinates[from_id]
                    to_coord = coordinates[to_id]
                    
                    # Desenhar seta
                    arrow = FancyArrowPatch(
                        from_coord, to_coord,
                        arrowstyle='->', mutation_scale=20,
                        color=colors[rota_idx], linewidth=2, alpha=0.7, zorder=2
                    )
                    ax.add_patch(arrow)
                    
                    # Desenhar linha
                    ax.plot([from_coord[0], to_coord[0]], 
                           [from_coord[1], to_coord[1]], 
                           color=colors[rota_idx], linewidth=2, alpha=0.5, zorder=1)
            
            # Destacar cliente atual (se não voltou ao depósito)
            if local_step > 0 and local_step <= num_customers_in_route:
                current_customer = customers_in_route[local_step - 1]
                ax.scatter(coordinates[current_customer, 0], coordinates[current_customer, 1],
                          s=400, c=colors[rota_idx], edgecolors='black', 
                          linewidth=2, marker='*', zorder=4, label=f'Rota {rota_idx + 1} (atual)')
            
            customers_visited += num_customers_in_route
        
        # Configurar gráfico
        ax.set_xlim(coordinates[:, 0].min() - 5, coordinates[:, 0].max() + 5)
        ax.set_ylim(coordinates[:, 1].min() - 5, coordinates[:, 1].max() + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # Título com progresso
        progress = int((step / total_customers) * 100)
        ax.set_title(f'Animação de Rotas - Progresso: {progress}%\n{len(rotas)} veículos, {n_customers} clientes', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Coordenada X', fontsize=11)
        ax.set_ylabel('Coordenada Y', fontsize=11)
        
        # Salvar frame
        frame_file = output_path / f"route_frame_{frame_idx:04d}.png"
        plt.savefig(frame_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        frame_idx += 1
        
        if frame_idx % 5 == 0:
            print(f"  ✅ {frame_idx} frames criados...")
    
    print(f"✅ Total de {frame_idx} frames criados")
    
    return output_path


def create_gif_from_route_frames(frames_dir, output_dir, fps=2,  output_gif='route_animation.gif'):
    """Cria GIF a partir dos frames de rota"""
    
    print(f"\n🎬 Criando GIF de rotas...")
    
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob('route_frame_*.png'))
    
    if not frame_files:
        print("❌ Nenhum frame encontrado!")
        return
    
    print(f"📸 Carregando {len(frame_files)} frames...")
    
    images = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        images.append(img)
    
    print(f"✅ {len(images)} frames carregados")
    print(f"💾 Salvando GIF: {output_gif}")
    
    # Criar diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)
    
    path = output_dir+"/"+output_gif

    # Criar GIF
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False
    )
    
    print(f"✅ GIF criado com sucesso: {output_gif}")
    print(f"   Dimensões: {images[0].size}")
    print(f"   Frames: {len(images)}")
    print(f"   FPS: {fps}")
    print(f"   Duração: {len(images) / fps:.1f} segundos")


def main():
    parser = argparse.ArgumentParser(
        description='Cria GIF animado das rotas dos veículos'
    )
    parser.add_argument('--instance_json', type=str, required=True,
                       help='Arquivo JSON da instância')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Arquivo CSV com resultados')
    parser.add_argument('--frames_dir', type=str, default='./visualization/frames/route_frames',
                       help='Diretório para frames temporários')
    parser.add_argument('--output_dir', type=str, default='route_animation.gif',
                       help='Nome do arquivo GIF de saída')
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames por segundo')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GERADOR DE GIF DE ROTAS - NSGA-II VRP")
    print("=" * 70 + "\n")
    
    # Carregar dados
    print("📁 Carregando instância...")
    instance = load_instance(args.instance_json)
    print(f"✅ Instância carregada: {instance['Number_of_customers']} clientes")
    
    print("\n📥 Importando funções...")
    funcs = get_functions_from_runAlgorithm()
    print("✅ Funções importadas")
    
    print("\n🔍 Extraindo melhor solução...")
    best_solution = get_best_solution(args.results_file)
    print(f"✅ Solução extraída: {best_solution[:10]}...")
    
    # Criar frames
    print("\n🎬 Criando frames...")
    frames_dir = create_route_animation_frames(instance, best_solution, funcs, args.frames_dir)
    
    # Criar GIF
    print()
    create_gif_from_route_frames(frames_dir, args.output_dir, args.fps)
    
    print("\n" + "=" * 70)
    print("✅ GIF DE ROTAS CRIADO COM SUCESSO!")
    print("=" * 70)
    print(f"\nAbra o arquivo para ver a animação das rotas:")
    print(f"  {args.output_dir}")
    print()


if __name__ == '__main__':
    main()
