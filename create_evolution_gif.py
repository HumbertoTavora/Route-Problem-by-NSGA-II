import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from PIL import Image
import io


def parse_csv_row(row):
    """Parse uma linha do CSV para extrair valores numéricos"""
    try:
        # Parse min [num_vehicles, distance]
        min_str = str(row['min']).strip('[]').split()
        min_vals = [float(x) for x in min_str if x.strip()]
        
        # Parse max
        max_str = str(row['max']).strip('[]').split()
        max_vals = [float(x) for x in max_str if x.strip()]
        
        # Parse avg
        avg_str = str(row['avg']).strip('[]').split()
        avg_vals = [float(x) for x in avg_str if x.strip()]
        
        return {
            'generation': int(row['Generation']),
            'best_vehicles': min_vals[0] if len(min_vals) > 0 else None,
            'best_distance': min_vals[1] if len(min_vals) > 1 else None,
            'max_vehicles': max_vals[0] if len(max_vals) > 0 else None,
            'max_distance': max_vals[1] if len(max_vals) > 1 else None,
            'avg_vehicles': avg_vals[0] if len(avg_vals) > 0 else None,
            'avg_distance': avg_vals[1] if len(avg_vals) > 1 else None,
        }
    except Exception as e:
        print(f"Erro ao parsear linha: {e}")
        return None


def create_evolution_frames(csv_file, output_dir):
    """Cria frames PNG para cada geração"""
    
    # Criar diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)
    
    # Carregar CSV
    print(f"📊 Carregando dados de {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Parsear dados
    data = []
    for idx, row in df.iterrows():
        parsed = parse_csv_row(row)
        if parsed:
            data.append(parsed)
    
    print(f"✅ {len(data)} gerações carregadas")
    
    # Criar figuras para cada geração (a cada 5 gerações para GIF menor)
    step = max(1, len(data) // 50)  # Máximo 50 frames
    
    print(f"🎬 Criando {len(data)//step} frames de animação...")
    
    for idx, i in enumerate(range(0, len(data), step)):
        gen_data = data[:i+1]  # Até a geração atual
        
        # Criar figura com 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Evolução NSGA-II - Geração {gen_data[-1]["generation"]:.0f}', 
                    fontsize=14, fontweight='bold')
        
        generations = [d['generation'] for d in gen_data]
        vehicles = [d['best_vehicles'] for d in gen_data]
        distances = [d['best_distance'] for d in gen_data]
        avg_vehicles = [d['avg_vehicles'] for d in gen_data]
        avg_distances = [d['avg_distance'] for d in gen_data]
        
        # Subplot 1: Convergência de Veículos
        ax1 = axes[0]
        ax1.plot(generations, vehicles, 'o-', color='#2E86AB', linewidth=2.5, markersize=4, label='Melhor')
        ax1.plot(generations, avg_vehicles, 's--', color='#2E86AB', linewidth=1.5, alpha=0.7, label='Média')
        ax1.fill_between(generations, vehicles, avg_vehicles, alpha=0.2, color='#2E86AB')
        ax1.set_xlabel('Geração', fontsize=11)
        ax1.set_ylabel('Número de Veículos', fontsize=11)
        ax1.set_title('(A) Convergência: Veículos', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        ax1.set_ylim([min(vehicles) - 0.5, max(vehicles) + 0.5])
        
        # Subplot 2: Convergência de Distância
        ax2 = axes[1]
        ax2.plot(generations, distances, 's-', color='#A23B72', linewidth=2.5, markersize=4, label='Melhor')
        ax2.plot(generations, avg_distances, 'o--', color='#A23B72', linewidth=1.5, alpha=0.7, label='Média')
        ax2.fill_between(generations, distances, avg_distances, alpha=0.2, color='#A23B72')
        ax2.set_xlabel('Geração', fontsize=11)
        ax2.set_ylabel('Distância Total (km)', fontsize=11)
        ax2.set_title('(B) Convergência: Distância', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # Subplot 3: Scatter (Trade-off)
        ax3 = axes[2]
        colors = plt.cm.viridis(np.linspace(0, 1, len(gen_data)))
        scatter = ax3.scatter(vehicles, distances, c=range(len(gen_data)), cmap='viridis', 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.plot(vehicles, distances, '-', color='gray', alpha=0.3, linewidth=1)
        ax3.set_xlabel('Número de Veículos', fontsize=11)
        ax3.set_ylabel('Distância Total (km)', fontsize=11)
        ax3.set_title('(C) Trade-off entre Objetivos', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Geração')
        
        # Adicionar texto com estatísticas atuais
        last = gen_data[-1]
        stats_text = f"Gen: {last['generation']:.0f} | Veículos: {last['best_vehicles']:.0f} | Distância: {last['best_distance']:.1f} km"
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Salvar frame
        frame_file = output_path / f"frame_{idx:04d}.png"
        plt.savefig(frame_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 10 == 0:
            print(f"  ✅ {idx + 1} frames criados...")
    
    return output_path


def create_gif_from_frames(frames_dir, output_dir, fps=3, output_gif='evolution.gif'):
    """Cria GIF a partir dos frames PNG"""
    
    print(f"\n🎬 Criando GIF a partir dos frames...")
    
    frames_path = Path(frames_dir)
    frame_files = sorted(frames_path.glob('frame_*.png'))
    
    if not frame_files:
        print("❌ Nenhum frame encontrado!")
        return
    
    print(f"📸 Carregando {len(frame_files)} frames...")
    
    images = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        images.append(img)
    
    print(f"✅ {len(images)} frames carregados")
    
    # Criar diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)
    
    path = output_dir+"/"+output_gif

    print(f"💾 Salvando GIF: {path}")
    
    # Criar GIF
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),  # Duração de cada frame em ms
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
        description='Cria GIF animado da evolução do NSGA-II'
    )
    parser.add_argument('--results_file', type=str, required=True,
                       help='Arquivo CSV com resultados')
    parser.add_argument('--output_dir', type=str, default='./gifs/evolution.gif',
                       help='Nome do arquivo GIF de saída')
    parser.add_argument('--frames_dir', type=str, default='./visualization/frames/evolution_frames',
                       help='Diretório para armazenar frames temporários')
    parser.add_argument('--fps', type=int, default=3,
                       help='Frames por segundo no GIF')
    
    args = parser.parse_args()
    
    # Criar frames
    frames_dir = create_evolution_frames(args.results_file, args.frames_dir)
    
    # Criar GIF
    create_gif_from_frames(frames_dir, args.output_dir, args.fps)
    
    print("\n" + "=" * 70)
    print("✅ EVOLUÇÃO GIF CRIADO COM SUCESSO!")
    print("=" * 70)
    print(f"\nAbra o arquivo para ver a animação:")
    print(f"  {args.output_dir}")
    print()


if __name__ == '__main__':
    main()
