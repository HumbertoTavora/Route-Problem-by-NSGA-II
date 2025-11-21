import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from nsga_vrp.NSGA2_vrp import load_instance, routeToSubroute, eval_indvidual_fitness


# Configuração de estilo científico
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['lines.linewidth'] = 2

class VRPVisualizer:
    """Gerador de visualizações para NSGA-II VRP com dados do formato real"""
    
    def __init__(self, results_csv: str):
        """
        Inicializa visualizador e faz parse dos dados
        
        Args:
            results_csv: Caminho para arquivo CSV com resultados
        """
        self.raw_df = pd.read_csv(results_csv)
        self.results_df = self._parse_results()
        print(f"✅ Dados carregados: {len(self.results_df)} gerações")
    
    def _parse_results(self) -> pd.DataFrame:
        """
        Converte formato de string arrays em colunas numéricas
        
        Entrada: min=[  7.  529.95], max=[  9.  802.48]
        Saída: colunas separadas para each objetivo
        """
        parsed_data = []
        
        for idx, row in self.raw_df.iterrows():
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
                
                # Parse std
                std_str = str(row['std']).strip('[]').split()
                std_vals = [float(x) for x in std_str if x.strip()]
                
                if len(min_vals) >= 2 and len(max_vals) >= 2:
                    parsed_data.append({
                        'Generation': int(row['Generation']),
                        'Best_Vehicles': min_vals[0],
                        'Best_Distance': min_vals[1],
                        'Max_Vehicles': max_vals[0],
                        'Max_Distance': max_vals[1],
                        'Avg_Vehicles': avg_vals[0] if len(avg_vals) > 0 else min_vals[0],
                        'Avg_Distance': avg_vals[1] if len(avg_vals) > 1 else min_vals[1],
                        'Std_Vehicles': std_vals[0] if len(std_vals) > 0 else 0,
                        'Std_Distance': std_vals[1] if len(std_vals) > 1 else 0,
                    })
            except Exception as e:
                print(f"⚠️ Erro na linha {idx}: {e}")
                continue
        
        return pd.DataFrame(parsed_data)


    def fig_1_convergence_analysis(self) -> plt.Figure:
        """
        Figura 1: Análise de Convergência Multiobjeto
        """
        fig = plt.figure(figsize=(14, 5))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Veículos
        ax1 = fig.add_subplot(gs[0, 0])
        generations = self.results_df['Generation'].values
        vehicles_best = self.results_df['Best_Vehicles'].values
        vehicles_avg = self.results_df['Avg_Vehicles'].values
        
        ax1.fill_between(generations, vehicles_best, vehicles_avg, 
                         alpha=0.3, color='#2E86AB', label='Best-Avg')
        ax1.plot(generations, vehicles_best, 'o-', color='#2E86AB', 
                label='Melhor', linewidth=2.5, markersize=4)
        ax1.plot(generations, vehicles_avg, 's--', color='#2E86AB', 
                label='Média', linewidth=1.5, markersize=3, alpha=0.7)
        ax1.set_xlabel('Geração')
        ax1.set_ylabel('Número de Veículos')
        ax1.set_title('(A) Convergência: Veículos')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # Subplot 2: Distância
        ax2 = fig.add_subplot(gs[0, 1])
        distance_best = self.results_df['Best_Distance'].values
        distance_avg = self.results_df['Avg_Distance'].values
        
        ax2.fill_between(generations, distance_best, distance_avg, 
                         alpha=0.3, color='#A23B72', label='Best-Avg')
        ax2.plot(generations, distance_best, 's-', color='#A23B72', 
                label='Melhor', linewidth=2.5, markersize=4)
        ax2.plot(generations, distance_avg, 'o--', color='#A23B72', 
                label='Média', linewidth=1.5, markersize=3, alpha=0.7)
        ax2.set_xlabel('Geração')
        ax2.set_ylabel('Distância Total (km)')
        ax2.set_title('(B) Convergência: Distância')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # Subplot 3: Melhoria Relativa
        ax3 = fig.add_subplot(gs[0, 2])
        initial_vehicles = vehicles_best[0]
        initial_distance = distance_best[0]
        
        vehicles_improvement = ((initial_vehicles - vehicles_best) / initial_vehicles * 100)
        distance_improvement = ((initial_distance - distance_best) / initial_distance * 100)
        
        ax3.plot(generations, vehicles_improvement, 'o-', color='#2E86AB', 
                label='Veículos', linewidth=2.5, markersize=4)
        ax3.plot(generations, distance_improvement, 's-', color='#A23B72', 
                label='Distância', linewidth=2.5, markersize=4)
        ax3.set_xlabel('Geração')
        ax3.set_ylabel('Melhoria Relativa (%)')
        ax3.set_title('(C) Progresso Relativo')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        
        fig.suptitle('Análise de Convergência Multiobjeto - NSGA-II VRP', 
                    fontsize=13, fontweight='bold', y=1.00)
        
        return fig
    
    def fig_2_pareto_front_evolution(self) -> plt.Figure:
        """
        Figura 2: Evolução do Espaço de Objetivos
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Evolução do Espaço de Objetivos ao Longo das Gerações', 
                    fontsize=13, fontweight='bold', y=0.995)
        
        n_gens = len(self.results_df)
        generations_to_plot = np.linspace(0, n_gens-1, 6, dtype=int)
        
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        
        for ax_idx, (ax, gen_idx, color) in enumerate(zip(axes.flat, generations_to_plot, colors)):
            gen = self.results_df.iloc[gen_idx]
            
            # Plotar ponto de best encontrado
            best_v = gen['Best_Vehicles']
            best_d = gen['Best_Distance']
            
            # Plotar range: min para max
            max_v = gen['Max_Vehicles']
            max_d = gen['Max_Distance']
            avg_v = gen['Avg_Vehicles']
            avg_d = gen['Avg_Distance']
            
            # Plotar distribuição (min para max)
            ax.scatter([best_v], [best_d], s=200, c=[color], marker='*', 
                      label='Melhor', edgecolors='black', linewidth=2, zorder=5)
            ax.scatter([avg_v], [avg_d], s=100, c=[color], marker='o', 
                      label='Média', alpha=0.6, edgecolors='black', linewidth=1, zorder=4)
            ax.scatter([max_v], [max_d], s=50, c=[color], marker='x', 
                      label='Máximo', alpha=0.5, linewidth=2, zorder=3)
            
            # Conectar com linha para mostrar range
            ax.plot([best_v, max_v], [best_d, max_d], color=color, 
                   linestyle='--', alpha=0.3, linewidth=1.5)
            
            ax.set_xlabel('Número de Veículos')
            ax.set_ylabel('Distância Total (km)')
            ax.set_title(f'Geração {gen["Generation"]:.0f}')
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def fig_3_statistical_analysis(self) -> plt.Figure:
        """
        Figura 3: Análise Estatística de Convergência
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle('Análise Estatística: Convergência e Variabilidade', 
                    fontsize=13, fontweight='bold')
        
        generations = self.results_df['Generation'].values
        
        # Subplot 1: Veículos com banda de desvio
        ax1 = fig.add_subplot(gs[0, 0])
        best_v = self.results_df['Best_Vehicles'].values
        avg_v = self.results_df['Avg_Vehicles'].values
        std_v = self.results_df['Std_Vehicles'].values
        
        ax1.plot(generations, best_v, 'o-', color='#2E86AB', label='Melhor', linewidth=2)
        ax1.plot(generations, avg_v, 's--', color='#2E86AB', label='Média', linewidth=1.5, alpha=0.7)
        ax1.fill_between(generations, avg_v - std_v, avg_v + std_v, 
                         alpha=0.2, color='#2E86AB', label='±1 Desvio')
        ax1.set_xlabel('Geração')
        ax1.set_ylabel('Número de Veículos')
        ax1.set_title('(A) Veículos: Convergência e Variabilidade')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # Subplot 2: Distância com banda de desvio
        ax2 = fig.add_subplot(gs[0, 1])
        best_d = self.results_df['Best_Distance'].values
        avg_d = self.results_df['Avg_Distance'].values
        std_d = self.results_df['Std_Distance'].values
        
        ax2.plot(generations, best_d, 's-', color='#A23B72', label='Melhor', linewidth=2)
        ax2.plot(generations, avg_d, 'o--', color='#A23B72', label='Média', linewidth=1.5, alpha=0.7)
        ax2.fill_between(generations, avg_d - std_d, avg_d + std_d, 
                         alpha=0.2, color='#A23B72', label='±1 Desvio')
        ax2.set_xlabel('Geração')
        ax2.set_ylabel('Distância Total (km)')
        ax2.set_title('(B) Distância: Convergência e Variabilidade')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # Subplot 3: Scatter Veículos vs Distância
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(best_v, best_d, c=generations, cmap='viridis', 
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Número de Veículos')
        ax3.set_ylabel('Distância Total (km)')
        ax3.set_title('(C) Trade-off entre Objetivos')
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Geração')
        
        # Subplot 4: Estatísticas
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        final_gen = self.results_df.iloc[-1]
        
        stats_text = f"""
ESTATÍSTICAS DA ÚLTIMA GERAÇÃO ({int(final_gen['Generation'])})

Número de Veículos:
  Melhor: {final_gen['Best_Vehicles']:.0f}
  Média: {final_gen['Avg_Vehicles']:.2f}
  Máximo: {final_gen['Max_Vehicles']:.0f}
  Desvio: {final_gen['Std_Vehicles']:.2f}

Distância Total (km):
  Melhor: {final_gen['Best_Distance']:.2f}
  Média: {final_gen['Avg_Distance']:.2f}
  Máximo: {final_gen['Max_Distance']:.2f}
  Desvio: {final_gen['Std_Distance']:.2f}

Melhoria Total:
  Veículos: {((best_v[0]-best_v[-1])/best_v[0]*100):.1f}%
  Distância: {((best_d[0]-best_d[-1])/best_d[0]*100):.1f}%
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontfamily='monospace', fontsize=9.5, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        return fig
    
    def fig_4_convergence_rate(self) -> plt.Figure:
        """
        Figura 4: Taxa de Convergência e Melhoria por Geração
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Taxa de Convergência: Melhoria Incremental', 
                    fontsize=13, fontweight='bold')
        
        generations = self.results_df['Generation'].values
        best_v = self.results_df['Best_Vehicles'].values
        best_d = self.results_df['Best_Distance'].values
        
        # Subplot 1: Melhoria incremental em Veículos
        ax1 = axes[0]
        improvement_v = np.diff(best_v, prepend=best_v[0])
        improvement_v[0] = 0
        
        colors_v = ['green' if x < 0 else 'red' for x in improvement_v]
        ax1.bar(generations, improvement_v, color=colors_v, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_xlabel('Geração')
        ax1.set_ylabel('Mudança em Veículos')
        ax1.set_title('(A) Melhoria Incremental: Veículos')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Melhoria incremental em Distância
        ax2 = axes[1]
        improvement_d = np.diff(best_d, prepend=best_d[0])
        improvement_d[0] = 0
        
        colors_d = ['green' if x < 0 else 'red' for x in improvement_d]
        ax2.bar(generations, improvement_d, color=colors_d, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Geração')
        ax2.set_ylabel('Mudança em Distância (km)')
        ax2.set_title('(B) Melhoria Incremental: Distância')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


def main():
    parser = argparse.ArgumentParser(
        description='Gera visualizações avançadas para análise NSGA-II VRP'
    )
    parser.add_argument('--results_file', type=str, required=True,
                       help='Caminho para arquivo CSV de resultados')
    parser.add_argument('--output_dir', type=str, default='./figures_advanced',
                       help='Diretório de saída para figuras')
    
    args = parser.parse_args()
    
    # Criar diretório de saída
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Inicializar visualizador
    viz = VRPVisualizer(args.results_file)
    
    print("\n🎨 Gerando visualizações científicas...")

    # Figura 1: Convergência
    print("  [1/4] Convergência Multiobjeto...")
    fig1 = viz.fig_1_convergence_analysis()
    fig1.savefig(f'{args.output_dir}/01_Convergence_Analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Figura 2: Evolução Pareto
    print("  [2/4] Evolução do Espaço de Objetivos...")
    fig2 = viz.fig_2_pareto_front_evolution()
    fig2.savefig(f'{args.output_dir}/02_Pareto_Front_Evolution.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Figura 3: Análise Estatística
    print("  [3/4] Análise Estatística...")
    fig3 = viz.fig_3_statistical_analysis()
    fig3.savefig(f'{args.output_dir}/03_Statistical_Analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Figura 4: Taxa de Convergência
    print("  [4/4] Taxa de Convergência...")
    fig4 = viz.fig_4_convergence_rate()
    fig4.savefig(f'{args.output_dir}/04_Convergence_Rate.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print(f"\n✅ Visualizações salvas em: {args.output_dir}/")
    print("\nArquivos gerados:")
    print("  - 01_Convergence_Analysis.png")
    print("  - 02_Pareto_Front_Evolution.png")
    print("  - 03_Statistical_Analysis.png")
    print("  - 04_Convergence_Rate.png")


if __name__ == '__main__':
    main()