import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import argparse

def plot_synthesis_results(csv_file_path, save_plots=True, output_dir=None):
    """
    Plot synthesis results from OpenROAD CSV file.
    Each new point represents a new line entry (new Verilog design).
    
    Args:
        csv_file_path (str): Path to the synthesis results CSV file
        save_plots (bool): Whether to save plots to files
        output_dir (str): Directory to save plots (defaults to same as CSV file)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        if output_dir is None:
            output_dir = os.path.dirname(csv_file_path)
        
        # Extract base filename for plot titles and filenames
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Synthesis Results: {base_name}', fontsize=16)
        
        # Plot 1: Area over designs
        axes[0, 0].plot(range(len(df)), df['area'], 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Area vs Design Index')
        axes[0, 0].set_xlabel('Design Index')
        axes[0, 0].set_ylabel('Area')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Power over designs
        axes[0, 1].plot(range(len(df)), df['power'], 'g-o', linewidth=2, markersize=4)
        axes[0, 1].set_title('Power vs Design Index')
        axes[0, 1].set_xlabel('Design Index')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Delay over designs
        axes[1, 0].plot(range(len(df)), df['delay'], 'r-o', linewidth=2, markersize=4)
        axes[1, 0].set_title('Delay vs Design Index')
        axes[1, 0].set_xlabel('Design Index')
        axes[1, 0].set_ylabel('Delay')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Fanout over designs
        axes[1, 1].plot(range(len(df)), df['fanout'], 'm-o', linewidth=2, markersize=4)
        axes[1, 1].set_title('Fanout vs Design Index')
        axes[1, 1].set_xlabel('Design Index')
        axes[1, 1].set_ylabel('Fanout')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            output_path = os.path.join(output_dir, f'{base_name}_synthesis_plots.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Synthesis plots saved to: {output_path}")
        
        plt.show()
        
        # Create a summary statistics plot
        fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Multi-metric comparison (normalized)
        metrics = ['area', 'power', 'delay', 'fanout']
        normalized_data = {}
        
        for metric in metrics:
            if df[metric].std() != 0:  # Avoid division by zero
                normalized_data[metric] = (df[metric] - df[metric].mean()) / df[metric].std()
            else:
                normalized_data[metric] = df[metric] - df[metric].mean()
        
        x = range(len(df))
        for metric in metrics:
            ax.plot(x, normalized_data[metric], '-o', label=metric, linewidth=2, markersize=4)
        
        ax.set_title(f'Normalized Metrics Comparison: {base_name}')
        ax.set_xlabel('Design Index')
        ax.set_ylabel('Normalized Value (z-score)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_plots:
            output_path = os.path.join(output_dir, f'{base_name}_normalized_metrics.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Normalized metrics plot saved to: {output_path}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\nSynthesis Results Summary for {base_name}:")
        print("=" * 50)
        for metric in ['area', 'power', 'delay', 'fanout']:
            print(f"{metric.capitalize()}:")
            print(f"  Mean: {df[metric].mean():.4f}")
            print(f"  Std:  {df[metric].std():.4f}")
            print(f"  Min:  {df[metric].min():.4f}")
            print(f"  Max:  {df[metric].max():.4f}")
            print()
            
    except Exception as e:
        print(f"Error plotting synthesis results: {e}")

def plot_training_metrics(csv_file_path, save_plots=True, output_dir=None):
    """
    Plot training results with a new graph per episode.
    Each episode shows line plots for reward, bellman_target, and loss aggregated per batch.
    
    Args:
        csv_file_path (str): Path to the training results CSV file
        save_plots (bool): Whether to save plots to files
        output_dir (str): Directory to save plots (defaults to same as CSV file)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        if output_dir is None:
            output_dir = os.path.dirname(csv_file_path)
        
        # Extract base filename for plot titles and filenames
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Get unique episodes
        episodes = sorted(df['episode'].unique())
        
        print(f"Found {len(episodes)} episodes: {episodes}")
        
        # Create plots for each episode with batch aggregation
        for episode in episodes:
            episode_data = df[df['episode'] == episode].copy()
            
            # Aggregate data per batch (step)
            batch_data = episode_data.groupby('step').agg({
                'reward': ['mean', 'std', 'count'],
                'bellman_target': ['mean', 'std'],
                'loss': ['mean', 'std']
            }).reset_index()
            
            # Flatten column names
            batch_data.columns = ['step', 'reward_mean', 'reward_std', 'reward_count', 
                                 'bellman_target_mean', 'bellman_target_std', 
                                 'loss_mean', 'loss_std']
            
            # Create figure with subplots for this episode
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f'Training Metrics - Episode {episode} ({base_name}) - Per Batch', fontsize=16)
            
            # Get batches for x-axis
            batches = batch_data['step'].values
            
            # Plot 1: Reward over batches with error bars
            axes[0].errorbar(batches, batch_data['reward_mean'], 
                           yerr=batch_data['reward_std'], 
                           fmt='b-o', linewidth=2, markersize=4, capsize=5, label='Mean ± Std')
            axes[0].set_title(f'Mean Reward vs Batch (Episode {episode}) - {len(batches)} batches')
            axes[0].set_xlabel('Batch')
            axes[0].set_ylabel('Mean Reward')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Bellman Target over batches with error bars
            axes[1].errorbar(batches, batch_data['bellman_target_mean'], 
                           yerr=batch_data['bellman_target_std'], 
                           fmt='g-o', linewidth=2, markersize=4, capsize=5, label='Mean ± Std')
            axes[1].set_title(f'Mean Bellman Target vs Batch (Episode {episode})')
            axes[1].set_xlabel('Batch')
            axes[1].set_ylabel('Mean Bellman Target')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Loss over batches with error bars
            axes[2].errorbar(batches, batch_data['loss_mean'], 
                           yerr=batch_data['loss_std'], 
                           fmt='r-o', linewidth=2, markersize=4, capsize=5, label='Mean ± Std')
            axes[2].set_title(f'Mean Loss vs Batch (Episode {episode})')
            axes[2].set_xlabel('Batch')
            axes[2].set_ylabel('Mean Loss')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plot_filename = f"{base_name}_episode_{episode}_batch_metrics.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved training plot: {plot_path}")
            
            plt.show()
            
            # Print batch statistics for this episode
            print(f"\nEpisode {episode} Batch Statistics:")
            print(f"  Total batches: {len(batch_data)}")
            print(f"  Total samples: {batch_data['reward_count'].sum()}")
            print(f"  Avg samples per batch: {batch_data['reward_count'].mean():.1f}")
            print(f"  Reward (batch means) - Mean: {batch_data['reward_mean'].mean():.4f}, Std: {batch_data['reward_mean'].std():.4f}")
            print(f"  Bellman Target (batch means) - Mean: {batch_data['bellman_target_mean'].mean():.2f}, Std: {batch_data['bellman_target_mean'].std():.2f}")
            print(f"  Loss (batch means) - Mean: {batch_data['loss_mean'].mean():.2e}, Std: {batch_data['loss_mean'].std():.2e}")
        
        # Create summary plot across all episodes (batch-aggregated)
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f'Training Metrics Summary: {base_name} (Per Batch)', fontsize=16)
        
        # Color map for episodes
        colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))
        
        for i, episode in enumerate(episodes):
            episode_data = df[df['episode'] == episode]
            
            # Aggregate data per batch
            batch_data = episode_data.groupby('step').agg({
                'reward': 'mean',
                'bellman_target': 'mean',
                'loss': 'mean'
            }).reset_index()
            
            batches = batch_data['step'].values
            
            # Plot reward
            axes[0].plot(batches, batch_data['reward'], 
                        color=colors[i], label=f'Episode {episode}', linewidth=2, marker='o', markersize=3)
            
            # Plot bellman target
            axes[1].plot(batches, batch_data['bellman_target'], 
                        color=colors[i], label=f'Episode {episode}', linewidth=2, marker='o', markersize=3)
            
            # Plot loss
            axes[2].plot(batches, batch_data['loss'], 
                        color=colors[i], label=f'Episode {episode}', linewidth=2, marker='o', markersize=3)
        
        # Configure plots
        axes[0].set_title('Mean Reward vs Batch (All Episodes)')
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Mean Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Mean Bellman Target vs Batch (All Episodes)')
        axes[1].set_xlabel('Batch')
        axes[1].set_ylabel('Mean Bellman Target')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title('Mean Loss vs Batch (All Episodes)')
        axes[2].set_xlabel('Batch')
        axes[2].set_ylabel('Mean Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            summary_filename = f"{base_name}_batch_summary.png"
            summary_path = os.path.join(output_dir, summary_filename)
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            print(f"Saved training summary plot: {summary_path}")
        
        plt.show()
        
        # Print overall statistics
        print(f"\nTraining Summary for {base_name} (Per Batch Aggregation):")
        print("=" * 60)
        print(f"Total Episodes: {len(episodes)}")
        print(f"Total Samples: {len(df)}")
        for episode in episodes:
            episode_data = df[df['episode'] == episode]
            batch_data = episode_data.groupby('step').agg({
                'reward': ['mean', 'count'],
                'bellman_target': 'mean',
                'loss': 'mean'
            })
            print(f"\nEpisode {episode}:")
            print(f"  Batches: {len(batch_data)}")
            print(f"  Samples: {len(episode_data)}")
            print(f"  Avg Reward (per batch): {batch_data['reward']['mean'].mean():.4f}")
            print(f"  Avg Bellman Target (per batch): {batch_data['bellman_target']['mean'].mean():.2f}")
            print(f"  Avg Loss (per batch): {batch_data['loss']['mean'].mean():.2e}")
            
    except Exception as e:
        print(f"Error plotting training metrics: {e}")

def analyze_all_files(analysis_dir=None):
    """
    Analyze all CSV files in the analysis directory.
    
    Args:
        analysis_dir (str): Path to analysis directory (defaults to current script directory)
    """
    if analysis_dir is None:
        analysis_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Analyzing files in: {analysis_dir}")
    
    # Find all CSV files
    openroad_files = glob.glob(os.path.join(analysis_dir, "*openroad*.csv"))
    training_files = glob.glob(os.path.join(analysis_dir, "*training*.csv"))
    
    print(f"Found {len(openroad_files)} OpenROAD files:")
    for f in openroad_files:
        print(f"  - {os.path.basename(f)}")
    
    print(f"Found {len(training_files)} training files:")
    for f in training_files:
        print(f"  - {os.path.basename(f)}")
    
    # Process OpenROAD files
    for openroad_file in openroad_files:
        print(f"\n{'='*60}")
        print(f"Processing OpenROAD file: {os.path.basename(openroad_file)}")
        print(f"{'='*60}")
        plot_synthesis_results(openroad_file)
    
    # Process training files
    for training_file in training_files:
        print(f"\n{'='*60}")
        print(f"Processing training file: {os.path.basename(training_file)}")
        print(f"{'='*60}")
        plot_training_metrics(training_file)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze CSV files in the analysis directory.")
    parser.add_argument("--analysis_dir", type=str, default=None, help="Path to analysis directory (defaults to current script directory)")
    return parser.parse_args()

def main():
    """
    Main function to analyze CSV files in the analysis directory.
    """
    print("RTL Optimization RL Analysis Tool")
    print("=" * 40)
    
    args = parse_arguments()
    analyze_all_files(args.analysis_dir)

if __name__ == "__main__":
    main()