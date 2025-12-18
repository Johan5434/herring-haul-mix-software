#!/usr/bin/env python3
"""
visualize_results.py

Create horizontal stacked bar charts showing predicted vs true proportions
for simulated hauls. Each haul gets two bars - one for true proportions,
one for predicted proportions, side-by-side for easy comparison.

Usage:
    python visualize_results.py results/predictions_*.csv
    python visualize_results.py results/predictions_0001-0040_*.csv
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_predictions(csv_file):
    """Load predictions from CSV file."""
    df = pd.read_csv(csv_file)
    return df


def create_stacked_bar_chart(df, output_path=None, fig_size=(14, 20)):
    """
    Create a horizontal stacked bar chart showing true vs predicted proportions.
    
    Args:
        df: DataFrame with columns for true_* and pred_* percentages
        output_path: Where to save the figure (default: inferred from CSV)
        fig_size: Figure size (width, height)
    """
    
    # Population colors
    colors = {
        'Autumn': '#E8B4A8',      # Soft peachy-orange
        'North': '#9DB4C4',       # Soft slate blue
        'Central': '#A8C9A8',     # Soft sage green
        'South': '#D4A5D4',       # Soft lavender-mauve
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Calculate positions for bars
    n_hauls = len(df)
    bar_height = 0.35  # Height of each bar
    y_positions = np.arange(n_hauls) * (2 * bar_height + 0.3)  # Space between haul pairs
    
    # Prepare data
    populations = ['Autumn', 'North', 'Central', 'South']
    
    # Plot for each haul
    for i, (idx, row) in enumerate(df.iterrows()):
        haul_id = row['haul_id']
        
        # True proportions
        true_vals = [row[f'true_{pop}'] for pop in populations]
        pred_vals = [row[f'pred_{pop}'] for pop in populations]
        
        # Cumulative values for stacking
        true_cumsum = np.insert(np.cumsum(true_vals)[:-1], 0, 0)
        pred_cumsum = np.insert(np.cumsum(pred_vals)[:-1], 0, 0)
        
        y_true = y_positions[i] + bar_height / 2
        y_pred = y_positions[i] - bar_height / 2
        
        # Plot true proportions (top bar)
        for j, pop in enumerate(populations):
            ax.barh(y_true, true_vals[j], left=true_cumsum[j], height=bar_height,
                   label=pop if i == 0 else "", color=colors[pop], edgecolor='black', linewidth=0.5)
        
        # Plot predicted proportions (bottom bar)
        for j, pop in enumerate(populations):
            ax.barh(y_pred, pred_vals[j], left=pred_cumsum[j], height=bar_height,
                   label=pop if i == 0 else "", color=colors[pop], alpha=0.6, 
                   edgecolor='black', linewidth=0.5)
        
        # Add haul label
        ax.text(-2, y_positions[i], haul_id, ha='right', va='center', fontsize=8, fontweight='bold')
        
        # Add separator line
        if i < n_hauls - 1:
            ax.axhline(y=y_positions[i] + bar_height + 0.15, color='lightgray', linestyle='--', linewidth=0.5)
    
    # Formatting
    ax.set_xlim(-1, 100)
    ax.set_ylim(-1, y_positions[-1] + 1)
    ax.set_xlabel('Proportion (%)', fontsize=12, fontweight='bold')
    ax.set_title('Haul Classification Results: True vs Predicted Proportions', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Custom legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[pop], edgecolor='black', label=f'{pop} (True)')
                      for pop in populations]
    legend_elements += [plt.Rectangle((0,0),1,1, facecolor=colors[pop], alpha=0.6, edgecolor='black', label=f'{pop} (Pred)')
                       for pop in populations]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
    
    # Remove y-axis ticks
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    
    # Add annotation for bar meanings
    fig.text(0.12, 0.02, 'Top bar = True proportions | Bottom bar = Predicted proportions', 
            ha='left', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path is None:
        # Infer from the CSV file name
        output_path = output_path or "stacked_bar_chart.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved chart to {output_path}")
    
    plt.close()


def create_summary_statistics(df):
    """Create and print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Calculate errors
    df['mae_total'] = (abs(df['error_Autumn']) + abs(df['error_North']) + 
                       abs(df['error_Central']) + abs(df['error_South'])) / 4
    
    print(f"\nTotal hauls analyzed: {len(df)}")
    print(f"\nMean Absolute Error (MAE):")
    print(f"  Autumn:  {df['error_Autumn'].abs().mean():.2f}%")
    print(f"  North:   {df['error_North'].abs().mean():.2f}%")
    print(f"  Central: {df['error_Central'].abs().mean():.2f}%")
    print(f"  South:   {df['error_South'].abs().mean():.2f}%")
    print(f"  Overall: {df['mae_total'].mean():.2f}%")
    
    print(f"\nPerfect predictions (0% error): {(df['mae_total'] == 0).sum()} hauls")
    print(f"Excellent predictions (≤4% MAE): {(df['mae_total'] <= 4).sum()} hauls")
    print(f"Good predictions (≤10% MAE): {(df['mae_total'] <= 10).sum()} hauls")
    
    print("\n" + "="*70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <predictions_csv_file>")
        print("       python visualize_results.py results/predictions_*.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    print(f"Loading predictions from {csv_file}...")
    df = load_predictions(csv_file)
    
    # Print summary statistics
    create_summary_statistics(df)
    
    # Generate output filename
    base_name = Path(csv_file).stem  # e.g., "predictions_0001-0040_20251211_144143"
    output_path = os.path.join(os.path.dirname(csv_file), f"{base_name}_visualization.png")
    
    # Determine figure size based on number of hauls
    n_hauls = len(df)
    fig_height = max(10, n_hauls * 0.3)  # Scale height with number of hauls
    
    print(f"\nGenerating visualization for {n_hauls} hauls...")
    create_stacked_bar_chart(df, output_path=output_path, fig_size=(14, fig_height))
    
    print(f"\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
