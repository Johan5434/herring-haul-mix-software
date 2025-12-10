#!/usr/bin/env python3
"""
analyze_individual_diversity.py

Analyze how many unique individuals are used across hauls and replicates.
Shows whether hauls 1-10 (same proportion) use different random individuals.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def load_haul_data():
    """Load haul metadata and proportions."""
    metadata = pd.read_csv("simulated_hauls_metadata.txt", sep="\t")
    proportions = pd.read_csv("haul_proportions.txt", sep="\t")
    return metadata, proportions


def analyze_diversity(metadata, proportions):
    """Analyze individual diversity across hauls and proportions."""
    
    print("="*70)
    print("INDIVIDUAL DIVERSITY ANALYSIS")
    print("="*70)
    
    # Group hauls by proportion
    prop_groups = proportions.groupby("proportion_name")
    
    for prop_name, group in prop_groups:
        print(f"\n{'='*70}")
        print(f"Proportion: {prop_name}")
        print(f"{'='*70}")
        
        haul_ids = group['haul_id'].tolist()
        
        # Get individuals for each haul in this proportion
        all_individuals = []
        overlap_matrix = np.zeros((len(haul_ids), len(haul_ids)), dtype=int)
        
        for i, haul_id in enumerate(haul_ids):
            individuals = set(metadata[metadata['haul_id'] == haul_id]['sample_id'].tolist())
            all_individuals.append(individuals)
            
            print(f"\n{haul_id}:")
            print(f"  Total individuals: {len(individuals)}")
            print(f"  Sample IDs: {sorted(list(individuals))[:5]}... (showing first 5)")
        
        # Calculate overlap between hauls
        print(f"\n{'─'*70}")
        print("OVERLAP BETWEEN REPLICATES:")
        print(f"{'─'*70}")
        
        for i, haul_i in enumerate(haul_ids):
            for j, haul_j in enumerate(haul_ids):
                if i < j:  # Only upper triangle
                    overlap = len(all_individuals[i] & all_individuals[j])
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
                    print(f"{haul_i} ∩ {haul_j}: {overlap} shared individuals "
                          f"({overlap/30*100:.1f}% of haul)")
        
        # Overall statistics for this proportion
        all_unique = set().union(*all_individuals)
        print(f"\n{'─'*70}")
        print(f"SUMMARY for {prop_name}:")
        print(f"  Total unique individuals across all 10 replicates: {len(all_unique)}")
        print(f"  Average individuals per haul: {np.mean([len(ind) for ind in all_individuals]):.1f}")
        print(f"  Average overlap between replicates: {np.mean(overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)]):.1f} individuals")
        print(f"  Min overlap: {np.min(overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)])} individuals")
        print(f"  Max overlap: {np.max(overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)])} individuals")
    
    # Global statistics
    print(f"\n{'='*70}")
    print("GLOBAL STATISTICS")
    print(f"{'='*70}")
    
    total_unique = len(metadata['sample_id'].unique())
    total_assignments = len(metadata)
    hauls_per_individual = metadata.groupby('sample_id').size()
    
    print(f"\nTotal unique individuals used: {total_unique}")
    print(f"Total individual assignments: {total_assignments}")
    print(f"Average hauls per individual: {hauls_per_individual.mean():.2f}")
    print(f"Max hauls per individual: {hauls_per_individual.max()}")
    print(f"Min hauls per individual: {hauls_per_individual.min()}")
    
    # Show most-used individuals
    most_used = hauls_per_individual.sort_values(ascending=False).head(10)
    print(f"\nMost frequently used individuals:")
    for sample_id, count in most_used.items():
        print(f"  {sample_id}: {count} hauls")
    
    # Histogram of usage
    usage_counts = hauls_per_individual.value_counts().sort_index()
    print(f"\nUsage distribution:")
    for n_hauls, n_individuals in usage_counts.items():
        bar = "█" * min(50, n_individuals)
        print(f"  Used in {n_hauls:2d} haul(s): {n_individuals:4d} individuals {bar}")


def check_replicate_randomness(metadata, proportions):
    """
    Check if replicates truly use different individuals.
    For each proportion, show what percentage of individuals are unique to each replicate.
    """
    print(f"\n{'='*70}")
    print("REPLICATE RANDOMNESS CHECK")
    print(f"{'='*70}")
    
    prop_groups = proportions.groupby("proportion_name")
    
    for prop_name, group in prop_groups:
        print(f"\n{prop_name}:")
        haul_ids = group['haul_id'].tolist()
        
        # Get all individuals for each haul
        haul_individuals = {}
        for haul_id in haul_ids:
            individuals = set(metadata[metadata['haul_id'] == haul_id]['sample_id'].tolist())
            haul_individuals[haul_id] = individuals
        
        # Calculate unique individuals per haul (not in any other haul)
        for haul_id in haul_ids:
            other_hauls = set().union(*[haul_individuals[h] for h in haul_ids if h != haul_id])
            unique_to_this_haul = haul_individuals[haul_id] - other_hauls
            print(f"  {haul_id}: {len(unique_to_this_haul)}/30 individuals unique to this replicate "
                  f"({len(unique_to_this_haul)/30*100:.1f}%)")


def main():
    """Main entry point."""
    metadata, proportions = load_haul_data()
    
    analyze_diversity(metadata, proportions)
    check_replicate_randomness(metadata, proportions)
    
    print(f"\n{'='*70}")
    print("✓ Analysis complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
