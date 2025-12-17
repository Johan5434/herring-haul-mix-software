#!/usr/bin/env python3
"""
analyze_pc_combinations.py

Test all combinations of 2 vs 3 PCs for Reference and Spring PCA separately.
This isolates the effect of each PCA on classification performance.

Usage:
    python analyze_pc_combinations.py --hauls 41,45,47,...
"""

import sys
import os
import json
import numpy as np
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath("../Pipeline version 1"))

from data_loading import genotype_matrix
from pca_hauls import (
    build_reference_pca_from_GM,
    build_spring_pca_from_ref,
    project_individuals_to_PCs,
)
from classification import (
    compute_season_centroids,
    compute_population_centroids,
    apply_step1_rules_for_haul,
    rule_step2_and_3_within_spring
)


def project_ref_to_spring_pc(pc_ref_coords, ref_pca, spring_pca):
    """Transform coordinates from reference PC space to Spring PC space."""
    pc_ref = np.asarray(pc_ref_coords, dtype=float)
    ref_components = np.array(ref_pca["components"])
    spring_components = np.array(spring_pca["components"])
    
    G_std_approx = pc_ref @ ref_components
    pc_spring = G_std_approx @ spring_components.T
    
    return pc_spring


def load_simulated_hauls(metadata_file, vcf_file, haul_ids_to_test, ref_pca=None):
    """Load genotype data for specified simulated hauls."""
    print(f"Loading simulated hauls from {metadata_file}...")
    
    sample_to_hauls_list = {}
    with open(metadata_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            sample_id = parts[0]
            haul_id = parts[1]
            
            if haul_id in haul_ids_to_test:
                if sample_id not in sample_to_hauls_list:
                    sample_to_hauls_list[sample_id] = []
                sample_to_hauls_list[sample_id].append(haul_id)
    
    samples_in_hauls = set(sample_to_hauls_list.keys())
    
    G_filtered = None
    M_filtered = None
    missing = set()
    if ref_pca is not None and "_G_all" in ref_pca and "_M_all" in ref_pca:
        G_ref = ref_pca["_G_all"]
        M_ref = ref_pca["_M_all"]
        
        idx_map = {sid: i for i, sid in enumerate(M_ref[:, 0])}
        rows_G = []
        rows_M = []
        for sid in samples_in_hauls:
            idx = idx_map.get(sid)
            if idx is None:
                missing.add(sid)
                continue
            rows_G.append(G_ref[idx])
            rows_M.append(M_ref[idx])
        
        if rows_G:
            G_filtered = np.array(rows_G)
            M_filtered = np.array(rows_M, dtype=object)
    
    remaining = missing if G_filtered is not None else samples_in_hauls
    if remaining:
        original_metadata_path = "../Pipeline version 1/All_Sample_Metadata.txt"
        G_full, M_full = genotype_matrix(vcf_file, metadata_filename=original_metadata_path)
        
        mask = np.array([M_full[i, 0] in remaining for i in range(M_full.shape[0])])
        G_vcf = G_full[mask]
        M_vcf = M_full[mask]
        
        if G_filtered is None:
            G_filtered, M_filtered = G_vcf, M_vcf
        else:
            G_filtered = np.vstack([G_filtered, G_vcf])
            M_filtered = np.vstack([M_filtered, M_vcf])
    
    haul_to_indices = {}
    G_expanded = []
    M_expanded = []
    
    for i in range(M_filtered.shape[0]):
        sample_id = M_filtered[i, 0]
        hauls_for_sample = sample_to_hauls_list[sample_id]
        
        for haul_id in hauls_for_sample:
            G_expanded.append(G_filtered[i])
            M_row = [M_filtered[i, 0], haul_id, M_filtered[i, 2]]
            M_expanded.append(M_row)
            
            expanded_idx = len(G_expanded) - 1
            if haul_id not in haul_to_indices:
                haul_to_indices[haul_id] = []
            haul_to_indices[haul_id].append(expanded_idx)
    
    G_expanded = np.array(G_expanded)
    M_expanded = np.array(M_expanded, dtype=object)
    
    print(f"  ✓ Loaded {G_expanded.shape[0]} individual-haul assignments")
    
    return G_expanded, M_expanded, haul_to_indices


def project_hauls_to_pca(G, M, ref_pca, haul_to_indices):
    """Project hauls to PCA space."""
    PCs_new = project_individuals_to_PCs(
        G,
        ref_pca["mean_snps"],
        ref_pca["std_snps"],
        ref_pca["valid_snps"],
        ref_pca["components"]
    )
    
    haul_centroids = {}
    for haul_id, indices in sorted(haul_to_indices.items()):
        centroid = PCs_new[indices].mean(axis=0)
        haul_centroids[haul_id] = centroid
    
    return haul_centroids


def classify_hauls(haul_centroids, ref_pca, ground_truth, spring_pca=None):
    """Classify hauls."""
    season_to_centroid = compute_season_centroids(ref_pca)
    pop_to_centroid = compute_population_centroids(ref_pca)
    
    predictions = {}
    spring_relevant_hauls = []
    
    for haul_id in sorted(haul_centroids.keys()):
        centroid = haul_centroids[haul_id]
        true_mix = ground_truth[haul_id]
        
        step1_result = apply_step1_rules_for_haul(centroid, true_mix, season_to_centroid)
        
        pred_season = step1_result["pred_season_perc"]
        autumn_pct = pred_season["Autumn"]
        spring_pct = pred_season["Spring"]
        spring_relevant = step1_result["spring_relevant"]
        
        if spring_relevant:
            spring_relevant_hauls.append({
                "haul_id": haul_id,
                "centroid": centroid,
                "autumn_pct": autumn_pct,
                "spring_pct": spring_pct
            })
        
        spring_per_pop = int(spring_pct / 3)
        predictions[haul_id] = {
            "Autumn": int(autumn_pct),
            "North": spring_per_pop,
            "Central": spring_per_pop,
            "South": spring_per_pop
        }
    
    if spring_relevant_hauls and spring_pca is not None:
        spring_pop_to_centroid = compute_population_centroids(spring_pca)
        
        for haul_data in spring_relevant_hauls:
            haul_id = haul_data["haul_id"]
            centroid_ref = haul_data["centroid"]
            autumn_pct = haul_data["autumn_pct"]
            spring_pct = haul_data["spring_pct"]
            
            centroid_spring = project_ref_to_spring_pc(centroid_ref, ref_pca, spring_pca)
            
            spring_pops = {pop: spring_pop_to_centroid[pop] for pop in ["North", "Central", "South"] if pop in spring_pop_to_centroid}
            spring_result = rule_step2_and_3_within_spring(centroid_spring, spring_pops)
            
            pred_sub = spring_result["pred_sub_perc"]
            
            predictions[haul_id] = {
                "Autumn": int(autumn_pct),
                "North": int(pred_sub["North"] * spring_pct / 100),
                "Central": int(pred_sub["Central"] * spring_pct / 100),
                "South": int(pred_sub["South"] * spring_pct / 100)
            }
    
    return predictions


def calculate_errors(haul_ids, ground_truth, predictions):
    """Calculate MAE for each population and by step."""
    errors_A, errors_N, errors_C, errors_S = [], [], [], []
    step1_errors = []  # A vs Spring errors
    step2_errors = []  # N/C/S errors (for Spring hauls only)
    
    for haul_id in haul_ids:
        true = ground_truth[haul_id]
        pred = predictions[haul_id]
        
        errors_A.append(abs(pred["Autumn"] - true["Autumn"]))
        errors_N.append(abs(pred["North"] - true["North"]))
        errors_C.append(abs(pred["Central"] - true["Central"]))
        errors_S.append(abs(pred["South"] - true["South"]))
        
        # Step 1: Autumn vs Spring
        true_spring = true["North"] + true["Central"] + true["South"]
        pred_spring = pred["North"] + pred["Central"] + pred["South"]
        step1_errors.append(abs(true["Autumn"] - pred["Autumn"]) + abs(true_spring - pred_spring))
        
        # Step 2: N/C/S (only for Spring hauls)
        if true_spring > 0:
            step2_errors.append(abs(pred["North"] - true["North"]) + 
                              abs(pred["Central"] - true["Central"]) + 
                              abs(pred["South"] - true["South"]))
    
    return {
        "Autumn": np.mean(errors_A),
        "North": np.mean(errors_N),
        "Central": np.mean(errors_C),
        "South": np.mean(errors_S),
        "Overall": np.mean(errors_A + errors_N + errors_C + errors_S),
        "Step1_AvS": np.mean(step1_errors),
        "Step2_NCS": np.mean(step2_errors) if step2_errors else 0.0,
        "n_spring_hauls": len(step2_errors)
    }


def parse_haul_range(haul_spec):
    """Parse haul specification string into list of haul IDs."""
    haul_ids = []
    
    parts = haul_spec.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            start_num = int(start)
            end_num = int(end)
            for i in range(start_num, end_num + 1):
                haul_ids.append(f"haul_{i:04d}")
        else:
            num = int(part)
            haul_ids.append(f"haul_{num:04d}")
    
    return haul_ids


def load_ground_truth_proportions(proportions_file="haul_proportions.txt"):
    """Load ground truth proportions from haul_proportions.txt."""
    ground_truth = {}
    
    with open(proportions_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            haul_id = parts[0]
            autumn_pct = int(parts[2])
            north_pct = int(parts[3])
            central_pct = int(parts[4])
            south_pct = int(parts[5])
            
            ground_truth[haul_id] = {
                "Autumn": autumn_pct,
                "North": north_pct,
                "Central": central_pct,
                "South": south_pct,
            }
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description="Analyze PC combinations: Ref 2/3 × Spring 2/3")
    parser.add_argument("--hauls", required=True, help="Haul range to test")
    parser.add_argument("--vcf", default="../Pipeline version 1/Bioinformatics_Course_2025_Herring_Sample_Subset.vcf")
    parser.add_argument("--metadata", default="simulated_hauls_metadata.txt")
    parser.add_argument("--proportions", default="haul_proportions.txt")
    parser.add_argument("--qc-file", default="qc_passed_individuals.txt")
    parser.add_argument("--ref-metadata", default="../Pipeline version 1/All_Sample_Metadata.txt")
    
    args = parser.parse_args()
    
    print("="*80)
    print("PC COMBINATIONS ANALYSIS: Reference (2/3) × Spring (2/3)")
    print("="*80)
    
    haul_ids = parse_haul_range(args.hauls)
    print(f"\nTesting {len(haul_ids)} hauls")
    
    # Load reference data
    print("\nLoading reference data...")
    if os.path.exists(args.qc_file):
        with open(args.qc_file, 'r') as f:
            qc_ids = set(line.strip() for line in f if line.strip())
    else:
        qc_ids = None
    
    G_ref, M_ref = genotype_matrix(args.vcf, metadata_filename=args.ref_metadata)
    
    if qc_ids:
        mask = np.array([M_ref[i, 0] in qc_ids for i in range(M_ref.shape[0])])
        G_ref = G_ref[mask]
        M_ref = M_ref[mask]
    
    # Build all PCA combinations
    print("\nBuilding PCA models...")
    pca_models = {}
    for ref_n in [2, 3]:
        for spring_n in [2, 3]:
            print(f"  Building: Ref {ref_n} PCs + Spring {spring_n} PCs")
            ref_pca = build_reference_pca_from_GM(G_ref, M_ref, n_components=ref_n)
            spring_pca = build_spring_pca_from_ref(ref_pca, n_components=spring_n)
            pca_models[(ref_n, spring_n)] = {"ref": ref_pca, "spring": spring_pca}
    
    # Load simulated hauls
    print("\nLoading simulated hauls...")
    ground_truth = load_ground_truth_proportions(args.proportions)
    
    # Use 3-component model to load hauls (for SNP alignment)
    ref_pca_3 = pca_models[(3, 3)]["ref"]
    G_hauls, M_hauls, haul_to_indices = load_simulated_hauls(
        args.metadata, args.vcf, haul_ids, ref_pca=ref_pca_3
    )
    
    # Test all combinations
    print("\n" + "="*80)
    print("TESTING ALL COMBINATIONS")
    print("="*80)
    
    results = {}
    
    for (ref_n, spring_n) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        print(f"\n{'='*80}")
        print(f"Configuration: Reference {ref_n} PCs + Spring {spring_n} PCs")
        print(f"{'='*80}")
        
        ref_pca = pca_models[(ref_n, spring_n)]["ref"]
        spring_pca = pca_models[(ref_n, spring_n)]["spring"]
        
        haul_centroids = project_hauls_to_pca(G_hauls, M_hauls, ref_pca, haul_to_indices)
        predictions = classify_hauls(haul_centroids, ref_pca, ground_truth, spring_pca=spring_pca)
        errors = calculate_errors(haul_ids, ground_truth, predictions)
        
        results[(ref_n, spring_n)] = errors
        
        print(f"\n  MAE Results:")
        print(f"    Overall MAE:      {errors['Overall']:.2f}%")
        print(f"    Step 1 (A vs S):  {errors['Step1_AvS']:.2f}%")
        print(f"    Step 2 (N/C/S):   {errors['Step2_NCS']:.2f}% ({errors['n_spring_hauls']} Spring hauls)")
        print(f"    By population: A={errors['Autumn']:.2f}%, N={errors['North']:.2f}%, C={errors['Central']:.2f}%, S={errors['South']:.2f}%")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print("\nOverall MAE by Configuration:")
    print(f"{'Ref PCs':<10} {'Spring PCs':<12} {'Overall':<12} {'Step1(A/S)':<14} {'Step2(N/C/S)':<14} {'A':<8} {'N':<8} {'C':<8} {'S':<8}")
    print("-" * 100)
    
    for (ref_n, spring_n) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
        err = results[(ref_n, spring_n)]
        print(f"{ref_n:<10} {spring_n:<12} {err['Overall']:>6.2f}%      {err['Step1_AvS']:>6.2f}%        {err['Step2_NCS']:>6.2f}%        {err['Autumn']:>4.2f}%  {err['North']:>4.2f}%  {err['Central']:>4.2f}%  {err['South']:>4.2f}%")
    
    print("\n" + "="*80)
    print("ISOLATING EFFECTS")
    print("="*80)
    
    print("\nEffect of Reference PCA components (holding Spring constant):")
    print(f"  Overall MAE:")
    print(f"    Spring with 2 PCs: Ref 2→3 changes by {results[(3,2)]['Overall'] - results[(2,2)]['Overall']:+.2f} pp")
    print(f"    Spring with 3 PCs: Ref 2→3 changes by {results[(3,3)]['Overall'] - results[(2,3)]['Overall']:+.2f} pp")
    print(f"  Step 1 (A vs Spring) MAE:")
    print(f"    Spring with 2 PCs: Ref 2→3 changes by {results[(3,2)]['Step1_AvS'] - results[(2,2)]['Step1_AvS']:+.2f} pp")
    print(f"    Spring with 3 PCs: Ref 2→3 changes by {results[(3,3)]['Step1_AvS'] - results[(2,3)]['Step1_AvS']:+.2f} pp")
    
    print("\nEffect of Spring PCA components (holding Reference constant):")
    print(f"  Overall MAE:")
    print(f"    Ref with 2 PCs: Spring 2→3 changes by {results[(2,3)]['Overall'] - results[(2,2)]['Overall']:+.2f} pp")
    print(f"    Ref with 3 PCs: Spring 2→3 changes by {results[(3,3)]['Overall'] - results[(3,2)]['Overall']:+.2f} pp")
    print(f"  Step 2 (N/C/S) MAE:")
    print(f"    Ref with 2 PCs: Spring 2→3 changes by {results[(2,3)]['Step2_NCS'] - results[(2,2)]['Step2_NCS']:+.2f} pp")
    print(f"    Ref with 3 PCs: Spring 2→3 changes by {results[(3,3)]['Step2_NCS'] - results[(3,2)]['Step2_NCS']:+.2f} pp")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
