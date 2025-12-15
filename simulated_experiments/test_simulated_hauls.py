#!/usr/bin/env python3
"""
test_simulated_hauls.py

Test simulated hauls against a pre-built reference PCA model.
This script loads:
  - Reference PCA model (from reference_model/ directory)
  - Simulated hauls metadata (simulated_hauls_metadata.txt)
  - Original VCF file
  - Ground truth proportions (haul_proportions.txt)

Then projects simulated hauls into PCA space, classifies them,
and outputs results comparing true vs predicted proportions.

Usage:
    python test_simulated_hauls.py --hauls 1-15
    python test_simulated_hauls.py --hauls 16-30
    python test_simulated_hauls.py --hauls 1,5,10-20
"""

import sys
import os
import json
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add parent directory to path to import from simulations/
sys.path.insert(0, os.path.abspath("../simulations"))

from data_loading import genotype_matrix
from pca_hauls import project_individuals_to_PCs, build_spring_pca_from_ref
from classification import (
    compute_season_centroids,
    compute_population_centroids,
    apply_step1_rules_for_haul,
    rule_step2_and_3_within_spring
)


def load_reference_model(model_dir="reference_model"):
    """
    Load the pre-built reference PCA model.
    
    Expected structure:
      reference_model/
        reference_pca.json       - PCA parameters
        reference_centroids.csv  - Population centroids
    
    Returns:
      ref_pca: dict with PCA parameters
    """
    pca_path = os.path.join(model_dir, "reference_pca.json")
    
    if not os.path.exists(pca_path):
        raise FileNotFoundError(
            f"Reference model not found at {pca_path}.\n"
            f"Please run 'python build_reference_model.py' first."
        )
    
    print(f"Loading reference model from {model_dir}...")
    
    with open(pca_path, 'r') as f:
        ref_data = json.load(f)
    
    # Convert lists back to numpy arrays
    ref_pca = {
        "mean_snps": np.array(ref_data["mean_snps"]),
        "std_snps": np.array(ref_data["std_snps"]),
        "valid_snps": np.array(ref_data["valid_snps"], dtype=bool),
        "components": np.array(ref_data["components"]),
        "explained_var_ratio": np.array(ref_data["explained_var_ratio"]),
        "centroids": np.array(ref_data["centroids"]),
        "haul_ids": np.array(ref_data["haul_ids"]),
        "haul_pops": np.array(ref_data["haul_pops"]),
    }
    
    print(f"  ✓ Loaded reference PCA with {ref_pca['centroids'].shape[0]} haul centroids")
    print(f"  ✓ Explained variance: PC1={ref_pca['explained_var_ratio'][0]*100:.1f}%, "
          f"PC2={ref_pca['explained_var_ratio'][1]*100:.1f}%, "
          f"PC3={ref_pca['explained_var_ratio'][2]*100:.1f}%")
    
    return ref_pca


def load_spring_pca(model_dir="reference_model"):
    """
    Load the pre-built Spring PCA model.
    
    Expected structure:
      reference_model/
        spring_pca.json  - Spring-only PCA parameters
    
    Returns:
      spring_pca: dict with PCA parameters, or None if not found
    """
    spring_path = os.path.join(model_dir, "spring_pca.json")
    
    if not os.path.exists(spring_path):
        print(f"  ⚠ Spring PCA model not found at {spring_path}")
        print(f"    Will skip Step 2 (Spring PCA) classification")
        return None
    
    with open(spring_path, 'r') as f:
        spring_data = json.load(f)
    
    # Convert lists back to numpy arrays
    spring_pca = {
        "mean_snps": np.array(spring_data["mean_snps"]),
        "std_snps": np.array(spring_data["std_snps"]),
        "valid_snps": np.array(spring_data["valid_snps"], dtype=bool),
        "components": np.array(spring_data["components"]),
        "explained_var_ratio": np.array(spring_data["explained_var_ratio"]),
        "centroids": np.array(spring_data["centroids"]),
        "haul_ids": np.array(spring_data["haul_ids"]),
        "haul_pops": np.array(spring_data["haul_pops"]),
    }
    
    print(f"  ✓ Loaded Spring PCA with {spring_pca['centroids'].shape[0]} Spring haul centroids")
    
    return spring_pca


def parse_haul_range(haul_spec):
    """
    Parse haul specification string into list of haul IDs.
    
    Examples:
      "1-15"      -> ["haul_0001", "haul_0002", ..., "haul_0015"]
      "1,5,10-12" -> ["haul_0001", "haul_0005", "haul_0010", "haul_0011", "haul_0012"]
    
    Returns:
      list of haul_id strings
    """
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
    """
    Load ground truth proportions from haul_proportions.txt.
    
    Returns:
      dict: {haul_id: {"Autumn": %, "North": %, "Central": %, "South": %}}
    """
    ground_truth = {}
    
    with open(proportions_file, 'r') as f:
        header = f.readline()  # skip header
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


def load_simulated_hauls(metadata_file, vcf_file, haul_ids_to_test):
    """
    Load genotype data for specified simulated hauls.
    
    Parameters:
      metadata_file: path to simulated_hauls_metadata.txt
      vcf_file: path to original VCF
      haul_ids_to_test: list of haul IDs to load (e.g., ["haul_0001", "haul_0002"])
    
    Returns:
      G: genotype matrix (n_individuals, n_snps)
      M: metadata matrix (n_individuals, 3) [sample_id, haul_id, population]
      haul_to_indices: dict mapping haul_id -> list of row indices in G/M
    """
    print(f"\nLoading simulated hauls from {metadata_file}...")
    
    # Read metadata and filter to requested hauls
    # IMPORTANT: Keep individuals in ALL hauls they appear in
    sample_to_hauls_list = {}  # sample_id -> list of haul_ids (in order from metadata)
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
    
    print(f"  ✓ Found {len(sample_to_hauls_list)} individuals")
    
    # Load original metadata to get population info
    original_metadata_path = "../simulations/All_Sample_Metadata.txt"
    sample_to_pop = {}
    with open(original_metadata_path, 'r') as f:
        header = f.readline()
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) < 6:
                continue
            sample_to_pop[cols[0]] = cols[5]  # Population is column 5
    
    # Load VCF with proper metadata path
    print(f"  Loading genotypes from VCF...")
    G, M = genotype_matrix(vcf_file, metadata_filename=original_metadata_path)
    
    # Create a set of all samples that appear in any haul we're testing
    samples_in_hauls = set(sample_to_hauls_list.keys())
    
    # Filter to only samples in our hauls
    mask = np.array([M[i, 0] in samples_in_hauls for i in range(M.shape[0])])
    G_filtered = G[mask]
    M_filtered = M[mask]
    
    print(f"  ✓ Loaded {G_filtered.shape[0]} individuals, {G_filtered.shape[1]} SNPs")
    
    # Build haul_to_indices mapping
    # For individuals in multiple hauls, we need to duplicate them
    haul_to_indices = {}
    G_expanded = []
    M_expanded = []
    
    for i in range(M_filtered.shape[0]):
        sample_id = M_filtered[i, 0]
        hauls_for_sample = sample_to_hauls_list[sample_id]
        
        # Add this individual once for each haul it appears in
        for haul_id in hauls_for_sample:
            G_expanded.append(G_filtered[i])
            M_row = [M_filtered[i, 0], haul_id, M_filtered[i, 2]]
            M_expanded.append(M_row)
            
            # Track the index for this haul
            expanded_idx = len(G_expanded) - 1
            if haul_id not in haul_to_indices:
                haul_to_indices[haul_id] = []
            haul_to_indices[haul_id].append(expanded_idx)
    
    # Convert back to numpy arrays
    G_expanded = np.array(G_expanded)
    M_expanded = np.array(M_expanded, dtype=object)
    
    print(f"  ✓ Loaded {G_expanded.shape[0]} individual-haul assignments, {G_expanded.shape[1]} SNPs")
    
    return G_expanded, M_expanded, haul_to_indices


def project_hauls_to_pca(G, M, ref_pca, haul_to_indices):
    """
    Project simulated hauls into reference PCA space.
    
    Returns:
      haul_centroids: dict {haul_id: PC coordinates (n_components,)}
    """
    print("\nProjecting simulated hauls to PCA space...")
    
    # Project all individuals
    PCs_new = project_individuals_to_PCs(
        G,
        ref_pca["mean_snps"],
        ref_pca["std_snps"],
        ref_pca["valid_snps"],
        ref_pca["components"]
    )
    
    # Compute centroids per haul
    haul_centroids = {}
    for haul_id, indices in sorted(haul_to_indices.items()):  # Sort for consistency
        centroid = PCs_new[indices].mean(axis=0)
        haul_centroids[haul_id] = centroid
    
    print(f"  ✓ Computed centroids for {len(haul_centroids)} hauls")
    
    return haul_centroids


def project_ref_to_spring_pc(pc_ref_coords, ref_pca, spring_pca):
    """
    Transform a point from reference PC space to Spring PC space.
    
    Both PCAs use the same SNP set, so we can transform between them:
      1. Reconstruct standardized genotype space: G_std ≈ pc_ref @ ref_components
      2. Project to Spring space: pc_spring = G_std @ spring_components.T
    
    Parameters:
      pc_ref_coords: coordinates in reference PC space (1D array)
      ref_pca: reference PCA model
      spring_pca: spring PCA model
    
    Returns:
      coordinates in Spring PC space
    """
    pc_ref = np.asarray(pc_ref_coords, dtype=float)
    ref_components = np.array(ref_pca["components"])  # (n_components, n_snps)
    spring_components = np.array(spring_pca["components"])  # (n_components, n_snps)
    
    # Reconstruct standardized genotype space
    G_std_approx = pc_ref @ ref_components
    
    # Project to Spring space
    pc_spring = G_std_approx @ spring_components.T
    
    return pc_spring


def classify_hauls(haul_centroids, ref_pca, ground_truth, spring_pca=None, use_spring_pca=True):
    """
    Classify simulated hauls using the pipeline's rule-based classification.
    
    Step 1: ALL hauls classified to Autumn vs Spring only
    Step 2: IF Spring ≥90%, break Spring down into North/Central/South
            (Uses Spring PC space projection if spring_pca provided)
    
    Parameters:
      haul_centroids: dict {haul_id: PC coordinates in reference space}
      ref_pca: reference PCA model (used for Season centroids)
      ground_truth: dict {haul_id: true proportions}
      spring_pca: Spring PCA model (for Step 2 classification)
      use_spring_pca: if True, run N/C/S classification for Spring-dominant hauls
    
    Returns:
      predictions: dict {haul_id: {"Autumn": %, "North": %, "Central": %, "South": %}}
    """
    print("\nClassifying simulated hauls...")
    print("  Step 1: Classifying all hauls to Autumn vs Spring")
    
    # Compute season centroids (Autumn vs Spring)
    season_to_centroid = compute_season_centroids(ref_pca)
    print(f"  ✓ Computed Autumn and Spring centroids from reference")
    
    # Get Spring population centroids from reference PCA
    pop_to_centroid = compute_population_centroids(ref_pca)
    print(f"  ✓ Computed N/C/S centroids from reference")
    
    # Step 1: Classify ALL hauls to Autumn vs Spring
    predictions = {}
    spring_relevant_hauls = []  # Track which hauls need Step 2
    
    # Process hauls in sorted order to ensure consistent results
    for haul_id in sorted(haul_centroids.keys()):
        centroid = haul_centroids[haul_id]
        true_mix = ground_truth[haul_id]
        
        # Step 1: Autumn vs Spring
        step1_result = apply_step1_rules_for_haul(centroid, true_mix, season_to_centroid)
        
        pred_season = step1_result["pred_season_perc"]
        autumn_pct = pred_season["Autumn"]
        spring_pct = pred_season["Spring"]
        spring_relevant = step1_result["spring_relevant"]
        
        # Check if this haul should proceed to Step 2
        if use_spring_pca and spring_relevant:
            spring_relevant_hauls.append({
                "haul_id": haul_id,
                "centroid": centroid,
                "autumn_pct": autumn_pct,
                "spring_pct": spring_pct
            })
        
        # For now, initialize with equal Spring distribution (will update if Step 2 runs)
        spring_per_pop = int(spring_pct / 3)
        predictions[haul_id] = {
            "Autumn": int(autumn_pct),
            "North": spring_per_pop,
            "Central": spring_per_pop,
            "South": spring_per_pop
        }
    
    print(f"  ✓ Step 1 complete: {len(haul_centroids)} hauls classified (A/S only)")
    
    # Step 2: For Spring-relevant hauls, break Spring into N/C/S
    if use_spring_pca and spring_relevant_hauls and spring_pca is not None:
        print(f"\n  Step 2: Breaking down Spring for {len(spring_relevant_hauls)} hauls (Spring ≥90%)")
        print(f"         Using Spring PC space transformation")
        
        # Get Spring population centroids in Spring PC space
        spring_pop_to_centroid = compute_population_centroids(spring_pca)
        
        for haul_data in spring_relevant_hauls:
            haul_id = haul_data["haul_id"]
            centroid_ref = haul_data["centroid"]
            autumn_pct = haul_data["autumn_pct"]
            spring_pct = haul_data["spring_pct"]
            
            # Transform haul centroid from reference space to Spring space
            centroid_spring = project_ref_to_spring_pc(centroid_ref, ref_pca, spring_pca)
            
            # Classify within Spring populations using Spring PC space
            spring_pops = {pop: spring_pop_to_centroid[pop] for pop in ["North", "Central", "South"] if pop in spring_pop_to_centroid}
            spring_result = rule_step2_and_3_within_spring(centroid_spring, spring_pops)
            
            pred_sub = spring_result["pred_sub_perc"]
            
            # Scale Spring sub-populations by total Spring percentage
            predictions[haul_id] = {
                "Autumn": int(autumn_pct),
                "North": int(pred_sub["North"] * spring_pct / 100),
                "Central": int(pred_sub["Central"] * spring_pct / 100),
                "South": int(pred_sub["South"] * spring_pct / 100)
            }
        
        print(f"  ✓ Step 2 complete: Spring populations separated for {len(spring_relevant_hauls)} hauls")
    elif use_spring_pca and spring_relevant_hauls and spring_pca is None:
        print(f"\n  Step 2: Spring PCA not loaded, using equal Spring distribution for {len(spring_relevant_hauls)} hauls")
        # Need to fix the predictions for spring_relevant_hauls since they still have equal distribution
        for haul_data in spring_relevant_hauls:
            haul_id = haul_data["haul_id"]
            autumn_pct = haul_data["autumn_pct"]
            spring_pct = haul_data["spring_pct"]
            spring_per_pop = int(spring_pct / 3)
            predictions[haul_id] = {
                "Autumn": int(autumn_pct),
                "North": spring_per_pop,
                "Central": spring_per_pop,
                "South": spring_per_pop
            }
    elif use_spring_pca and not spring_relevant_hauls:
        print(f"\n  Step 2: No hauls with Spring ≥90%")
    
    print(f"\n  ✓ Classification complete: {len(predictions)} hauls (A/N/C/S)")
    
    return predictions


def write_results_csv(haul_ids, ground_truth, predictions, output_dir="results"):
    """
    Write results to CSV file with timestamp.
    
    Format:
      haul_id, true_A%, true_N%, true_C%, true_S%, pred_A%, pred_N%, pred_C%, pred_S%, error_A, error_N, error_C, error_S
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    haul_range = f"{haul_ids[0].split('_')[1]}-{haul_ids[-1].split('_')[1]}"
    output_file = os.path.join(output_dir, f"predictions_{haul_range}_{timestamp}.csv")
    
    with open(output_file, 'w') as f:
        # Header
        f.write("haul_id,true_Autumn,true_North,true_Central,true_South,"
                "pred_Autumn,pred_North,pred_Central,pred_South,"
                "error_Autumn,error_North,error_Central,error_South\n")
        
        # Data rows
        for haul_id in sorted(haul_ids):
            true = ground_truth[haul_id]
            pred = predictions[haul_id]
            
            error_A = pred["Autumn"] - true["Autumn"]
            error_N = pred["North"] - true["North"]
            error_C = pred["Central"] - true["Central"]
            error_S = pred["South"] - true["South"]
            
            f.write(f"{haul_id},"
                   f"{true['Autumn']},{true['North']},{true['Central']},{true['South']},"
                   f"{pred['Autumn']},{pred['North']},{pred['Central']},{pred['South']},"
                   f"{error_A},{error_N},{error_C},{error_S}\n")
    
    print(f"\n✓ Results saved to: {output_file}")
    return output_file


def print_summary(haul_ids, ground_truth, predictions, ref_pca, haul_centroids):
    """Print detailed summary with Step 1 (A/S) and Step 2 (N/C/S for Spring) results."""
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("="*80)
    
    # Separate hauls by whether they're Spring-relevant
    season_to_centroid = compute_season_centroids(ref_pca)
    
    step1_results = {}  # Store A/S only predictions
    step2_hauls = []    # Track which hauls had Step 2
    
    for haul_id in haul_ids:
        centroid = haul_centroids[haul_id]
        true_mix = ground_truth[haul_id]
        
        # Recompute Step 1 to get A/S
        step1_result = apply_step1_rules_for_haul(centroid, true_mix, season_to_centroid)
        pred_season = step1_result["pred_season_perc"]
        spring_relevant = step1_result["spring_relevant"]
        
        step1_results[haul_id] = {
            "autumn": pred_season["Autumn"],
            "spring": pred_season["Spring"],
            "spring_relevant": spring_relevant
        }
        
        if spring_relevant:
            step2_hauls.append(haul_id)
    
    # Calculate true A/S proportions
    true_autumn_spring = {}
    for haul_id in haul_ids:
        true = ground_truth[haul_id]
        true_a = true["Autumn"]
        true_s = true["North"] + true["Central"] + true["South"]
        true_autumn_spring[haul_id] = {"autumn": true_a, "spring": true_s}
    
    print(f"\nTested {len(haul_ids)} hauls: {haul_ids[0]} to {haul_ids[-1]}")
    
    # ===== STEP 1 RESULTS =====
    print(f"\n" + "="*80)
    print("STEP 1: AUTUMN VS SPRING CLASSIFICATION")
    print("="*80)
    
    errors_A_step1, errors_S_step1 = [], []
    for haul_id in haul_ids:
        true_as = true_autumn_spring[haul_id]
        pred_as = step1_results[haul_id]
        
        errors_A_step1.append(abs(pred_as["autumn"] - true_as["autumn"]))
        errors_S_step1.append(abs(pred_as["spring"] - true_as["spring"]))
    
    print(f"\nMean Absolute Error (MAE) for A/S classification:")
    print(f"  Autumn: {np.mean(errors_A_step1):.1f}%")
    print(f"  Spring: {np.mean(errors_S_step1):.1f}%")
    
    print(f"\nStep 1 predictions for all hauls:")
    print(f"{'Haul ID':<12} {'True A/S':<15} {'Pred A/S':<15} {'Error':<8} {'S-rel'}")
    print("-" * 60)
    
    for haul_id in haul_ids:
        true_as = true_autumn_spring[haul_id]
        pred_as = step1_results[haul_id]
        
        true_str = f"A:{true_as['autumn']:>3}% S:{true_as['spring']:>3}%"
        pred_str = f"A:{pred_as['autumn']:>3}% S:{pred_as['spring']:>3}%"
        error = abs(pred_as["autumn"] - true_as["autumn"]) + abs(pred_as["spring"] - true_as["spring"])
        srel = "✓" if step1_results[haul_id]["spring_relevant"] else " "
        
        print(f"{haul_id:<12} {true_str:<15} {pred_str:<15} {error:>6.0f}%  {srel}")
    
    # ===== STEP 2 RESULTS (if applicable) =====
    if step2_hauls:
        print(f"\n" + "="*80)
        print(f"STEP 2: SPRING N/C/S BREAKDOWN ({len(step2_hauls)} hauls with Spring ≥90%)")
        print("="*80)
        
        errors_N, errors_C, errors_S = [], [], []
        
        for haul_id in step2_hauls:
            true = ground_truth[haul_id]
            pred = predictions[haul_id]
            
            errors_N.append(abs(pred["North"] - true["North"]))
            errors_C.append(abs(pred["Central"] - true["Central"]))
            errors_S.append(abs(pred["South"] - true["South"]))
        
        print(f"\nMean Absolute Error (MAE) for N/C/S classification:")
        print(f"  North:   {np.mean(errors_N):.1f}%")
        print(f"  Central: {np.mean(errors_C):.1f}%")
        print(f"  South:   {np.mean(errors_S):.1f}%")
        
        print(f"\nStep 2 predictions for all {len(step2_hauls)} Spring-relevant hauls:")
        print(f"{'Haul ID':<12} {'True N/C/S':<20} {'Pred N/C/S':<20} {'Error'}")
        print("-" * 60)
        
        for haul_id in step2_hauls:
            true = ground_truth[haul_id]
            pred = predictions[haul_id]
            
            true_str = f"N:{true['North']:>3}% C:{true['Central']:>3}% S:{true['South']:>3}%"
            pred_str = f"N:{pred['North']:>3}% C:{pred['Central']:>3}% S:{pred['South']:>3}%"
            
            error = (abs(pred["North"] - true["North"]) + 
                    abs(pred["Central"] - true["Central"]) + 
                    abs(pred["South"] - true["South"]))
            
            print(f"{haul_id:<12} {true_str:<20} {pred_str:<20} {error:.0f}%")
    else:
        print(f"\n  No hauls qualified for Step 2 (all have Spring < 90%)")
    
    # ===== OVERALL RESULTS =====
    print(f"\n" + "="*80)
    print("OVERALL RESULTS (All 4 populations: A/N/C/S)")
    print("="*80)
    
    errors_A, errors_N, errors_C, errors_S = [], [], [], []
    for haul_id in haul_ids:
        true = ground_truth[haul_id]
        pred = predictions[haul_id]
        
        errors_A.append(abs(pred["Autumn"] - true["Autumn"]))
        errors_N.append(abs(pred["North"] - true["North"]))
        errors_C.append(abs(pred["Central"] - true["Central"]))
        errors_S.append(abs(pred["South"] - true["South"]))
    
    print(f"\nMean Absolute Error (MAE) by population:")
    print(f"  Autumn:  {np.mean(errors_A):.1f}%")
    print(f"  North:   {np.mean(errors_N):.1f}%")
    print(f"  Central: {np.mean(errors_C):.1f}%")
    print(f"  South:   {np.mean(errors_S):.1f}%")
    print(f"  Overall: {np.mean(errors_A + errors_N + errors_C + errors_S):.1f}%")
    
    print("="*80 + "\n")


def plot_pca_with_hauls(ref_pca, haul_centroids, ground_truth, haul_ids, output_dir="results"):
    """
    Create a PCA plot showing reference hauls, centroids, and simulated hauls.
    Saves as PNG in output_dir and displays on screen.
    
    Parameters:
      haul_ids: list of haul IDs tested (used for filename)
    """
    print("\nGenerating PCA visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Get reference data
    ref_centroids = ref_pca["centroids"]
    ref_pops = ref_pca["haul_pops"]
    
    # Compute season centroids (Autumn vs Spring)
    season_to_centroid = compute_season_centroids(ref_pca)
    
    # Compute population centroids (N/C/S)
    pop_to_centroid = compute_population_centroids(ref_pca)
    
    # Define colors for each population
    pop_colors = {
        "Autumn": "#FF6B35",      # Orange-red
        "North": "#004E89",       # Dark blue
        "Central": "#1B9CFC",     # Light blue
        "South": "#F7DC6F"        # Yellow
    }
    
    # ===== SUBPLOT 1: PC1 vs PC2 =====
    ax = axes[0]
    
    # Plot reference hauls
    for pop in ["Autumn", "North", "Central", "South"]:
        mask = ref_pops == pop
        ax.scatter(ref_centroids[mask, 0], ref_centroids[mask, 1], 
                  c=pop_colors[pop], s=100, alpha=0.6, label=f"Ref {pop}", 
                  marker='o', edgecolors='black', linewidth=0.5)
    
    # Plot population centroids (used for classification)
    for pop in ["Autumn", "North", "Central", "South"]:
        centroid = pop_to_centroid[pop]
        ax.scatter(centroid[0], centroid[1], c=pop_colors[pop], s=300, 
                  alpha=1.0, marker='*', edgecolors='black', linewidth=2,
                  label=f"{pop} centroid", zorder=10)
    
    # Plot Spring centroid (merged N+C+S, used in Step 1)
    spring_centroid = season_to_centroid["Spring"]
    ax.scatter(spring_centroid[0], spring_centroid[1], c='#2ECC71', s=400, 
              alpha=1.0, marker='*', edgecolors='black', linewidth=2.5,
              label="Spring centroid (Step 1)", zorder=11)
    
    # Plot simulated hauls (all in same distinct color)
    first_sim = True
    for haul_id, centroid in haul_centroids.items():
        label = "Simulated hauls" if first_sim else None
        ax.scatter(centroid[0], centroid[1], c='#9B59B6', s=50, 
                  alpha=0.9, marker='X', edgecolors='black', linewidth=0.8, label=label)
        first_sim = False
    
    ax.set_xlabel(f"PC1 ({ref_pca['explained_var_ratio'][0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({ref_pca['explained_var_ratio'][1]*100:.1f}%)", fontsize=12)
    ax.set_title(f"Ref Hauls (circles), Centroids (stars), Simulated (X)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='upper left', fontsize=7, title="Reference Data", ncol=2)
    
    # ===== SUBPLOT 2: PC1 vs PC3 =====
    ax = axes[1]
    
    # Plot reference hauls
    for pop in ["Autumn", "North", "Central", "South"]:
        mask = ref_pops == pop
        ax.scatter(ref_centroids[mask, 0], ref_centroids[mask, 2], 
                  c=pop_colors[pop], s=100, alpha=0.6, label=f"Ref {pop}", 
                  marker='o', edgecolors='black', linewidth=0.5)
    
    # Plot population centroids (used for classification)
    for pop in ["Autumn", "North", "Central", "South"]:
        centroid = pop_to_centroid[pop]
        ax.scatter(centroid[0], centroid[2], c=pop_colors[pop], s=300, 
                  alpha=1.0, marker='*', edgecolors='black', linewidth=2,
                  label=f"{pop} centroid", zorder=10)
    
    # Plot Spring centroid (merged N+C+S, used in Step 1)
    spring_centroid = season_to_centroid["Spring"]
    ax.scatter(spring_centroid[0], spring_centroid[2], c='#2ECC71', s=400, 
              alpha=1.0, marker='*', edgecolors='black', linewidth=2.5,
              label="Spring centroid (Step 1)", zorder=11)
    
    # Plot simulated hauls (all in same distinct color)
    first_sim = True
    for haul_id, centroid in haul_centroids.items():
        label = "Simulated hauls" if first_sim else None
        ax.scatter(centroid[0], centroid[2], c='#9B59B6', s=50, 
                  alpha=0.9, marker='X', edgecolors='black', linewidth=0.8, label=label)
        first_sim = False
    
    ax.set_xlabel(f"PC1 ({ref_pca['explained_var_ratio'][0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC3 ({ref_pca['explained_var_ratio'][2]*100:.1f}%)", fontsize=12)
    ax.set_title(f"Ref Hauls (circles), Centroids (stars), Simulated (X)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure with haul range in filename
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    haul_range = f"{haul_ids[0].split('_')[1]}-{haul_ids[-1].split('_')[1]}"
    plot_file = os.path.join(output_dir, f"pca_plot_{haul_range}_{timestamp}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ PCA plot saved to: {plot_file}")
    
    # Display on screen (non-blocking)
    try:
        plt.show(block=False)
        print(f"  ✓ PCA plot displayed on screen")
    except Exception as e:
        print(f"  (Could not display plot on screen, but saved to file)")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test simulated hauls against reference PCA")
    parser.add_argument("--hauls", default="1-15", help="Haul range to test (e.g., '1-15', '16-30', '1,5,10-20')")
    parser.add_argument("--vcf", default="../simulations/Bioinformatics_Course_2025_Herring_Sample_Subset.vcf",
                       help="Path to original VCF file")
    parser.add_argument("--metadata", default="simulated_hauls_metadata.txt",
                       help="Path to simulated hauls metadata")
    parser.add_argument("--proportions", default="haul_proportions.txt",
                       help="Path to ground truth proportions")
    parser.add_argument("--model-dir", default="reference_model",
                       help="Directory containing reference model")
    parser.add_argument("--output-dir", default="results",
                       help="Directory for output CSV files")
    
    args = parser.parse_args()
    
    print("="*80)
    print("SIMULATED HAUL TESTING PIPELINE")
    print("="*80)
    
    # Parse haul range
    haul_ids = parse_haul_range(args.hauls)
    print(f"\nTesting hauls: {haul_ids[0]} to {haul_ids[-1]} ({len(haul_ids)} hauls)")
    
    # Load reference model
    ref_pca = load_reference_model(args.model_dir)
    
    # Load Spring PCA (optional)
    spring_pca = load_spring_pca(args.model_dir)
    
    # Load ground truth
    ground_truth = load_ground_truth_proportions(args.proportions)
    
    # Load simulated haul data
    G, M, haul_to_indices = load_simulated_hauls(args.metadata, args.vcf, haul_ids)
    
    # Project to PCA space
    haul_centroids = project_hauls_to_pca(G, M, ref_pca, haul_to_indices)
    
    # Classify hauls
    predictions = classify_hauls(haul_centroids, ref_pca, ground_truth, spring_pca=spring_pca, use_spring_pca=True)
    
    # Write results
    csv_file = write_results_csv(haul_ids, ground_truth, predictions, args.output_dir)
    
    # Print detailed summary
    print_summary(haul_ids, ground_truth, predictions, ref_pca, haul_centroids)
    
    # Generate PCA plot
    plot_pca_with_hauls(ref_pca, haul_centroids, ground_truth, haul_ids, args.output_dir)
    
    print(f"✓ Testing complete! Results saved to: {csv_file}\n")


if __name__ == "__main__":
    main()
