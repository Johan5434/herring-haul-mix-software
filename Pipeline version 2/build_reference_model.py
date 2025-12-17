#!/usr/bin/env python3
"""
build_reference_model.py

Rebuild the reference PCA model from the original empirical hauls.
This includes:
  1. Load all individuals from the original VCF
  2. Build PCA from all individuals
  3. Calculate haul centroids for ALL empirical hauls
  4. Save reference model with empirical haul data

Run this whenever you want to rebuild the reference from scratch.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path (shared modules live in Pipeline version 1)
sys.path.insert(0, os.path.abspath("../Pipeline version 1"))

from data_loading import genotype_matrix
from pca_hauls import project_individuals_to_PCs


def build_reference_model_from_empirical(vcf_file, metadata_file, output_dir="reference_model_current", 
                                         qc_passed_file="qc_passed_individuals.txt"):
    """
    Build reference PCA model from empirical hauls using QC-passed individuals.
    
    Args:
        vcf_file: Path to VCF file with original data
        metadata_file: Path to metadata file with haul assignments
        output_dir: Where to save the model
        qc_passed_file: Path to file with QC-passed sample IDs (one per line)
    """
    
    print("="*80)
    print("BUILDING REFERENCE PCA MODEL FROM EMPIRICAL HAULS")
    print("="*80)
    
    # Load QC-passed individuals
    print("\n1. Loading QC-passed individuals...")
    qc_passed_ids = set()
    if os.path.exists(qc_passed_file):
        with open(qc_passed_file, 'r') as f:
            qc_passed_ids = set(line.strip() for line in f if line.strip())
        print(f"   ✓ Loaded {len(qc_passed_ids)} QC-passed sample IDs from {qc_passed_file}")
    else:
        print(f"   ⚠ Warning: QC file not found ({qc_passed_file}), using all individuals")
    
    # Load data
    print("\n2. Loading empirical data...")
    G_tuple = genotype_matrix(vcf_file, metadata_file)
    # genotype_matrix returns (genotypes, metadata)
    if isinstance(G_tuple, tuple):
        G, M = G_tuple
    else:
        G = G_tuple
        M = None
    
    metadata = pd.read_csv(metadata_file, sep="\t")
    
    print(f"   ✓ Loaded genotypes: {G.shape[0]} samples × {G.shape[1]} SNPs")
    print(f"   ✓ Loaded metadata: {len(metadata)} samples")
    
    # Filter to QC-passed individuals
    if qc_passed_ids and M is not None:
        print("\n3. Filtering to QC-passed individuals...")
        sample_ids = M[:, 0]  # First column is sample ID
        qc_mask = np.array([sid in qc_passed_ids for sid in sample_ids])
        n_before = G.shape[0]
        G = G[qc_mask]
        M = M[qc_mask]
        metadata = metadata[metadata.iloc[:, 0].isin(qc_passed_ids)]
        print(f"   ✓ Kept {G.shape[0]} / {n_before} individuals ({G.shape[0]/n_before*100:.1f}%)")
    else:
        print("\n3. No QC filtering applied (using all individuals)")
    
    print(f"   ✓ Final dataset: {G.shape[0]} samples × {G.shape[1]} SNPs")
    
    # Build PCA from all QC-passed individuals
    print("\n4. Building PCA from QC-passed individuals...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Standardize genotypes
    scaler = StandardScaler()
    G_scaled = scaler.fit_transform(G)
    
    # Build PCA
    pca_model = PCA(n_components=3)
    G_pca = pca_model.fit_transform(G_scaled)
    
    print(f"   ✓ Explained variance: PC1={pca_model.explained_variance_ratio_[0]*100:.2f}%, "
          f"PC2={pca_model.explained_variance_ratio_[1]*100:.2f}%, "
          f"PC3={pca_model.explained_variance_ratio_[2]*100:.2f}%")
    print(f"   ✓ Total: {pca_model.explained_variance_ratio_.sum()*100:.2f}%")
    
    # Calculate haul centroids
    print("\n5. Calculating empirical haul centroids...")
    haul_centroids = {}
    haul_pops = {}
    valid_hauls = []
    
    for haul_id in metadata["Capture_Population"].unique():
        if haul_id == "Capture_Population":  # Skip header if present
            continue
        
        haul_mask = metadata["Capture_Population"] == haul_id
        haul_indices = np.where(haul_mask)[0]
        
        if len(haul_indices) > 0:
            centroid = G_pca[haul_indices].mean(axis=0)
            haul_centroids[haul_id] = centroid
            
            # Get population for this haul
            pop = metadata[haul_mask]["Population"].iloc[0]
            haul_pops[haul_id] = pop
            valid_hauls.append(haul_id)
    
    print(f"   ✓ Calculated centroids for {len(valid_hauls)} empirical hauls")
    print(f"   Populations: Autumn={sum(1 for p in haul_pops.values() if p=='Autumn')}, "
          f"North={sum(1 for p in haul_pops.values() if p=='North')}, "
          f"Central={sum(1 for p in haul_pops.values() if p=='Central')}, "
          f"South={sum(1 for p in haul_pops.values() if p=='South')}")
    
    # Prepare model data
    print("\n6. Preparing model data...")
    
    # Get PCA parameters
    mean_snps = scaler.mean_
    std_snps = scaler.scale_
    
    # Get valid SNPs (those not all zeros)
    valid_snps = (std_snps > 0).astype(bool)
    
    # Prepare centroids array
    centroid_array = np.array([haul_centroids[hid] for hid in valid_hauls])
    haul_pops_array = np.array([haul_pops[hid] for hid in valid_hauls])
    
    # Create reference model
    ref_model = {
        "mean_snps": mean_snps.tolist(),
        "std_snps": std_snps.tolist(),
        "valid_snps": valid_snps.tolist(),
        "components": pca_model.components_.tolist(),
        "explained_var_ratio": pca_model.explained_variance_ratio_.tolist(),
        "centroids": centroid_array.tolist(),
        "haul_ids": valid_hauls,
        "haul_pops": haul_pops_array.tolist(),
        "_G_all": G.tolist(),  # Store genotype matrix for Spring PCA
        "_M_all": M.tolist(),  # Store metadata for Spring PCA
    }
    
    # Save model
    print("\n7. Saving reference model...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Safety: never overwrite the pinned model directory
    pinned = os.path.abspath(os.path.join(os.path.dirname(__file__), 'reference_model_v1npz'))
    if os.path.abspath(output_dir) == pinned:
        raise RuntimeError("Refusing to overwrite pinned model directory: reference_model_v1npz. Use a different output_dir.")
    model_path = os.path.join(output_dir, "reference_pca.json")
    with open(model_path, 'w') as f:
        json.dump(ref_model, f, indent=2)
    
    print(f"   ✓ Saved to {model_path}")
    
    # Build Spring PCA
    print("\n8. Building Spring PCA (for N/C/S classification)...")
    from pca_hauls import build_spring_pca_from_ref
    
    # Convert to numpy arrays for spring PCA building
    ref_model_for_spring = {
        "_G_all": G,  # Keep as numpy array
        "_M_all": M,  # Keep as numpy array
    }
    spring_pca = build_spring_pca_from_ref(ref_model_for_spring, n_components=3)
    # Convert any numpy arrays in spring_pca to JSON-serializable lists
    spring_pca_json = {}
    for k, v in spring_pca.items():
        try:
            spring_pca_json[k] = v.tolist() if hasattr(v, "tolist") else v
        except Exception:
            spring_pca_json[k] = v
    
    spring_path = os.path.join(output_dir, "spring_pca.json")
    with open(spring_path, 'w') as f:
        json.dump(spring_pca_json, f, indent=2)
    
        print(f"   ✓ Saved Spring PCA to {spring_path}")
        print(f"   ✓ Spring PCA variance: PC1={spring_pca_json['explained_var_ratio'][0]*100:.2f}%, "
            f"PC2={spring_pca_json['explained_var_ratio'][1]*100:.2f}%, "
            f"PC3={spring_pca_json['explained_var_ratio'][2]*100:.2f}%")
    
    # Summary
    print("\n" + "="*80)
    print("REFERENCE MODEL SUMMARY")
    print("="*80)
    print(f"Number of empirical hauls: {len(valid_hauls)}")
    print(f"Number of SNPs: {G.shape[1]}")
    print(f"PCA components: 3")
    print(f"Reference PCA variance: {pca_model.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"Spring PCA variance: {np.sum(spring_pca['explained_var_ratio'][:3])*100:.2f}%")
    print(f"Model files:")
    print(f"  - {model_path}")
    print(f"  - {spring_path}")
    print("="*80)
    
    return ref_model


if __name__ == "__main__":
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.abspath("../Pipeline version 1")
    vcf_file = os.path.join(sim_dir, "Bioinformatics_Course_2025_Herring_Sample_Subset.vcf")
    metadata_file = os.path.join(sim_dir, "All_Sample_Metadata.txt")
    qc_file = os.path.join(script_dir, "qc_passed_individuals.txt")
    
    if not os.path.exists(vcf_file):
        print(f"Error: VCF file not found: {vcf_file}")
        sys.exit(1)
    
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found: {metadata_file}")
        sys.exit(1)
    
    # Change working directory to simulations so genotype_matrix finds metadata
    orig_dir = os.getcwd()
    os.chdir(sim_dir)
    
    try:
        build_reference_model_from_empirical(vcf_file, metadata_file, 
                                            output_dir=os.path.join(orig_dir, "reference_model"),
                                            qc_passed_file=qc_file)
    finally:
        os.chdir(orig_dir)
