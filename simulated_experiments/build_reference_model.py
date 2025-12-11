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

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../simulations"))

from data_loading import genotype_matrix
from pca_hauls import project_individuals_to_PCs


def build_reference_model_from_empirical(vcf_file, metadata_file, output_dir="reference_model"):
    """
    Build reference PCA model from empirical hauls.
    
    Args:
        vcf_file: Path to VCF file with original data
        metadata_file: Path to metadata file with haul assignments
        output_dir: Where to save the model
    """
    
    print("="*80)
    print("BUILDING REFERENCE PCA MODEL FROM EMPIRICAL HAULS")
    print("="*80)
    
    # Load data
    print("\n1. Loading empirical data...")
    G_tuple = genotype_matrix(vcf_file)
    # genotype_matrix returns (genotypes, sample_ids)
    if isinstance(G_tuple, tuple):
        G, sample_ids = G_tuple
    else:
        G = G_tuple
    
    metadata = pd.read_csv(metadata_file, sep="\t")
    
    print(f"   ✓ Loaded genotypes: {G.shape[0]} samples × {G.shape[1]} SNPs")
    print(f"   ✓ Loaded metadata: {len(metadata)} samples")
    
    # Build PCA from all individuals
    print("\n2. Building PCA from all individuals...")
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
    print("\n3. Calculating empirical haul centroids...")
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
    print("\n4. Preparing model data...")
    
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
    }
    
    # Save model
    print("\n5. Saving reference model...")
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "reference_pca.json")
    with open(model_path, 'w') as f:
        json.dump(ref_model, f, indent=2)
    
    print(f"   ✓ Saved to {model_path}")
    
    # Summary
    print("\n" + "="*80)
    print("REFERENCE MODEL SUMMARY")
    print("="*80)
    print(f"Number of empirical hauls: {len(valid_hauls)}")
    print(f"Number of SNPs: {G.shape[1]}")
    print(f"PCA components: 3")
    print(f"Total explained variance: {pca_model.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"Model file: {model_path}")
    print("="*80)
    
    return ref_model


if __name__ == "__main__":
    # Change to simulations directory so genotype_matrix can find metadata
    sim_dir = os.path.abspath("../simulations")
    vcf_file = os.path.join(sim_dir, "Bioinformatics_Course_2025_Herring_Sample_Subset.vcf")
    metadata_file = os.path.join(sim_dir, "All_Sample_Metadata.txt")
    
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
                                            output_dir=os.path.join(orig_dir, "reference_model"))
    finally:
        os.chdir(orig_dir)
