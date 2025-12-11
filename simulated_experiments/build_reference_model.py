#!/usr/bin/env python3
"""
build_reference_model.py

DOCUMENTATION SCRIPT: Explains how the reference PCA model was built.

The reference model (in reference_model/) is pre-built and contains:
  - reference_pca.json: PCA components and haul centroids from original herring hauls
  - spring_pca.json: Spring-specific PCA model for N/C/S classification

WORKFLOW:
1. Load original genotype data from Bioinformatics_Course_2025_Herring_Sample_Subset.vcf
2. Compute PCA (3 components) on all individuals
3. Project all original hauls to PCA space
4. Compute centroids for each haul in PC space
5. Save PCA parameters and centroids to reference_pca.json
6. Build Spring-specific PCA from reference PCA
7. Save Spring PCA to spring_pca.json

The reference model is used by:
  - test_simulated_hauls.py: Load it and use centroids for classification
  - Haul distance-based classification: Compare simulated haul centroids to reference centroids
  
NOTE: This script is for reference/documentation. The actual reference model already exists
and is loaded by test_simulated_hauls.py. To rebuild it from scratch, you would need to
modify the imports and PCA building logic based on the existing test_simulated_hauls.py
implementation.
"""

import os
import json

def show_reference_model_info():
    """Display info about the existing reference model."""
    ref_dir = "./reference_model"
    
    print("=" * 70)
    print("REFERENCE PCA MODEL INFORMATION")
    print("=" * 70)
    
    # Check files exist
    pca_file = os.path.join(ref_dir, "reference_pca.json")
    spring_file = os.path.join(ref_dir, "spring_pca.json")
    
    if not os.path.exists(pca_file):
        print(f"\n✗ Reference PCA model not found at {pca_file}")
        print(f"  The model should be pre-built in the {ref_dir}/ directory")
        return
    
    print(f"\n✓ Found reference model files:")
    print(f"  {pca_file}")
    print(f"  {spring_file}")
    
    # Load and display info
    with open(pca_file, 'r') as f:
        ref_pca = json.load(f)
    
    print(f"\nReference PCA Statistics:")
    print(f"  Number of hauls: {len(ref_pca.get('haul_ids', []))}")
    print(f"  Number of SNPs: {len(ref_pca.get('valid_snps', []))}")
    
    if 'explained_var_ratio' in ref_pca:
        var_ratio = ref_pca['explained_var_ratio']
        print(f"\nExplained Variance Ratio:")
        print(f"  PC1: {var_ratio[0]*100:.2f}%")
        print(f"  PC2: {var_ratio[1]*100:.2f}%")
        print(f"  PC3: {var_ratio[2]*100:.2f}%")
        print(f"  Total: {sum(var_ratio)*100:.2f}%")
    
    if 'haul_ids' in ref_pca and 'haul_pops' in ref_pca:
        print(f"\nHaul Populations:")
        pop_counts = {}
        for pop in ref_pca['haul_pops']:
            pop_counts[pop] = pop_counts.get(pop, 0) + 1
        for pop in sorted(pop_counts.keys()):
            print(f"  {pop}: {pop_counts[pop]} hauls")
    
    if os.path.exists(spring_file):
        with open(spring_file, 'r') as f:
            spring_pca = json.load(f)
        
        print(f"\nSpring PCA Statistics:")
        if 'centroids' in spring_pca:
            print(f"  Spring population centroids: {len(spring_pca['centroids'])} populations")
    
    print("\n" + "=" * 70)
    print("USE IN test_simulated_hauls.py:")
    print("  load_reference_model() - loads and returns the reference PCA")
    print("  load_spring_pca() - loads the Spring-specific PCA")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    show_reference_model_info()


