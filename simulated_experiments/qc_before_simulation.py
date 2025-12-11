#!/usr/bin/env python3
"""
qc_before_simulation.py

Run QC filters on the original data BEFORE creating simulated hauls.
This ensures simulated hauls use only QC-passed individuals.

Usage:
  python qc_before_simulation.py
  
This will:
1. Load the original VCF + metadata
2. Apply QC filters (heterozygosity, F-score, IBS, combined outliers)
3. Output a list of QC-passed sample IDs
4. Then run generate_simulated_hauls.py to create hauls from these individuals

Note: This assumes the original VCF and metadata are in ../simulations/
"""

import numpy as np
import sys
import os

# Add simulations folder to path so we can import from pipeline
sys.path.insert(0, '../simulations')

from data_loading import genotype_matrix
from qc_metrics import build_table
from qc_core import iqr_outlier_mask, filter_to_box_mid


def qc_single_vcf(
    vcf_filename,
    metadata_filename="../simulations/All_Sample_Metadata.txt",
    filter_het=True,
    filter_f=True,
    filter_ibs=True,
    filter_combined=True,
    whisker=1.5,
):
    """
    Run QC on a single VCF and return list of QC-passed sample IDs.
    
    Parameters:
      vcf_filename: Path to VCF file
      metadata_filename: Path to metadata file
      filter_het, filter_f, filter_ibs, filter_combined: QC flags
      whisker: IQR whisker for outlier detection (1.5 = standard)
      
    Returns:
      kept_ids: List of sample IDs that pass QC
    """
    print(f"\nRunning QC on {vcf_filename}...")
    
    # Load genotypes and metadata
    G, M = genotype_matrix(vcf_filename, metadata_filename)
    print(f"  Loaded {G.shape[0]} individuals, {G.shape[1]} SNPs")
    
    # Build QC table
    T = build_table(G, M)
    print(f"  Computed QC metrics")
    
    n_original = G.shape[0]
    
    # 1) IQR filter on HET
    if filter_het:
        print("  Applying IQR filter on heterozygosity (H_obs)...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=3, whisker=whisker)
        print(f"    After HET filter: {G.shape[0]} individuals remain")
    
    # 2) IQR filter on F
    if filter_f:
        print("  Applying IQR filter on F-score...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=4, whisker=whisker)
        print(f"    After F filter: {G.shape[0]} individuals remain")
    
    # 3) IQR filter on IBS
    if filter_ibs:
        print("  Applying IQR filter on IBS-score...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=5, whisker=whisker)
        print(f"    After IBS filter: {G.shape[0]} individuals remain")
    
    # 4) Combined outlier filter
    if filter_combined:
        print("  Applying combined outlier filter (HET AND F AND IBS)...")
        T_comb = build_table(G, M)
        het_vals = T_comb[:, 3].astype(float)
        f_vals = T_comb[:, 4].astype(float)
        ibs_vals = T_comb[:, 5].astype(float)
        
        out_het = iqr_outlier_mask(het_vals, whisker=whisker)
        out_f = iqr_outlier_mask(f_vals, whisker=whisker)
        out_ibs = iqr_outlier_mask(ibs_vals, whisker=whisker)
        
        combined_outliers = out_het & out_f & out_ibs
        keep = ~combined_outliers
        
        G = G[keep]
        M = M[keep]
        print(f"    After combined filter: {G.shape[0]} individuals remain")
    
    # Extract kept sample IDs
    kept_ids = M[:, 0]
    n_removed = n_original - len(kept_ids)
    
    print(f"\n  Summary:")
    print(f"    Original: {n_original} individuals")
    print(f"    Removed:  {n_removed} individuals ({n_removed/n_original*100:.1f}%)")
    print(f"    Kept:     {len(kept_ids)} individuals ({len(kept_ids)/n_original*100:.1f}%)")
    
    return kept_ids


def main():
    """Main entry point."""
    original_vcf = "../simulations/Bioinformatics_Course_2025_Herring_Sample_Subset.vcf"
    metadata_file = "../simulations/All_Sample_Metadata.txt"
    output_file = "./qc_passed_individuals.txt"
    
    print("="*70)
    print("QC BEFORE SIMULATION")
    print("="*70)
    
    # Run QC with standard filters (matching pipeline defaults)
    kept_ids = qc_single_vcf(
        vcf_filename=original_vcf,
        metadata_filename=metadata_file,
        filter_het=True,
        filter_f=True,
        filter_ibs=True,
        filter_combined=True,
        whisker=1.5,
    )
    
    # Save to file
    print(f"\nWriting QC-passed individuals to {output_file}...")
    with open(output_file, 'w') as f:
        for sample_id in kept_ids:
            f.write(f"{sample_id}\n")
    print(f"✓ Wrote {len(kept_ids)} sample IDs to {output_file}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print(f"1. Edit generate_simulated_hauls.py to use QC-passed individuals")
    print(f"2. Run: python generate_simulated_hauls.py")
    print(f"3. Run: python create_simulated_metadata.py")
    print(f"4. In pipeline: choose 'no' to all QC filters")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
