#!/usr/bin/env python3
"""
qc_before_simulation.py

Run QC filters on the original data BEFORE creating simulated hauls.
This ensures simulated hauls use only QC-passed individuals.

Usage:
  python qc_before_simulation.py
  
This will:
1. Load the original VCF + metadata
2. Apply SNP missingness filter (≤5% missing)
3. Apply individual missingness filter (≤10% missing, i.e., ≥90% call rate)
4. Apply individual QC filters (heterozygosity, F-score, IBS, combined outliers)
5. Output a list of QC-passed sample IDs
6. Then run generate_simulated_hauls.py to create hauls from these individuals

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


def filter_snps_by_missingness(G, max_snp_missing=0.05):
    """
    Remove SNPs with missingness > max_snp_missing.
    
    Parameters:
      G: Genotype matrix (individuals × SNPs), -1 = missing
      max_snp_missing: Maximum allowed missing fraction per SNP (default 0.05 = 5%)
      
    Returns:
      G_filtered: Genotype matrix with only SNPs passing filter
      kept_snp_mask: Boolean mask of SNPs that passed
    """
    n_individuals = G.shape[0]
    missing_per_snp = np.sum(G == -1, axis=0) / n_individuals
    kept_snp_mask = missing_per_snp <= max_snp_missing
    
    n_removed = np.sum(~kept_snp_mask)
    print(f"    SNPs with >{max_snp_missing*100:.0f}% missing: {n_removed}")
    print(f"    SNPs kept: {np.sum(kept_snp_mask)} / {G.shape[1]}")
    
    return G[:, kept_snp_mask], kept_snp_mask


def filter_individuals_by_missingness(G, M, max_ind_missing=0.10):
    """
    Remove individuals with missingness > max_ind_missing.
    
    Parameters:
      G: Genotype matrix (individuals × SNPs), -1 = missing
      M: Metadata matrix
      max_ind_missing: Maximum allowed missing fraction per individual (default 0.10 = 10%)
      
    Returns:
      G_filtered: Genotype matrix with only individuals passing filter
      M_filtered: Metadata with only individuals passing filter
      kept_ind_mask: Boolean mask of individuals that passed
    """
    n_snps = G.shape[1]
    missing_per_ind = np.sum(G == -1, axis=1) / n_snps
    kept_ind_mask = missing_per_ind <= max_ind_missing
    
    n_removed = np.sum(~kept_ind_mask)
    print(f"    Individuals with >{max_ind_missing*100:.0f}% missing: {n_removed}")
    print(f"    Individuals kept: {np.sum(kept_ind_mask)} / {G.shape[0]}")
    
    return G[kept_ind_mask], M[kept_ind_mask], kept_ind_mask


def qc_single_vcf(
    vcf_filename,
    metadata_filename="../simulations/All_Sample_Metadata.txt",
    filter_snp_missing=True,
    max_snp_missing=0.05,
    filter_ind_missing=True,
    max_ind_missing=0.10,
    filter_het=True,
    filter_f=True,
    filter_ibs=True,
    filter_combined=True,
    whisker=1.5,
):
    """
    Run QC on a single VCF and return list of QC-passed sample IDs.
    
    QC Steps (in order):
      1. SNP missingness filter (remove SNPs with >5% missing)
      2. Individual missingness filter (remove individuals with >10% missing)
      3. Heterozygosity outlier filter (IQR-based)
      4. F-score outlier filter (IQR-based)
      5. IBS-score outlier filter (IQR-based)
      6. Combined outlier filter (HET AND F AND IBS)
    
    Parameters:
      vcf_filename: Path to VCF file
      metadata_filename: Path to metadata file
      filter_snp_missing: Apply SNP missingness filter (default True)
      max_snp_missing: Maximum missing fraction per SNP (default 0.05 = 5%)
      filter_ind_missing: Apply individual missingness filter (default True)
      max_ind_missing: Maximum missing fraction per individual (default 0.10 = 10%)
      filter_het, filter_f, filter_ibs, filter_combined: Individual QC flags
      whisker: IQR whisker for outlier detection (1.5 = standard)
      
    Returns:
      kept_ids: List of sample IDs that pass QC
    """
    print(f"\nRunning QC on {vcf_filename}...")
    
    # Load genotypes and metadata
    G, M = genotype_matrix(vcf_filename, metadata_filename)
    print(f"  Loaded {G.shape[0]} individuals, {G.shape[1]} SNPs")
    
    n_original_ind = G.shape[0]
    n_original_snps = G.shape[1]
    
    # STEP 1: Filter SNPs by missingness
    if filter_snp_missing:
        print(f"\n  STEP 1: Filtering SNPs by missingness (max {max_snp_missing*100:.0f}%)...")
        G, kept_snp_mask = filter_snps_by_missingness(G, max_snp_missing=max_snp_missing)
        print(f"    After SNP filter: {G.shape[0]} individuals, {G.shape[1]} SNPs")
    
    # STEP 2: Filter individuals by missingness (call rate)
    if filter_ind_missing:
        print(f"\n  STEP 2: Filtering individuals by missingness (max {max_ind_missing*100:.0f}%)...")
        G, M, kept_ind_mask = filter_individuals_by_missingness(G, M, max_ind_missing=max_ind_missing)
        print(f"    After individual missingness filter: {G.shape[0]} individuals, {G.shape[1]} SNPs")
    
    # Build QC table for individual outlier detection
    print(f"\n  STEP 3-6: Individual outlier detection...")
    T = build_table(G, M)
    print(f"  Computed QC metrics (HET, F, IBS)")
    
    # 3) IQR filter on HET
    if filter_het:
        print("  Applying IQR filter on heterozygosity (H_obs)...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=3, whisker=whisker)
        print(f"    After HET filter: {G.shape[0]} individuals remain")
    
    # 4) IQR filter on F
    if filter_f:
        print("  Applying IQR filter on F-score...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=4, whisker=whisker)
        print(f"    After F filter: {G.shape[0]} individuals remain")
    
    # 5) IQR filter on IBS
    if filter_ibs:
        print("  Applying IQR filter on IBS-score...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=5, whisker=whisker)
        print(f"    After IBS filter: {G.shape[0]} individuals remain")
    
    # 6) Combined outlier filter
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
    n_removed = n_original_ind - len(kept_ids)
    
    print(f"\n  {'='*60}")
    print(f"  SUMMARY:")
    print(f"  {'='*60}")
    print(f"    Original SNPs:        {n_original_snps}")
    print(f"    SNPs after QC:        {G.shape[1]} ({G.shape[1]/n_original_snps*100:.1f}%)")
    print(f"    Original individuals: {n_original_ind}")
    print(f"    Individuals removed:  {n_removed} ({n_removed/n_original_ind*100:.1f}%)")
    print(f"    Individuals kept:     {len(kept_ids)} ({len(kept_ids)/n_original_ind*100:.1f}%)")
    print(f"  {'='*60}")
    
    return kept_ids


def main():
    """Main entry point."""
    original_vcf = "../simulations/Bioinformatics_Course_2025_Herring_Sample_Subset.vcf"
    metadata_file = "../simulations/All_Sample_Metadata.txt"
    output_file = "./qc_passed_individuals.txt"
    
    print("="*70)
    print("QC BEFORE SIMULATION - UPDATED PIPELINE")
    print("="*70)
    print("\nQC Steps:")
    print("  1. SNP missingness ≤5%")
    print("  2. Individual missingness ≤10% (≥90% call rate)")
    print("  3. Heterozygosity outliers (IQR whisker=1.5)")
    print("  4. F-score outliers (IQR whisker=1.5)")
    print("  5. IBS-score outliers (IQR whisker=1.5)")
    print("  6. Combined outliers (HET AND F AND IBS)")
    print("="*70)
    
    # Run QC with new missingness filters + standard individual filters
    kept_ids = qc_single_vcf(
        vcf_filename=original_vcf,
        metadata_filename=metadata_file,
        filter_snp_missing=True,
        max_snp_missing=0.05,      # ≤5% missing per SNP
        filter_ind_missing=True,
        max_ind_missing=0.10,      # ≤10% missing per individual (≥90% call rate)
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
    print(f"1. Run: python build_reference_model.py")
    print(f"   (Uses QC-passed individuals for reference PCA)")
    print(f"2. Run: python generate_simulated_hauls.py")
    print(f"   (Uses QC-passed individuals for simulations)")
    print(f"3. Run: python test_simulated_hauls.py --hauls all")
    print(f"   (Test classification on simulated hauls)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
