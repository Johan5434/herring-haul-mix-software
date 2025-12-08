# simulations/qc_core.py

import numpy as np

from data_loading import genotype_matrix
from qc_metrics import build_table, filter_to_box_mid


def iqr_outlier_mask(values: np.ndarray, whisker: float = 1.5) -> np.ndarray:
    """
    Return a boolean mask where True = OUTLIER
    according to the whisker rule: outside [Q1 - whisker*IQR, Q3 + whisker*IQR].
    """
    values = np.asarray(values, dtype=float)
    Q1 = np.nanpercentile(values, 25)
    Q3 = np.nanpercentile(values, 75)
    IQR = Q3 - Q1
    lower = Q1 - whisker * IQR
    upper = Q3 + whisker * IQR
    outlier = (values < lower) | (values > upper)
    return outlier


def qc_filter_single_vcf(
    vcf_filename: str,
    filter_het: bool,
    filter_f: bool,
    filter_ibs: bool,
    filter_combined: bool,
    whisker: float = 1.5,
):
    """
    Run QC filtering on ONE population-specific VCF.

    - Loads G, M via genotype_matrix
    - Builds T via build_table
    - Applies IQR filters according to flags:
        * filter_het       → IQR on heterozygosity (column 3 in T)
        * filter_f         → IQR on F-score (column 4 in T)
        * filter_ibs       → IQR on IBS (column 5 in T)
        * filter_combined  → remove individuals that are outliers in HET, F AND IBS simultaneously

    Returns:
        G_filt, M_filt, T_filt, kept_ids
    """
    print(f"\n  -> QC on {vcf_filename}...")
    G, M = genotype_matrix(vcf_filename)
    T = build_table(G, M)  # columns: 0:id,1:haul,2:pop,3:H_obs,4:F,5:IBS

    # 1) IQR filter on HET
    if filter_het:
        print("      IQR filter on heterozygosity (H_obs)...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=3, whisker=whisker)

    # 2) IQR filter on F
    if filter_f:
        print("      IQR filter on F-score...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=4, whisker=whisker)

    # 3) IQR filter on IBS
    if filter_ibs:
        print("      IQR filter on IBS-score...")
        G, M, T = filter_to_box_mid(T, G, M, col_index=5, whisker=whisker)

    # 4) combined outlier: must be outlier in HET, F AND IBS simultaneously
    if filter_combined:
        print("      Combined outlier filter (HET, F, IBS)...")
        # build QC table on the current G, M (after any previous filters)
        T_comb = build_table(G, M)
        het_vals = T_comb[:, 3].astype(float)
        f_vals = T_comb[:, 4].astype(float)
        ibs_vals = T_comb[:, 5].astype(float)

        out_het = iqr_outlier_mask(het_vals, whisker=whisker)
        out_f = iqr_outlier_mask(f_vals, whisker=whisker)
        out_ibs = iqr_outlier_mask(ibs_vals, whisker=whisker)

        combined_outliers = out_het & out_f & out_ibs
        keep = ~combined_outliers

        print(f"        -> Removing {combined_outliers.sum()} individuals as combined outliers.")

        G = G[keep]
        M = M[keep]
        T = T_comb[keep]

    kept_ids = M[:, 0]
    print(f"      Keeping {G.shape[0]} individuals after QC.")

    return G, M, T, kept_ids


def run_qc_on_pop_vcfs_with_flags(
    pop_vcfs,
    filter_het: bool,
    filter_f: bool,
    filter_ibs: bool,
    filter_combined: bool,
    whisker: float = 1.5,
):
    """
    Run QC on a list of population VCF files with the given filter flags (no interaction).

    Returns:
      results[vcf_filename] = {
          "G": G_filt,
          "M": M_filt,
          "T": T_filt,
          "kept_ids": kept_ids,
      }
    """
    if not pop_vcfs:
        print("No VCF file provided to the QC step.")
        return {}

    print("\n== STEP 3: Running QC on population-specific VCF files ==")
    results = {}

    for vcf in pop_vcfs:
        G_f, M_f, T_f, kept_ids = qc_filter_single_vcf(
            vcf_filename=vcf,
            filter_het=filter_het,
            filter_f=filter_f,
            filter_ibs=filter_ibs,
            filter_combined=filter_combined,
            whisker=whisker,
        )

        results[vcf] = {
            "G": G_f,
            "M": M_f,
            "T": T_f,
            "kept_ids": kept_ids,
        }

    return results
