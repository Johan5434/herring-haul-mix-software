# simulations/qc_metrics.py

import numpy as np


def _compute_heterozygosity_and_F(G):
    """
    Compute observed heterozygosity (H_obs) and a simple F-coefficient per individual.

    Assumptions:
      - G: (n_individuals, n_snps) with values {0, 1, 2, -1}
        where -1 = missing.

    H_obs_i = (# heterozygous loci / # non-missing loci)
    H_exp per SNP = 2p(1-p), p = alt-allele frequency (computed over all individuals)
    H_exp_i = mean of H_exp over loci where the individual is not missing
    F_i ≈ 1 - H_obs_i / H_exp_i
    """
    G = np.asarray(G, dtype=int)
    n_ind, n_snps = G.shape

    # alt-allele frequencies per SNP (over all individuals)
    valid = (G != -1)
    allele_sum = np.where(valid, G, 0).sum(axis=0).astype(float)
    n_ind_per_snp = valid.sum(axis=0).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = allele_sum / (2.0 * n_ind_per_snp)  # alt-allele frequency
    p[np.isnan(p)] = 0.5  # neutral if completely missing

    H_exp_per_snp = 2.0 * p * (1.0 - p)

    H_obs = np.zeros(n_ind, dtype=float)
    F = np.zeros(n_ind, dtype=float)

    for i in range(n_ind):
        g_i = G[i, :]
        mask = (g_i != -1)
        n_loci = mask.sum()
        if n_loci == 0:
            H_obs[i] = np.nan
            F[i] = np.nan
            continue

        hetero = (g_i == 1) & mask
        H_obs_i = hetero.sum() / float(n_loci)

        H_exp_i = H_exp_per_snp[mask].mean()
        if H_exp_i <= 0:
            F_i = np.nan
        else:
            F_i = 1.0 - H_obs_i / H_exp_i

        H_obs[i] = H_obs_i
        F[i] = F_i

    return H_obs, F


def _compute_IBS_per_individual(G, M):
    """
    Compute a simple IBS-score per individual.

    Definition here:
      - For each haul (M[:,1]) each individual is compared with ALL other individuals
        in the same haul.
      - For each pair (i, j) and each SNP where both are non-missing:
            diff = |g_i - g_j|
            similarity per SNP = 1 - diff/2  (1, 0.5 or 0)
        IBS_i is the mean "similarity" over all pairs (i, j) in the same haul.

    Returns:
      IBS: np.ndarray with shape (n_individuals,)
    """
    G = np.asarray(G, dtype=int)
    n_ind = G.shape[0]

    hauls = M[:, 1]
    IBS = np.full(n_ind, np.nan, dtype=float)

    # group individuals per haul
    haul_to_indices = {}
    for idx, h in enumerate(hauls):
        haul_to_indices.setdefault(h, []).append(idx)

    for h, idx_list in haul_to_indices.items():
        idx_arr = np.array(idx_list, dtype=int)
        G_h = G[idx_arr, :]  # (n_h, n_snps)
        n_h = G_h.shape[0]
        if n_h < 2:
            # IBS is not defined with only one individual
            IBS[idx_arr] = np.nan
            continue

        # For each individual in the haul:
        for local_i, global_i in enumerate(idx_arr):
            g_i = G_h[local_i, :]
            sim_sum = 0.0
            denom = 0

            for local_j in range(n_h):
                if local_j == local_i:
                    continue
                g_j = G_h[local_j, :]
                both_valid = (g_i != -1) & (g_j != -1)
                if not np.any(both_valid):
                    continue

                diff = np.abs(g_i[both_valid] - g_j[both_valid])
                sim = 1.0 - diff / 2.0  # 1, 0.5 or 0
                sim_mean = sim.mean()
                sim_sum += sim_mean
                denom += 1

            if denom > 0:
                IBS[global_i] = sim_sum / denom
            else:
                IBS[global_i] = np.nan

    return IBS


def build_table(G, M):
    """
    Build a QC table per individual.

    Input:
      G : np.ndarray, shape (n_individuals, n_snps), values {0,1,2,-1}
      M : np.ndarray, shape (n_individuals, 3) = [sample_id, haul, population]

    Returns:
      T : np.ndarray, shape (n_individuals, 6) with columns:
          0: sample_id (str)
          1: haul      (str)
          2: population(str)
          3: H_obs     (float)
          4: F         (float)
          5: IBS       (float)
    """
    G = np.asarray(G)
    M = np.asarray(M)
    n_ind = G.shape[0]

    sample_ids = M[:, 0]
    hauls = M[:, 1]
    pops = M[:, 2]

    # H_obs and F
    H_obs, F = _compute_heterozygosity_and_F(G)

    # IBS
    IBS = _compute_IBS_per_individual(G, M)

    # Build the table
    T = np.empty((n_ind, 6), dtype=object)
    T[:, 0] = sample_ids
    T[:, 1] = hauls
    T[:, 2] = pops
    T[:, 3] = H_obs
    T[:, 4] = F
    T[:, 5] = IBS

    return T


def filter_to_box_mid(T, G, M, col_index, whisker=1.5):
    """
    IQR-based filter: keep individuals that lie INSIDE box+whiskers
    for the selected column in T.

    Inputs:
      T : np.ndarray (QC table from build_table)
      G : genotype matrix (n_ind, n_snps)
      M : metadata matrix (n_ind, 3)
      col_index : int
          Which column in T to use (e.g. 3=H_obs, 4=F, 5=IBS).
      whisker : float (default 1.5)

    Returns:
      G_filt, M_filt, T_filt
    """
    vals = T[:, col_index].astype(float)

    # Compute boxplot limits
    Q1 = np.nanpercentile(vals, 25)
    Q3 = np.nanpercentile(vals, 75)
    IQR = Q3 - Q1
    lower = Q1 - whisker * IQR
    upper = Q3 + whisker * IQR

    # NOTE: we keep those that lie INSIDE [lower, upper]
    keep = (vals >= lower) & (vals <= upper)

    n_total = len(vals)
    n_keep = keep.sum()
    n_drop = n_total - n_keep

    print(
        f"filter_to_box_mid(col_index={col_index}): "
        f"keeping {n_keep} / {n_total} individuals, dropping {n_drop}."
    )

    G_filt = G[keep]
    M_filt = M[keep]
    T_filt = T[keep]

    return G_filt, M_filt, T_filt
