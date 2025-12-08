# simulations/pca_hauls.py

import numpy as np


# ============================================================
#  PART 1: Allele frequency / haul-based helper functions
#  (kept as an alternative toolbox, but NOT used for the
#   main reference map anymore)
# ============================================================

# def project_AF_to_PCs(AF_new, mean_AF_ref, components_ref):
#     """
#     Project new allele-frequency vectors into a PCA space defined by
#     (mean_AF_ref, components_ref) from pca_from_AF.
#
#     Parameters
#     ----------
#     AF_new : np.ndarray
#         Allele-frequency matrix for the hauls you want to test,
#         shape (n_new, n_snps). May contain np.nan.
#
#     mean_AF_ref : np.ndarray
#         Mean allele frequency per SNP from the reference PCA,
#         shape (n_snps,).
#
#     components_ref : np.ndarray
#         PCA components from the reference, shape (n_components, n_snps).
#
#     Returns
#     -------
#     PC_new : np.ndarray
#         Projected coordinates, shape (n_new, n_components).
#     """
#     AF_new = np.asarray(AF_new, dtype=float)
#
#     # Impute np.nan in AF_new with the reference mean (mean_AF_ref)
#     AF_imputed = np.where(np.isnan(AF_new), mean_AF_ref, AF_new)
#
#     # Center using the reference mean
#     X_new = AF_imputed - mean_AF_ref
#
#     # Projection: X_new * V^T, where V = components_ref
#     PC_new = X_new @ components_ref.T
#
#     return PC_new

# ============================================================
#  REAL reference logic – individual PCA → haul centroids
# ============================================================

def individual_pca(G, n_components=3):
    """
    PCA at individual level, with -1 as missing, exactly as in your old logic.

    G: genotype matrix (individuals x SNPs), with -1 as missing.

    Returns:
      PCs_ind      – (n_ind, n_components), individual PC coordinates
      mean_snps    – column means for the SNPs that are used
      std_snps     – column std for the same SNPs
      var_ratio    – explained variance ratio for all components
      valid_snps   – bool array (length = original n_snps),
                     True = SNP used in PCA
      components   – PCA components (loadings), shape (n_components, n_valid_snps)
    """
    G = np.asarray(G, dtype=float).copy()
    n_ind, n_snps = G.shape

    # 1) Drop SNPs where ALL individuals are missing
    valid_snps = ~(np.all(G == -1, axis=0))
    G_work = G[:, valid_snps]

    # 2) Impute missing (-1) with column means
    missing = (G_work == -1)
    for j in range(G_work.shape[1]):
        col = G_work[:, j]
        mask = ~missing[:, j]
        if np.any(mask):
            m = col[mask].mean()
        else:
            m = 0.0
        col[~mask] = m
        G_work[:, j] = col

    # 3) Standardise per SNP
    mean_snps = G_work.mean(axis=0)
    std_snps = G_work.std(axis=0)
    std_snps[std_snps == 0] = 1.0
    G_std = (G_work - mean_snps) / std_snps

    # 4) PCA via SVD
    U, S, Vt = np.linalg.svd(G_std, full_matrices=False)
    k = min(n_components, S.shape[0])

    PCs_ind = U[:, :k] * S[:k]

    eigenvalues = (S ** 2) / (G_std.shape[0] - 1)
    var_ratio = eigenvalues / eigenvalues.sum()

    components = Vt[:k, :]  # (k, n_valid_snps)

    return PCs_ind, mean_snps, std_snps, var_ratio, valid_snps, components


def haul_centroids_from_PCs(PCs_ind, M):
    """
    Build haul centroids in PC space.

    PCs_ind: individual PCs (n_ind x n_components)
    M: metadata matrix (same order as G),
       where column 1 = haul, column 2 = population.

    Returns:
      centroids    – shape (n_hauls x n_components)
      unique_hauls – array with haul names
      haul_pops    – population per haul
    """
    PCs_ind = np.asarray(PCs_ind, dtype=float)
    hauls = M[:, 1]
    pops = M[:, 2]
    unique_hauls = np.unique(hauls)

    centroids = []
    haul_pops = []

    for h in unique_hauls:
        mask = (hauls == h)
        centroid = PCs_ind[mask].mean(axis=0)
        centroids.append(centroid)

        pop_sub = pops[mask]
        haul_pops.append(pop_sub[0] if pop_sub.size > 0 else "NA")

    return np.array(centroids), unique_hauls, np.array(haul_pops)


def project_individuals_to_PCs(G_new, mean_snps, std_snps, valid_snps, components):
    """
    Project NEW individuals into the same PC space as built by individual_pca.

    G_new: genotype matrix (n_new_ind x n_snps_original), with -1 as missing.
    mean_snps, std_snps, valid_snps, components: come directly from individual_pca.

    Returns:
      PCs_new: shape (n_new_ind, n_components)
    """
    G_new = np.asarray(G_new, dtype=float).copy()

    # Filter to the same SNPs that were used in the PCA training
    G_work = G_new[:, valid_snps]

    # Impute missing with training means (mean_snps)
    missing = (G_work == -1)
    for j in range(G_work.shape[1]):
        col = G_work[:, j]
        mask = ~missing[:, j]
        m = mean_snps[j]
        col[~mask] = m
        G_work[:, j] = col

    std_nonzero = std_snps.copy()
    std_nonzero[std_nonzero == 0] = 1.0
    G_std_new = (G_work - mean_snps) / std_nonzero

    PCs_new = G_std_new @ components.T
    return PCs_new


def build_reference_pca_from_GM(G_all, M_all, n_components=3):
    """
    Build reference PCA (individual PCA → haul centroids) from G_all / M_all.

    Parameters
    ----------
    G_all : np.ndarray
        Genotype matrix (n_ind, n_snps), with -1 as missing.

    M_all : np.ndarray
        Metadata matrix (n_ind, 3): [sample_id, haul, population].

    n_components : int
        Number of PCA components to compute (default 3).

    Returns
    -------
    ref : dict with keys:
      - "PCs_ind"             : individual PC coordinates, shape (n_ind, n_components)
      - "centroids"           : haul centroids in PC space, shape (n_hauls, n_components)
      - "haul_ids"            : array with haul names
      - "haul_pops"           : population per haul
      - "mean_snps"           : column means (for the SNPs that are used)
      - "std_snps"            : column std
      - "valid_snps"          : bool array (True = SNP used in PCA)
      - "components"          : PCA components (loadings), shape (n_components, n_valid_snps)
      - "explained_var_ratio" : explained variance ratio for all components
    """
    (
        PCs_ind,
        mean_snps,
        std_snps,
        var_ratio,
        valid_snps,
        components,
    ) = individual_pca(G_all, n_components=n_components)

    centroids, unique_hauls, haul_pops = haul_centroids_from_PCs(PCs_ind, M_all)

    ref = {
        "PCs_ind": PCs_ind,
        "centroids": centroids,
        "haul_ids": unique_hauls,
        "haul_pops": haul_pops,
        "mean_snps": mean_snps,
        "std_snps": std_snps,
        "valid_snps": valid_snps,
        "components": components,
        "explained_var_ratio": var_ratio,
        "_G_all": G_all,
        "_M_all": M_all,
    }
    return ref


def build_spring_pca_from_ref(ref_pca, n_components=3):
    """
    Build a Spring-only PCA (only North/Central/South) based on
    the same G_all/M_all that were used for the main reference.

    Returns a ref dict in the same style as build_reference_pca_from_GM,
    but only with Spring individuals.
    """
    G_all = ref_pca["_G_all"]
    M_all = ref_pca["_M_all"]

    pops = M_all[:, 2]
    spring_mask = (
        (pops == "North") |
        (pops == "Central") |
        (pops == "South")
    )

    G_spring = G_all[spring_mask]
    M_spring = M_all[spring_mask]

    # Build individual PCA on Spring individuals
    PCs_ind, mean_snps, std_snps, var_ratio, valid_snps, components = \
        individual_pca(G_spring, n_components=n_components)

    # Haul centroids in the Spring PCA space
    centroids, haul_ids, haul_pops = haul_centroids_from_PCs(PCs_ind, M_spring)

    spring_ref = {
        "PCs_ind": PCs_ind,
        "centroids": centroids,
        "haul_ids": haul_ids,
        "haul_pops": haul_pops,
        "mean_snps": mean_snps,
        "std_snps": std_snps,
        "valid_snps": valid_snps,
        "components": components,
        "explained_var_ratio": var_ratio,
    }
    return spring_ref
