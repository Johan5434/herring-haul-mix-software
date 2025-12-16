#!/usr/bin/env python3
"""
Rebuild reference and Spring PCA using your friend's exact filtering:
- SNP/variant-level: call rate >= 90% (missing <= 10%), skip 'unplaced_scaffold' contigs
- Individual-level: combined IQR outlier removal (HET & F & IBS), whisker=1.5, per-population

Outputs:
- simulated_experiments/qc_passed_individuals.txt (combined across pops)
- simulated_experiments/reference_model/reference_pca.json
- simulated_experiments/reference_model/spring_pca.json
"""
import os
import sys
import json
import numpy as np

# Make simulations modules importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "simulations"))
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

from vcf_conversion import split_raw_vcf_by_population
from data_loading import genotype_matrix
from qc_core import run_qc_on_pop_vcfs_with_flags
from pca_hauls import build_reference_pca_from_GM, build_spring_pca_from_ref

RAW_VCF = os.path.join(SIM_DIR, "Bioinformatics_Course_2025_Herring_Sample_Subset.vcf")
METADATA = os.path.join(SIM_DIR, "All_Sample_Metadata.txt")
OUT_QC_IDS = os.path.join(THIS_DIR, "qc_passed_individuals.txt")
REF_DIR = os.path.join(THIS_DIR, "reference_model")
REF_JSON = os.path.join(REF_DIR, "reference_pca.json")
SPRING_JSON = os.path.join(REF_DIR, "spring_pca.json")

POPS = ["Autumn", "North", "Central", "South"]


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, dict, str, int, float, bool)) or obj is None:
        return obj
    try:
        return obj.tolist()  # fallback for array-likes
    except Exception:
        return obj


def dict_to_jsonable(d):
    return {k: to_jsonable(v) for k, v in d.items()}


def main():
    print("=" * 60)
    print("Rebuilding with friend's filters (call rate 90%, skip unplaced; combined IQR only)")
    print("=" * 60)

    # 1) Split raw VCF by population with variant-level filters
    print("\n[1/4] Creating population VCFs with call rate filter and unplaced skip...")
    pop_to_vcf = split_raw_vcf_by_population(
        raw_vcf=RAW_VCF,
        metadata_filename=METADATA,
        out_dir=SIM_DIR,
        apply_callrate_filter=True,
        max_missing=0.10,
        apply_skip_unplaced=True,
        skip_prefix="unplaced_scaffold",
    )

    # 2) Run per-population QC: combined outlier only
    print("\n[2/4] Running per-pop QC (combined IQR outliers only; whisker=1.5)...")
    pop_vcfs = [pop_to_vcf[p] for p in POPS if p in pop_to_vcf]
    # qc_core uses relative metadata path by default; run from SIM_DIR
    cwd = os.getcwd()
    try:
        os.chdir(SIM_DIR)
        results = run_qc_on_pop_vcfs_with_flags(
            pop_vcfs=pop_vcfs,
            filter_het=False,
            filter_f=False,
            filter_ibs=False,
            filter_combined=True,
            whisker=1.5,
        )
    finally:
        os.chdir(cwd)

    kept_ids_all = []
    per_pop_counts = {}
    for p in POPS:
        path = pop_to_vcf.get(p)
        if not path:
            continue
        kept = results.get(path, {}).get("kept_ids", [])
        kept_ids_all.extend(list(kept))
        per_pop_counts[p] = len(kept)

    kept_ids_all = list(dict.fromkeys(kept_ids_all))  # de-duplicate, preserve order
    print(f"  Kept individuals per pop: {per_pop_counts}")
    print(f"  Total kept individuals: {len(kept_ids_all)}")

    # Write QC-passed IDs for downstream tools
    with open(OUT_QC_IDS, "w") as f:
        for sid in kept_ids_all:
            f.write(f"{sid}\n")
    print(f"  ✓ Wrote kept IDs to {OUT_QC_IDS}")

    # 3) Build G_all/M_all from the filtered pop VCFs
    print("\n[3/4] Building G_all / M_all from filtered population VCFs...")
    G_blocks = []
    M_blocks = []
    for p in POPS:
        vcf = pop_to_vcf.get(p)
        if not vcf:
            continue
        Gp, Mp = genotype_matrix(vcf, metadata_filename=METADATA)
        G_blocks.append(Gp)
        M_blocks.append(Mp)

    if not G_blocks:
        raise RuntimeError("No population VCFs found to build G_all/M_all.")

    G_all = np.vstack(G_blocks)
    M_all = np.vstack(M_blocks)

    # Filter to kept IDs
    keep_mask = np.isin(M_all[:, 0], kept_ids_all)
    G_all = G_all[keep_mask]
    M_all = M_all[keep_mask]
    print(f"  Final matrices: G_all {G_all.shape}, M_all {M_all.shape}")

    # 4) Build and save reference + Spring PCAs
    print("\n[4/4] Building reference and Spring PCAs...")
    ref = build_reference_pca_from_GM(G_all, M_all, n_components=3)

    # summary
    var = ref["explained_var_ratio"]
    print(
        f"  Reference PCA variance: PC1={var[0]*100:.2f}%, PC2={var[1]*100:.2f}%, PC3={var[2]*100:.2f}%"
    )

    # Prepare JSON
    ensure_dir(REF_DIR)
    ref_json = dict_to_jsonable(ref)
    with open(REF_JSON, "w") as f:
        json.dump(ref_json, f, indent=2)
    print(f"  ✓ Saved reference PCA to {REF_JSON}")

    spring = build_spring_pca_from_ref(ref, n_components=3)
    svar = spring["explained_var_ratio"]
    print(
        f"  Spring PCA variance:    PC1={svar[0]*100:.2f}%, PC2={svar[1]*100:.2f}%, PC3={svar[2]*100:.2f}%"
    )

    spring_json = dict_to_jsonable(spring)
    with open(SPRING_JSON, "w") as f:
        json.dump(spring_json, f, indent=2)
    print(f"  ✓ Saved Spring PCA to {SPRING_JSON}")

    print("\nDone.")


if __name__ == "__main__":
    main()
