# simulations/main_simulation_motor.py

import os
import numpy as np

from vcf_conversion import split_raw_vcf_by_population
from data_loading import genotype_matrix
from qc_core import run_qc_on_pop_vcfs_with_flags
from vcf_utils import write_filtered_vcf

from make_simulations import run_simulations_from_qc_results
from pca_hauls import build_reference_pca_from_GM
from pca_plotting import plot_pca_2d, plot_pca_3d

from prompt import ask_yes_no

# ================== Block 1: choose VCF source ==================

def choose_vcf_source():
    """
    Ask whether to start from a raw VCF and split,
    or if the user wants to specify existing pop-VCF files.

    Returns:
      pop_vcfs: list of filenames of population-specific VCFs
    """
    print("== STEP 1: VCF source ==")
    use_raw = ask_yes_no(
        "Do you want to start from a RAW VCF file (yes) or existing pop-VCF files (no)?"
    )

    if use_raw:
        raw_vcf = input(
            "  Enter path to raw VCF "
            "(e.g. Bioinformatics_Course_2025_Herring_Sample_Subset.vcf): "
        ).strip()
        metadata = input(
            "  Enter path to metadata (default All_Sample_Metadata.txt): "
        ).strip() or "All_Sample_Metadata.txt"

        print("\n  Variant filter (SNP level) before split by population")
        use_callrate = ask_yes_no("    Remove variants with call rate < 90%?")
        use_unplaced = ask_yes_no("    Remove 'unplaced_scaffold' variants?")

        pop_to_vcf = split_raw_vcf_by_population(
            raw_vcf=raw_vcf,
            metadata_filename=metadata,
            out_dir=".",
            apply_callrate_filter=use_callrate,
            max_missing=0.1,
            apply_skip_unplaced=use_unplaced,
        )

        pop_vcfs = list(pop_to_vcf.values())
        print("\n  Population-specific VCF files created:")
        for pop, fn in pop_to_vcf.items():
            print(f"    {pop}: {fn}")
        return pop_vcfs

    else:
        print("\n  OK, we will use existing pop-VCF files.")
        print("  Enter filenames (e.g. 'Autumn.vcf North.vcf Central.vcf South.vcf')")
        vlist = input("  VCF files (separate by space): ").strip()
        pop_vcfs = [v for v in vlist.split() if v]
        if not pop_vcfs:
            print("  No VCF files provided.")
        return pop_vcfs


# ================== Block 2: choose QC filters ==================

def choose_qc_filters():
    """
    Ask which individual QC filters should be used (HET/F/IBS/combined).
    """
    print("\n== STEP 2: Individual QC (HET/F/IBS) ==")
    filter_het = ask_yes_no("  IQR filter on heterozygosity (H_obs)?")
    filter_f = ask_yes_no("  IQR filter on F-score (Hardy-Weinberg)?")
    filter_ibs = ask_yes_no("  IQR filter on IBS score?")
    filter_combined = ask_yes_no(
        "  Remove individuals that are outliers in HET, F AND IBS simultaneously?"
    )
    return filter_het, filter_f, filter_ibs, filter_combined


def run_qc_on_pop_vcfs(pop_vcfs, filter_het, filter_f, filter_ibs, filter_combined):
    """
    Uses qc_core.run_qc_on_pop_vcfs_with_flags to do the QC logic,
    then stacks G_all/M_all and extracts kept_ids per VCF.

    Returns:
      G_all, M_all, per_vcf_kept_ids
    """
    from qc_core import run_qc_on_pop_vcfs_with_flags  # local import for clarity

    qc_results = run_qc_on_pop_vcfs_with_flags(
        pop_vcfs,
        filter_het=filter_het,
        filter_f=filter_f,
        filter_ibs=filter_ibs,
        filter_combined=filter_combined,
    )

    if not qc_results:
        print("  After QC there are no individuals left in any file.")
        return None, None, {}

    G_list = []
    M_list = []
    per_vcf_kept_ids = {}

    for vcf, data in qc_results.items():
        G_f = data["G"]
        M_f = data["M"]
        kept_ids = data["kept_ids"]

        if G_f.shape[0] == 0:
            print(f"    Warning: no individuals left after QC in {vcf}. Skipping this file.")
            continue

        G_list.append(G_f)
        M_list.append(M_f)
        per_vcf_kept_ids[vcf] = kept_ids

    if not G_list:
        print("  After QC there are no individuals left in any file.")
        return None, None, {}

    G_all = np.vstack(G_list)
    M_all = np.vstack(M_list)

    print("\n  Summary after QC:")
    print("    Total number of individuals (all populations):", G_all.shape[0])
    print("    Number of SNPs:", G_all.shape[1])

    return G_all, M_all, per_vcf_kept_ids


# ================== Block 4: write cleaned VCF ==================

def maybe_write_cleaned_vcfs(per_vcf_kept_ids):
    """
    Ask whether to write *_cleaned.vcf per population, and do so if yes.
    """
    if not per_vcf_kept_ids:
        return

    save_vcfs = ask_yes_no(
        "\nDo you want to write new '*_cleaned.vcf' files with only the kept individuals?"
    )

    if not save_vcfs:
        print("  OK, no new VCF files will be written.")
        return

    print("\n  Writing cleaned VCF per population...")
    for vcf, kept_ids in per_vcf_kept_ids.items():
        base = os.path.splitext(os.path.basename(vcf))[0]
        if base.endswith("_cleaned"):
            base = base.replace("_cleaned", "")
        out_vcf = f"{base}_cleaned.vcf"
        write_filtered_vcf(vcf, out_vcf, kept_ids)
        print(f"    {vcf} -> {out_vcf}")


# ================== Block 4.5: save G_all / M_all ==================

def maybe_save_GM(G_all, M_all, default_path="GM_all_after_QC.npz"):
    """
    Ask whether G_all/M_all should be saved to npz for later experiments
    (main_pca_mixes_experiments.py)
    """
    if G_all is None or M_all is None:
        return

    if not ask_yes_no(f"\nDo you want to save G_all/M_all to '{default_path}'?"):
        return

    path = input(f"  Enter filename (empty for {default_path}): ").strip() or default_path
    np.savez(path, G_all=G_all, M_all=M_all)
    print(f"  Saved G_all/M_all to '{path}'.")


# ================== Block 5: large simulation text files ==================

def maybe_run_big_simulations(G_all, M_all):
    """
    Ask whether to generate the large simulation .txt files
    (simulated_hauls.txt, genomic_data.txt, proportions.txt)
    with all combinations (A, N, C, S summing to 10).

    If no: skip this block and continue.
    """
    if G_all is None or M_all is None:
        return

    do_sims = ask_yes_no(
        "\nDo you want to generate ALL mix combinations to text files "
        "(simulated_hauls.txt, genomic_data.txt, proportions.txt)?"
    )

    if not do_sims:
        print("  OK, skipping generation of large simulation files.")
        return

    n_individuals = int(
        input("  Enter number of individuals per simulated haul (e.g. 30): ").strip()
    )

    print("  Running run_simulations_from_qc_results on full G_all/M_all...")
    qc_results = {
        "ALL": {"G": G_all, "M": M_all, "T": None, "vcf_out": None}
    }

    run_simulations_from_qc_results(
        qc_results,
        n_individuals=n_individuals,
        sim_filename="simulated_hauls.txt",
        gen_filename="genomic_data.txt",
        prop_filename="proportions.txt",
    )


# ================== Block 6: reference PCA (individuals → haul centroids) ==================


def maybe_plot_reference_pca(G_all, M_all, n_components=3):
    """
    Build reference PCA (individual PCA → haul centroids)
    and ask whether a 2D/3D plot should be made.
    """
    if G_all is None or M_all is None:
        print("No data (G_all/M_all) available for PCA.")
        return None

    # Build reference map via pca_hauls.build_reference_pca_from_GM
    ref_pca = build_reference_pca_from_GM(G_all, M_all, n_components=n_components)

    # Prints corresponding to your old run_reference_pca
    var_ratio = ref_pca["explained_var_ratio"]
    print("\n== STEP 6: Individual PCA → haul centroids ==")
    print("  Explained variance (individual PCA):")
    for i, v in enumerate(var_ratio[:n_components]):
        print(f"    PC{i+1}: {v*100:.1f}%")

    centroids = ref_pca["centroids"]
    haul_ids = ref_pca["haul_ids"]
    haul_pops = ref_pca["haul_pops"]
    print("  Haul centroid matrix shape:", centroids.shape)

    # 2D plot?
    do_2d = ask_yes_no("\nDo you want to plot 2D PCA (PC1 vs PC2) for haul centroids?")
    if do_2d:
        plot_pca_2d(
            centroids,
            haul_ids,
            haul_pops,
            pc_x=1,
            pc_y=2,
            title="Reference PCA – haul centroids (PC1 vs PC2)",
            show=True,
            savepath="reference_pca_hauls_2d.png",
        )
        print("  Also saved 2D plot to 'reference_pca_hauls_2d.png'.")

    # 3D plot?
    if centroids.shape[1] >= 3:
        do_3d = ask_yes_no("Do you also want to plot 3D PCA (PC1, PC2, PC3) for haul centroids?")
        if do_3d:
            plot_pca_3d(
                centroids,
                haul_ids,
                haul_pops,
                title="Reference PCA – haul centroids (PC1, PC2, PC3)",
                show=True,
                savepath="reference_pca_hauls_3d.png",
            )
            print("  Also saved 3D plot to 'reference_pca_hauls_3d.png'.")
    else:
        print("Fewer than 3 PCA components – skipping 3D plot.")

    return ref_pca

# ================== Main pipeline ==================

def run_full_simulated_pipeline():
    """
    End-to-end controller for the SIMULATED part, with clear blocks:

      1) Choose VCF source (raw VCF + split, or existing pop-VCFs).
      2) Choose individual QC filters.
      3) Run QC per pop-VCF -> G_all, M_all.
      4) (optional) write *_cleaned.vcf.
      4.5) (optional) save G_all/M_all to npz.
      5) (optional) generate large simulation .txt files (all mix combos).
      6) Build reference PCA (individual PCA → haul centroids) + 2D/3D plot.
    """
    print("=== SIMULATED PIPELINE: from VCF to PCA/mixes ===\n")

    # 1) VCF source
    pop_vcfs = choose_vcf_source()
    if not pop_vcfs:
        print("No VCF source chosen. Aborting.")
        return

    # 2) QC filters
    filter_het, filter_f, filter_ibs, filter_combined = choose_qc_filters()

    # 3) QC per VCF -> G_all, M_all
    G_all, M_all, per_vcf_kept_ids = run_qc_on_pop_vcfs(
        pop_vcfs, filter_het, filter_f, filter_ibs, filter_combined
    )
    if G_all is None:
        print("No data left after QC. Aborting.")
        return

    # 4) (optional) write cleaned VCFs
    maybe_write_cleaned_vcfs(per_vcf_kept_ids)

    # 4.5) (optional) save G_all/M_all
    maybe_save_GM(G_all, M_all)

    # 5) (optional) large simulation .txt files
    maybe_run_big_simulations(G_all, M_all)

    # 6) reference PCA + (optional) 2D/3D plot
    ref_pca = maybe_plot_reference_pca(G_all, M_all)

    print("\n=== SIMULATED PIPELINE DONE ===")
    return {
        "G_all": G_all,
        "M_all": M_all,
        "ref_pca": ref_pca,
    }


if __name__ == "__main__":
    run_full_simulated_pipeline()
