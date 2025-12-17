# simulations/qc_filtering.py

import os
import glob
import numpy as np

from qc_core import run_qc_on_pop_vcfs_with_flags
from vcf_utils import write_filtered_vcf
from prompt import ask_yes_no


def auto_find_pop_vcfs():
    """
    Simple helper for interactive runs:
    - looks for .vcf in the current directory that are NOT intermediate (e.g. big raw file)
      you can add your own logic/filename filters here if you want.
    Currently: all .vcf in the directory.
    """
    vcfs = sorted(glob.glob("*.vcf"))
    return vcfs


def run_qc_on_pop_vcfs(pop_vcf_paths):
    """
    Interactive wrapper around qc_core.run_qc_on_pop_vcfs_with_flags:

      - Takes a list of population-specific VCF files as input
      - Asks the user which QC filters should be applied
      - Runs QC filtering via qc_core (G, M, T) per file
      - Finally asks whether new VCF files should be written via write_filtered_vcf

    Returns a dict:
      result[vcf_in] = {
          "G": G_filt,
          "M": M_filt,
          "T": T_filt,
          "kept_ids": kept_ids,
          "vcf_out": vcf_out_path or None
      }
    """
    if not pop_vcf_paths:
        print("No VCF file provided to the QC step.")
        return {}

    print("=== QC on population-specific VCF files ===")
    print("Files that will be processed:")
    for v in pop_vcf_paths:
        print("  -", v)

    # Choice of QC filters (all individual-based, in memory)
    print("\n== Choice of individual QC (IQR-based filters) ==")
    filter_het = ask_yes_no("IQR filter on heterozygosity (H_obs)?")
    filter_f = ask_yes_no("IQR filter on F-score (Hardy-Weinberg)?")
    filter_ibs = ask_yes_no("IQR filter on IBS-score?")
    filter_combined = ask_yes_no(
        "Remove individuals that are outliers in HET, F AND IBS at the same time?"
    )

    # Call the QC core
    qc_results = run_qc_on_pop_vcfs_with_flags(
        pop_vcf_paths,
        filter_het=filter_het,
        filter_f=filter_f,
        filter_ibs=filter_ibs,
        filter_combined=filter_combined,
    )

    results = {}

    for vcf, data in qc_results.items():
        results[vcf] = {
            "G": data["G"],
            "M": data["M"],
            "T": data["T"],
            "kept_ids": data["kept_ids"],
            "vcf_out": None,  # filled if we choose to write new files
        }

    # Ask if we should write new *_cleaned.vcf files
    print("\n== Output options ==")
    save_vcfs = ask_yes_no(
        "Do you want to write new VCF files with only the kept individuals?"
    )

    if save_vcfs:
        print("\nWriting new cleaned VCF files...")
        for vcf, data in results.items():
            kept_ids = data["kept_ids"]
            base = os.path.splitext(os.path.basename(vcf))[0]
            # avoid _cleaned_cleaned
            if base.endswith("_cleaned"):
                base = base.replace("_cleaned", "")
            out_vcf = f"{base}_cleaned.vcf"
            write_filtered_vcf(vcf, out_vcf, kept_ids)
            print(f"  {vcf} -> {out_vcf}")
            data["vcf_out"] = out_vcf
    else:
        print("\nOK, no new VCF files will be written. The results exist only in memory.")

    print("\n=== QC step finished ===")
    return results


if __name__ == "__main__":
    print("=== Interactive QC step for population-specific VCFs ===")
    vcfs = auto_find_pop_vcfs()
    if not vcfs:
        print("Did not find any .vcf files in this directory.")
    else:
        print("Found the following .vcf files:")
        for fn in vcfs:
            print("  -", fn)

        use_all = ask_yes_no("Do you want to run QC on ALL these .vcf files?")
        if not use_all:
            # let the user type a list manually
            vlist = input(
                "Enter VCF files to QC (separate with spaces): "
            ).strip()
            chosen = [v for v in vlist.split() if v]
        else:
            chosen = vcfs

        _ = run_qc_on_pop_vcfs(chosen)
