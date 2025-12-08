# simulations/data_loading.py

import numpy as np


def load_metadata(metadata_filename="All_Sample_Metadata.txt"):
    """
    Read the metadata file and return a dict:
        sample_id -> (haul, population)

    Assumptions about columns in the metadata file (tab-separated):
        col[0] = sample_id
        col[2] = haul
        col[5] = population
    """
    sample_to_meta = {}

    with open(metadata_filename, "r") as meta:
        header = next(meta)  # skip header row
        for line in meta:
            cols = line.strip().split("\t")
            if len(cols) < 6:
                continue
            sample_id = cols[0]
            haul = cols[2]
            pop = cols[5]
            sample_to_meta[sample_id] = (haul, pop)

    return sample_to_meta


def _parse_genotype_field(gt_field):
    """
    Take a field from the VCF sample column, e.g. '0/1:35,12,0'
    and extract the genotype itself ('0/1').

    Returns an integer:
        0 = homozygous reference (0/0)
        1 = heterozygous        (0/1 or 1/0)
        2 = homozygous alt      (1/1)
       -1 = missing or other/unknown

    We do not care about quality etc.
    """
    if gt_field is None or gt_field == ".":
        return -1

    # Split on ":" first → '0/1', '...' etc
    parts = gt_field.split(":")
    gt = parts[0]

    if gt == "./.":
        return -1
    if gt == "0/0":
        return 0
    if gt in ("0/1", "1/0"):
        return 1
    if gt == "1/1":
        return 2

    # If there are other variants (e.g. more alleles) → mark as missing
    return -1


def genotype_matrix(
    vcf_filename,
    metadata_filename="All_Sample_Metadata.txt",
):
    """
    Read a (population-specific) VCF and build:

      G: genotype matrix (n_samples, n_snps) with values {0,1,2,-1}
      M: metadata matrix (n_samples, 3) [sample_id, haul, population]

    Assumptions:
      - VCF is in standard form:
            ## ...
            #CHROM  POS ID REF ALT QUAL FILTER INFO FORMAT  sample1 sample2 ...
            chr1   ...
      - sample IDs in the VCF header match sample_id in the metadata file.
      - metadata file columns:
            col[0] = sample_id
            col[2] = haul
            col[5] = population
    """
    # 1) Read metadata
    sample_to_meta = load_metadata(metadata_filename)

    # 2) First pass: find header (#CHROM line) and sample IDs
    sample_ids = None
    variant_lines = []

    with open(vcf_filename, "r") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                cols = line.strip().split("\t")
                sample_ids = cols[9:]
                break

        # rest of the file → variant lines
        for line in f:
            if not line or line.startswith("#"):
                continue
            variant_lines.append(line.strip())

    if sample_ids is None:
        raise RuntimeError(f"Did not find any #CHROM line in {vcf_filename}.")

    n_samples = len(sample_ids)
    n_snps = len(variant_lines)

    # 3) Initialize the genotype matrix
    # we build SNP-major (row = SNP), then transpose
    G_snp_major = np.full((n_snps, n_samples), -1, dtype=int)

    # 4) Fill genotypes
    for snp_idx, line in enumerate(variant_lines):
        cols = line.split("\t")
        # fixed fields 0..8, sample fields 9..etc
        sample_fields = cols[9:]
        if len(sample_fields) != n_samples:
            raise RuntimeError(
                f"On SNP row {snp_idx} in {vcf_filename} "
                f"we have {len(sample_fields)} sample fields, expected {n_samples}."
            )

        for i, field in enumerate(sample_fields):
            G_snp_major[snp_idx, i] = _parse_genotype_field(field)

    # 5) Transpose so that G is (n_samples, n_snps)
    G = G_snp_major.T  # shape: (n_samples, n_snps)

    # 6) Build metadata matrix M: [sample_id, haul, pop]
    M_rows = []
    for sid in sample_ids:
        if sid in sample_to_meta:
            haul, pop = sample_to_meta[sid]
        else:
            # If some sample is missing in metadata → mark
            haul, pop = "UNKNOWN", "UNKNOWN"
        M_rows.append([sid, haul, pop])

    M = np.array(M_rows, dtype=object)

    print(
        f"genotype_matrix: read {n_samples} individuals and {n_snps} SNPs from {vcf_filename}"
    )

    return G, M
