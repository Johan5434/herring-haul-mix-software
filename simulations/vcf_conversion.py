# simulations/vcf_conversion.py

import os


def load_sample_to_pop(metadata_filename):
    """
    Read the metadata file and return sample_id -> population.

    Based on your earlier assumption:
        sample_id = col[0]
        haul      = col[2]
        pop       = col[5]
    """
    sample_to_pop = {}

    with open(metadata_filename, "r") as meta:
        header = next(meta)  # skip header
        for line in meta:
            cols = line.strip().split("\t")
            if len(cols) < 6:
                continue
            sample_id = cols[0]
            pop = cols[5]
            sample_to_pop[sample_id] = pop

    return sample_to_pop


def split_raw_vcf_by_population(
    raw_vcf,
    metadata_filename="All_Sample_Metadata.txt",
    out_dir=".",
    apply_callrate_filter=True,
    max_missing=0.1,
    apply_skip_unplaced=True,
    skip_prefix="unplaced_scaffold",
):
    """
    From one *large* raw VCF + metadata:
      - (optionally) filter out variants with too high missingness (1 - call rate)
      - (optionally) skip chromosomes that start with 'unplaced_scaffold'
      - write one VCF per population.

    Returns a dict: {population: population_vcf_path}
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    sample_to_pop = load_sample_to_pop(metadata_filename)

    pop_to_indices = {}      # pop -> list of indices in the VCF sample part
    out_files = {}           # pop -> file object
    pop_to_vcfpath = {}      # pop -> filename
    header_lines = []

    with open(raw_vcf, "r") as fin:
        # 1) read header, open population-specific files
        for line in fin:
            if line.startswith("##"):
                header_lines.append(line)
                continue

            if line.startswith("#CHROM"):
                header_lines.append(line)
                cols = line.rstrip("\n").split("\t")
                fixed_hdr = cols[:9]
                samples_hdr = cols[9:]

                for i, sid in enumerate(samples_hdr):
                    if sid not in sample_to_pop:
                        continue
                    pop = sample_to_pop[sid]
                    pop_to_indices.setdefault(pop, []).append(i)

                # open output files and write header
                for pop, indices in pop_to_indices.items():
                    out_vcf = os.path.join(out_dir, f"{pop}.vcf")
                    pop_to_vcfpath[pop] = out_vcf
                    f = open(out_vcf, "w")
                    out_files[pop] = f

                    for hline in header_lines:
                        if hline.startswith("#CHROM"):
                            hcols = hline.rstrip("\n").split("\t")
                            fh = hcols[:9]
                            sh = hcols[9:]
                            new_samples = [sh[j] for j in indices]
                            f.write("\t".join(fh + new_samples) + "\n")
                        else:
                            f.write(hline)
                break

        # 2) variant lines
        for line in fin:
            if line.startswith("#"):
                continue

            cols = line.rstrip("\n").split("\t")
            chrom = cols[0]

            if apply_skip_unplaced and chrom.startswith(skip_prefix):
                continue

            fixed = cols[:9]
            all_samples = cols[9:]
            num_samples = len(all_samples)

            # call rate / missingness
            if apply_callrate_filter:
                missing = 0
                for raw_gt in all_samples:
                    gt = raw_gt.split(":")[0]
                    if gt == "./.":
                        missing += 1
                missing_frac = missing / num_samples
                if missing_frac > max_missing:
                    continue

            # write to all population files
            for pop, indices in pop_to_indices.items():
                f = out_files[pop]
                pop_samples = [all_samples[j] for j in indices]
                f.write("\t".join(fixed + pop_samples) + "\n")

    for f in out_files.values():
        f.close()

    print("Done. Created population-specific VCF files:")
    for pop, path in pop_to_vcfpath.items():
        print(f"  {pop}: {path}")

    return pop_to_vcfpath
