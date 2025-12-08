# simulations/vcf_utils.py

def write_filtered_vcf(vcf_in, vcf_out, kept_sample_ids):
    """
    Write a new VCF file containing only the individuals whose sample_id
    appears in kept_sample_ids.

    Parameters
    ----------
    vcf_in : str
        Input VCF (with all individuals).

    vcf_out : str
        Output VCF (only the retained individuals).

    kept_sample_ids : sequence of str
        List/array of sample_ids to keep.
    """
    kept_set = set(kept_sample_ids)
    keep_indices = None  # will be filled when we encounter the #CHROM header

    with open(vcf_in, "r") as fin, open(vcf_out, "w") as fout:
        for line in fin:
            # Copy meta-header lines as-is
            if line.startswith("##"):
                fout.write(line)
                continue

            # The header line containing sample columns
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                fixed = cols[:9]
                samples = cols[9:]

                # Determine which sample columns to keep
                keep_indices = [
                    i for i, sid in enumerate(samples)
                    if sid in kept_set
                ]

                # Write a new header with only the retained samples
                new_samples = [samples[i] for i in keep_indices]
                fout.write("\t".join(fixed + new_samples) + "\n")
                continue

            # Normal variant line
            cols = line.rstrip("\n").split("\t")
            fixed = cols[:9]
            samples = cols[9:]

            # Extract only the desired sample columns
            new_samples = [samples[i] for i in keep_indices]
            fout.write("\t".join(fixed + new_samples) + "\n")
