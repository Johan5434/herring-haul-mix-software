# simulations/make_simulations.py

import numpy as np


def run_simulations(
    G,
    M,
    n_individuals,
    sim_filename="simulated_hauls.txt",
    gen_filename="genomic_data.txt",
    prop_filename="proportions.txt",
):
    """
    Run the full simulation step based on genotype and metadata matrices in memory.

    Parameters
    ----------
    G : np.ndarray
        Genotype matrix with shape (n_samples, n_snps),
        values {0,1,2,-1}.

    M : np.ndarray
        Metadata matrix with shape (n_samples, 3):
        column 0: sample_id
        column 1: haul
        column 2: population (Autumn/North/Central/South)

    n_individuals : int
        Number of individuals per simulated haul.

    sim_filename : str
        Filename for the text file describing which individuals are included in each simulated haul.

    gen_filename : str
        Filename for the text file with genotype data per individual per simulated haul.

    prop_filename : str
        Filename for the text file with mix proportions per haul.

    Output
    ------
    Writes three text files to disk:
        - sim_filename
        - gen_filename
        - prop_filename

    Returns nothing, but the files can be used in the next step of the pipeline.
    """

    np.random.seed(42)

    # population labels from metadata
    pops = M[:, 2]
    pop_names = ["Autumn", "North", "Central", "South"]

    # indices of individuals per population
    pop_to_idx = {
        pop: np.where(pops == pop)[0]
        for pop in pop_names
    }

    def counts_from_steps(steps, n_individuals_inner):
        """
        steps: array-like with integers in [0,10] for [A, N, C, S],
               where the sum = 10 (i.e. 100%).
        Returns integer counts of individuals per population that sum to n_individuals_inner.
        """
        steps_arr = np.array(steps, dtype=float)        # e.g. [3, 7, 0, 0]
        fractions = steps_arr / 10.0                    # e.g. [0.3, 0.7, 0, 0]
        raw_counts = fractions * n_individuals_inner    # "ideal" counts
        counts = np.floor(raw_counts).astype(int)       # floor
        rest = int(n_individuals_inner - counts.sum())  # how many individuals missing due to flooring

        if rest > 0:
            # distribute remaining individuals to the largest fractional parts
            frac_parts = raw_counts - counts
            order = np.argsort(-frac_parts)            # sort descending
            for k in order:
                if rest == 0:
                    break
                if steps_arr[k] > 0:                   # only add extra to populations that should be present
                    counts[k] += 1
                    rest -= 1

        return counts  # always sums to n_individuals_inner

    with open(sim_filename, "w") as sim, \
         open(gen_filename, "w") as gen, \
         open(prop_filename, "w") as prop:

        haul_counter = 0

        # Iterate over ALL combinations (a, n, c, s) in integer steps that sum to 10
        for a in range(0, 11):                 # 0..10
            for n in range(0, 11 - a):         # 0..(10-a)
                for c in range(0, 11 - a - n): # 0..(10-a-n)
                    s = 10 - a - n - c         # determined by the others, >= 0

                    steps = {
                        "Autumn":  a,
                        "North":   n,
                        "Central": c,
                        "South":   s,
                    }

                    step_vec = [
                        steps["Autumn"],
                        steps["North"],
                        steps["Central"],
                        steps["South"],
                    ]

                    # If you want to skip pure single-stock hauls:
                    # num_nonzero = sum(v > 0 for v in step_vec)
                    # if num_nonzero < 2:
                    #     continue

                    # Compute how many individuals per population
                    n_aut, n_nor, n_cen, n_sou = counts_from_steps(step_vec, n_individuals)

                    # Safety check: if a population should be present (step>0) but the dataset has 0 individuals -> skip
                    if (steps["Autumn"]  > 0 and pop_to_idx["Autumn"].size  == 0) or \
                       (steps["North"]   > 0 and pop_to_idx["North"].size   == 0) or \
                       (steps["Central"] > 0 and pop_to_idx["Central"].size == 0) or \
                       (steps["South"]   > 0 and pop_to_idx["South"].size   == 0):
                        continue

                    # Choose individuals from each population, WITH replacement
                    chosen_indices = []

                    if n_aut > 0:
                        chosen_indices.extend(
                            np.random.choice(pop_to_idx["Autumn"], size=n_aut, replace=True)
                        )

                    if n_nor > 0:
                        chosen_indices.extend(
                            np.random.choice(pop_to_idx["North"], size=n_nor, replace=True)
                        )

                    if n_cen > 0:
                        chosen_indices.extend(
                            np.random.choice(pop_to_idx["Central"], size=n_cen, replace=True)
                        )

                    if n_sou > 0:
                        chosen_indices.extend(
                            np.random.choice(pop_to_idx["South"], size=n_sou, replace=True)
                        )

                    chosen_indices = np.array(chosen_indices)

                    # safety check: we want exactly n_individuals
                    if chosen_indices.size != n_individuals:
                        raise RuntimeError(
                            f"Error: selected {chosen_indices.size} individuals instead of {n_individuals} "
                            f"for combination (A={a}, N={n}, C={c}, S={s})"
                        )

                    # --- Write to files ---

                    # proportions.txt
                    prop_line = (
                        f"haul {haul_counter}: "
                        f"{steps['Autumn']*10:.0f}% Autumn, "
                        f"{steps['North']*10:.0f}% North, "
                        f"{steps['Central']*10:.0f}% Central, "
                        f"{steps['South']*10:.0f}% South\n"
                    )
                    prop.write(prop_line)

                    # simulated_hauls.txt
                    sim.write(f"haul{haul_counter}\n")

                    # genomic_data.txt + sample ID list
                    for j in chosen_indices:
                        sample_id  = M[j][0]
                        haul_orig  = M[j][1]   # original (empirical) haul
                        population = M[j][2]

                        # write sample ID to sim file
                        sim.write(sample_id + "\n")

                        # genotype vector as comma-separated string
                        genotype_list_to_string = ",".join(map(str, G[j]))
                        gen_line = (
                            f"{sample_id},{haul_orig},{population},"
                            f"{genotype_list_to_string}\n"
                        )
                        gen.write(gen_line)

                    # blank line between hauls
                    sim.write("\n")
                    gen.write("\n")

                    haul_counter += 1

    print("Simulation finished.")
    print(f"  Wrote {haul_counter} hauls to {sim_filename}, {gen_filename}, {prop_filename}")


def run_simulations_from_qc_results(
    qc_results: dict,
    n_individuals: int,
    sim_filename="simulated_hauls.txt",
    gen_filename="genomic_data.txt",
    prop_filename="proportions.txt",
):
    """
    Take a dict with QC results (e.g. from run_qc_on_pop_vcfs)
    and run run_simulations on the combined dataset.

    qc_results:
        { vcf_name: {"G": G_filt, "M": M_filt, "T": T_filt, ...}, ... }

    n_individuals:
        number of individuals per simulated haul.

    Writes the same three files as run_simulations, but on ALL QC'ed individuals.
    """
    if not qc_results:
        raise ValueError("qc_results is empty – nothing to simulate on.")

    G_list = []
    M_list = []

    for vcf_name, data in qc_results.items():
        G_list.append(data["G"])
        M_list.append(data["M"])

    G_all = np.vstack(G_list)
    M_all = np.vstack(M_list)

    print("Total number of individuals after QC (all populations):", G_all.shape[0])
    print("Number of SNPs:", G_all.shape[1])

    run_simulations(
        G_all,
        M_all,
        n_individuals=n_individuals,
        sim_filename=sim_filename,
        gen_filename=gen_filename,
        prop_filename=prop_filename,
    )
