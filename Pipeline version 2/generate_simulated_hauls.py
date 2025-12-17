#!/usr/bin/env python3
"""
generate_simulated_hauls.py

Generate simulated hauls with known population proportions.
Each haul contains 30 individuals sampled from specified populations.
Multiple versions (replicates) of each proportion are created with random sampling.

Outputs:
  - simulated_hauls_metadata.txt: sample_id\thaul_id (mapping of which fish are in which haul)
  - haul_proportions.txt: haul_id\tAutumn\tNorth\tCentral\tSouth (expected proportions)
  - simulated_individuals_list.txt: list of all sample_ids used in simulated hauls
"""

import numpy as np
import os


def load_metadata(metadata_path):
    """Load original metadata file and return sample_id -> population mapping."""
    sample_to_pop = {}
    with open(metadata_path, 'r') as f:
        header = f.readline()
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) < 6:
                continue
            sample_id = cols[0]
            population = cols[5]
            sample_to_pop[sample_id] = population
    return sample_to_pop


def _load_samples_from_population_vcfs(simulations_dir):
    """Fallback: Build sample->population map from population-specific VCFs.

    Expects VCF files named Autumn.vcf, North.vcf, Central.vcf, South.vcf in
    the provided simulations_dir. Returns a dict sample_id -> population.
    """
    pop_files = {
        "Autumn": os.path.join(simulations_dir, "Autumn.vcf"),
        "North": os.path.join(simulations_dir, "North.vcf"),
        "Central": os.path.join(simulations_dir, "Central.vcf"),
        "South": os.path.join(simulations_dir, "South.vcf"),
    }
    sample_to_pop = {}

    for pop, vcf_path in pop_files.items():
        if not os.path.exists(vcf_path):
            continue
        with open(vcf_path, 'r') as f:
            for line in f:
                if line.startswith('#CHROM'):
                    cols = line.strip().split('\t')
                    sample_ids = cols[9:]
                    for sid in sample_ids:
                        sample_to_pop[sid] = pop
                    break

    if not sample_to_pop:
        raise FileNotFoundError(
            "Could not find population VCFs to infer sample populations. "
            f"Checked in: {simulations_dir}"
        )

    return sample_to_pop


def get_individuals_per_population(sample_to_pop, qc_passed_filter=None):
    """
    Return a dict: population -> list of sample_ids.
    Populations: Autumn, North, Central, South.
    
    If qc_passed_filter is provided (list of sample IDs), only use those individuals.
    """
    qc_set = set(qc_passed_filter) if qc_passed_filter else None
    
    pop_to_samples = {
        "Autumn": [],
        "North": [],
        "Central": [],
        "South": []
    }
    for sample_id, pop in sample_to_pop.items():
        # Filter by QC-passed if provided
        if qc_set and sample_id not in qc_set:
            continue
        if pop in pop_to_samples:
            pop_to_samples[pop].append(sample_id)
    return pop_to_samples


def calculate_counts(perc_dict, n_individuals=30):
    """
    Given a percentage dict {pop: perc}, calculate actual counts that sum to n_individuals.
    
    Parameters:
      perc_dict: {"Autumn": 50, "North": 0, "Central": 0, "South": 50}
      n_individuals: 30
      
    Returns:
      {"Autumn": 15, "North": 0, "Central": 0, "South": 15}
    """
    pops = ["Autumn", "North", "Central", "South"]
    perc_arr = np.array([perc_dict.get(p, 0) for p in pops], dtype=float)
    fractions = perc_arr / 100.0
    raw_counts = fractions * n_individuals
    counts = np.floor(raw_counts).astype(int)
    rest = int(n_individuals - counts.sum())
    
    # Distribute remaining individuals to populations with largest fractional parts
    if rest > 0:
        frac_parts = raw_counts - counts
        order = np.argsort(-frac_parts)
        for k in order:
            if rest == 0:
                break
            if perc_arr[k] > 0:  # only add to populations that should be present
                counts[k] += 1
                rest -= 1
    
    return {pops[i]: counts[i] for i in range(len(pops))}


def generate_hauls(
    sample_to_pop,
    proportions_list,
    n_replicates=10,
    n_individuals=30,
    seed=42,
    qc_passed_filter=None
):
    """
    Generate simulated hauls based on proportions.
    
    Parameters:
      sample_to_pop: dict from load_metadata()
      proportions_list: list of dicts, e.g.
        [
          {"Autumn": 50, "North": 0, "Central": 0, "South": 50},
          {"Autumn": 60, "North": 40, "Central": 0, "South": 0},
        ]
      n_replicates: number of independent hauls per proportion (10)
      n_individuals: individuals per haul (30)
      seed: random seed for reproducibility
      qc_passed_filter: optional list of sample IDs that passed QC. If provided,
                        only these individuals will be used in hauls.
      
    Returns:
      haul_info: list of dicts with keys:
        - haul_id: "haul_0001", "haul_0002", etc.
        - proportion_name: e.g. "50A_0N_0C_50S"
        - expected_props: {"Autumn": 50, "North": 0, ...}
        - actual_counts: {"Autumn": 15, "North": 0, ...}
        - sample_ids: list of sample_ids in this haul
    """
    rng = np.random.default_rng(seed)
    pop_to_samples = get_individuals_per_population(sample_to_pop, qc_passed_filter)
    
    haul_info = []
    haul_counter = 1
    
    for prop_dict in proportions_list:
        counts = calculate_counts(prop_dict, n_individuals)
        proportion_name = "_".join([
            f"{int(prop_dict['Autumn'])}A",
            f"{int(prop_dict['North'])}N",
            f"{int(prop_dict['Central'])}C",
            f"{int(prop_dict['South'])}S"
        ])
        
        for rep in range(n_replicates):
            haul_id = f"haul_{haul_counter:04d}"
            haul_counter += 1
            
            # Sample individuals for this haul
            sampled_ids = []
            for pop in ["Autumn", "North", "Central", "South"]:
                count = counts[pop]
                if count > 0:
                    pop_samples = pop_to_samples[pop]
                    if len(pop_samples) < count:
                        # If not enough individuals, sample with replacement
                        sampled = rng.choice(pop_samples, size=count, replace=True)
                    else:
                        sampled = rng.choice(pop_samples, size=count, replace=False)
                    sampled_ids.extend(sampled)
            
            # Shuffle to randomize order
            rng.shuffle(sampled_ids)
            
            haul_info.append({
                "haul_id": haul_id,
                "proportion_name": proportion_name,
                "expected_props": prop_dict.copy(),
                "actual_counts": counts.copy(),
                "sample_ids": sampled_ids
            })
    
    return haul_info


def write_metadata_file(haul_info, output_path):
    """
    Write simulated_hauls_metadata.txt with columns: sample_id\thaul_id
    """
    with open(output_path, 'w') as f:
        f.write("sample_id\thaul_id\n")
        for haul in haul_info:
            haul_id = haul["haul_id"]
            for sample_id in haul["sample_ids"]:
                f.write(f"{sample_id}\t{haul_id}\n")
    print(f"✓ Wrote metadata to {output_path}")


def write_proportions_file(haul_info, output_path):
    """
    Write haul_proportions.txt with columns: haul_id\tproportion_name\tAutumn\tNorth\tCentral\tSouth\tActual_A\tActual_N\tActual_C\tActual_S
    """
    with open(output_path, 'w') as f:
        f.write("haul_id\tproportion_name\tAutumn_%\tNorth_%\tCentral_%\tSouth_%\tActual_A_count\tActual_N_count\tActual_C_count\tActual_S_count\n")
        for haul in haul_info:
            haul_id = haul["haul_id"]
            prop_name = haul["proportion_name"]
            exp = haul["expected_props"]
            act = haul["actual_counts"]
            f.write(
                f"{haul_id}\t{prop_name}\t"
                f"{int(exp['Autumn'])}\t{int(exp['North'])}\t{int(exp['Central'])}\t{int(exp['South'])}\t"
                f"{act['Autumn']}\t{act['North']}\t{act['Central']}\t{act['South']}\n"
            )
    print(f"✓ Wrote proportions to {output_path}")


def write_individuals_list(haul_info, output_path):
    """
    Write a simple list of all sample_ids used in simulated hauls (one per line).
    """
    all_ids = set()
    for haul in haul_info:
        all_ids.update(haul["sample_ids"])
    
    with open(output_path, 'w') as f:
        for sample_id in sorted(all_ids):
            f.write(f"{sample_id}\n")
    print(f"✓ Wrote individual list to {output_path} ({len(all_ids)} unique individuals)")


def print_summary(haul_info):
    """Print a summary of generated hauls."""
    print("\n" + "="*70)
    print("SIMULATED HAULS SUMMARY")
    print("="*70)
    
    # Group by proportion
    props = {}
    for haul in haul_info:
        pname = haul["proportion_name"]
        if pname not in props:
            props[pname] = []
        props[pname].append(haul)
    
    for pname in sorted(props.keys()):
        hauls = props[pname]
        print(f"\n{pname} ({len(hauls)} replicates):")
        for haul in hauls:
            exp = haul["expected_props"]
            act = haul["actual_counts"]
            print(f"  {haul['haul_id']}: "
                  f"A:{act['Autumn']}/{int(exp['Autumn'])}% "
                  f"N:{act['North']}/{int(exp['North'])}% "
                  f"C:{act['Central']}/{int(exp['Central'])}% "
                  f"S:{act['South']}/{int(exp['South'])}%")
    
    print(f"\nTotal hauls generated: {len(haul_info)}")
    print("="*70 + "\n")


def run_diversity_analysis(metadata_path, proportions_path):
    """Analyze haul diversity to ensure replicates are genetically distinct."""
    
    def load_haul_individuals(path):
        """Load haul metadata and return haul -> set of sample_ids."""
        haul_to_samples = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sample_id, haul_id = parts[0], parts[1]
                    if haul_id not in haul_to_samples:
                        haul_to_samples[haul_id] = set()
                    haul_to_samples[haul_id].add(sample_id)
        return haul_to_samples
    
    def load_proportions(path):
        """Load haul proportions."""
        haul_props = {}
        with open(path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    haul_id = parts[0]
                    a = int(parts[2])
                    n = int(parts[3])
                    c = int(parts[4])
                    s = int(parts[5])
                    haul_props[haul_id] = {"A": a, "N": n, "C": c, "S": s}
        return haul_props
    
    # Load data
    haul_to_samples = load_haul_individuals(metadata_path)
    haul_props = load_proportions(proportions_path)
    
    # Group hauls by proportion type
    prop_groups = {}
    for haul_id in haul_to_samples.keys():
        if haul_id not in haul_props:
            continue
        props = haul_props[haul_id]
        prop_key = f"A{props['A']:03d}_N{props['N']:03d}_C{props['C']:03d}_S{props['S']:03d}"
        if prop_key not in prop_groups:
            prop_groups[prop_key] = []
        prop_groups[prop_key].append(haul_id)
    
    # Analyze each proportion group
    print("\nHAUL DIVERSITY ANALYSIS (Individual-level)")
    print("-" * 70)
    
    all_distinct = True
    for prop_key in sorted(prop_groups.keys()):
        haul_ids = sorted(prop_groups[prop_key])
        
        if len(haul_ids) < 2:
            continue
        
        # Get individuals for each haul
        haul_sample_sets = {haul_id: haul_to_samples[haul_id] for haul_id in haul_ids}
        
        # Compute pairwise overlap
        overlaps = []
        for i in range(len(haul_ids)):
            for j in range(i + 1, len(haul_ids)):
                haul_i = haul_ids[i]
                haul_j = haul_ids[j]
                samples_i = haul_sample_sets[haul_i]
                samples_j = haul_sample_sets[haul_j]
                overlap = len(samples_i & samples_j)
                overlaps.append(overlap)
        
        if overlaps:
            mean_overlap = sum(overlaps) / len(overlaps)
            is_distinct = mean_overlap < 30
            all_distinct = all_distinct and is_distinct
            
            status = "✓ DISTINCT" if is_distinct else "⚠️  IDENTICAL"
            print(f"{prop_key}: {status} (avg {mean_overlap:.1f} shared individuals)")
    
    print("-" * 70)
    if all_distinct:
        print("✓ Haul diversity validation PASSED")
    else:
        print("⚠️  WARNING: Some hauls have identical individuals")


def main():
    """Main entry point."""
    # Resolve paths relative to this script's directory so it works from any CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Inputs
    original_metadata_path = os.path.normpath(os.path.join(script_dir, "..", "simulations", "All_Sample_Metadata.txt"))
    qc_passed_path = os.path.join(script_dir, "qc_passed_individuals.txt")
    # Outputs (written next to this script)
    output_dir = script_dir
    metadata_out = os.path.join(output_dir, "simulated_hauls_metadata.txt")
    proportions_out = os.path.join(output_dir, "haul_proportions.txt")
    individuals_out = os.path.join(output_dir, "simulated_individuals_list.txt")
    
    print("Loading original metadata...")
    if os.path.exists(original_metadata_path):
        sample_to_pop = load_metadata(original_metadata_path)
        print(f"  Loaded {len(sample_to_pop)} individuals from metadata file")
    else:
        print(f"  Metadata not found at: {original_metadata_path}")
        print("  Falling back to population VCF headers in ../simulations/")
        sim_dir = os.path.normpath(os.path.join(script_dir, "..", "simulations"))
        sample_to_pop = _load_samples_from_population_vcfs(sim_dir)
        print(f"  Inferred {len(sample_to_pop)} individuals from population VCFs")
    
    # Check if QC-passed individuals list exists
    qc_passed_filter = None
    if os.path.exists(qc_passed_path):
        print(f"\nLoading QC-passed individuals from {qc_passed_path}...")
        with open(qc_passed_path, 'r') as f:
            qc_passed_filter = [line.strip() for line in f if line.strip()]
        print(f"  Loaded {len(qc_passed_filter)} QC-passed individuals")
        print("  (Will use only these individuals in simulated hauls)")
    else:
        print(f"\nNote: {qc_passed_path} not found.")
        print("  Using all available individuals for simulated hauls.")
        print("  (Tip: Run qc_before_simulation.py first to QC-filter individuals)")
    
    # Define proportions (you can modify this as needed)
    proportions_list = [
        {"Autumn": 100, "North": 0, "Central": 0, "South": 0},    # 100 Autumn
        {"Autumn": 0, "North": 100, "Central": 0, "South": 0},    # 100 Spring (North)
        {"Autumn": 0, "North": 0, "Central": 100, "South": 0},    # 100 Spring (Central)
        {"Autumn": 0, "North": 0, "Central": 0, "South": 100},    # 100 Spring (South)
        {"Autumn": 0, "North": 33, "Central": 33, "South": 34},    # 100 Spring (Equal mix)

        {"Autumn": 90, "North": 10, "Central": 0, "South": 0},    # 90/10 Autumn (North)
        {"Autumn": 90, "North": 0, "Central": 10, "South": 0},    # 90/10 Autumn (Central)
        {"Autumn": 90, "North": 0, "Central": 0, "South": 10},    # 90/10 Autumn (South)
        {"Autumn": 90, "North": 3, "Central": 3, "South": 4},    # 90/10 Autumn (Equal mix)

        {"Autumn": 75, "North": 25, "Central": 0, "South": 0},    # 75/25 Autumn (North)
        {"Autumn": 75, "North": 0, "Central": 25, "South": 0},    # 75/25 Autumn (Central)
        {"Autumn": 75, "North": 0, "Central": 0, "South": 25},    # 75/25 Autumn (South)
        {"Autumn": 75, "North": 8, "Central": 8, "South": 9},    # 75/25 Autumn (Equal mix)

        {"Autumn": 50, "North": 50, "Central": 0, "South": 0},    # 50/50 Autumn-North
        {"Autumn": 50, "North": 0, "Central": 50, "South": 0},    # 50/50 Autumn-Central
        {"Autumn": 50, "North": 0, "Central": 0, "South": 50},    # 50/50 Autumn-South
        {"Autumn": 50, "North": 16, "Central": 17, "South": 17},    # 50/50 Autumn-Spring (mixed)


        {"Autumn": 25, "North": 75, "Central": 0, "South": 0},    # 25/75 Autumn (North)
        {"Autumn": 25, "North": 0, "Central": 75, "South": 0},    # 25/75 Autumn (Central)
        {"Autumn": 25, "North": 0, "Central": 0, "South": 75},    # 25/75 Autumn (South)
        {"Autumn": 25, "North": 25, "Central": 25, "South": 25},    # 25/75 Autumn (Equal mix)

        {"Autumn": 10, "North": 90, "Central": 0, "South": 0},    # 10/90 Spring (North)
        {"Autumn": 10, "North": 0, "Central": 90, "South": 0},    # 10/90 Spring (Central)
        {"Autumn": 10, "North": 0, "Central": 0, "South": 90},    # 10/90 Spring (South)
        {"Autumn": 10, "North": 30, "Central": 30, "South": 30},    # 10/90 Spring (Mixed)
    ]
    
    print(f"\nGenerating hauls with {len(proportions_list)} proportion(s), 10 replicates each...")
    haul_info = generate_hauls(
        sample_to_pop,
        proportions_list,
        n_replicates=10,
        n_individuals=30,
        seed=42,
        qc_passed_filter=qc_passed_filter
    )
    
    print(f"\nWriting output files...")
    write_metadata_file(haul_info, metadata_out)
    write_proportions_file(haul_info, proportions_out)
    write_individuals_list(haul_info, individuals_out)
    
    print_summary(haul_info)
    
    # Run diversity analysis
    print("\n" + "="*70)
    print("Running haul diversity analysis...")
    print("="*70)
    run_diversity_analysis(metadata_out, proportions_out)


if __name__ == "__main__":
    main()
