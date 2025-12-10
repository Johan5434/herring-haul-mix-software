#!/usr/bin/env python3
"""
create_simulated_metadata.py

Create a proper metadata file for simulated hauls in the format expected by the pipeline.
Format: VCF_ID\tBioSample_ID\tCapture_Population\tSpawning_Ecotype\tPhenotype_ID\tPopulation\tSimulated_Haul_ID\tTrue_Proportion

This allows the pipeline to track both the original information and the simulated haul/proportion assignment.
"""

import os


def load_original_metadata(original_metadata_path):
    """Load the original metadata file."""
    metadata = {}
    with open(original_metadata_path, 'r') as f:
        header = f.readline().strip()
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) < 6:
                continue
            sample_id = cols[0]
            metadata[sample_id] = cols
    return metadata, header


def load_haul_metadata(haul_metadata_path):
    """Load the simulated hauls metadata (sample_id -> haul_id)."""
    haul_map = {}
    with open(haul_metadata_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) == 2:
                sample_id, haul_id = cols
                haul_map[sample_id] = haul_id
    return haul_map


def load_proportions(proportions_path):
    """Load the proportions file to map haul_id -> expected proportions."""
    props = {}
    with open(proportions_path, 'r') as f:
        f.readline()  # skip header
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) >= 2:
                haul_id = cols[0]
                proportion_name = cols[1]
                props[haul_id] = proportion_name
    return props


def main():
    """Main entry point."""
    original_metadata_path = "../simulations/All_Sample_Metadata.txt"
    haul_metadata_path = "./simulated_hauls_metadata.txt"
    proportions_path = "./haul_proportions.txt"
    output_path = "./simulated_hauls_extended_metadata.txt"
    
    print("Loading original metadata...")
    original_metadata, original_header = load_original_metadata(original_metadata_path)
    print(f"  Loaded {len(original_metadata)} individuals")
    
    print("Loading haul assignments...")
    haul_map = load_haul_metadata(haul_metadata_path)
    print(f"  Loaded {len(haul_map)} haul assignments")
    
    print("Loading proportions...")
    props = load_proportions(proportions_path)
    print(f"  Loaded {len(props)} haul proportions")
    
    print(f"\nWriting extended metadata to {output_path}...")
    with open(output_path, 'w') as fout:
        # Extended header
        extended_header = original_header + "\tSimulated_Haul_ID\tTrue_Proportion"
        fout.write(extended_header + "\n")
        
        # Write rows for all individuals that are in simulated hauls
        for sample_id in sorted(haul_map.keys()):
            if sample_id in original_metadata:
                haul_id = haul_map[sample_id]
                proportion_name = props.get(haul_id, "unknown")
                original_cols = original_metadata[sample_id]
                row = "\t".join(original_cols) + f"\t{haul_id}\t{proportion_name}"
                fout.write(row + "\n")
    
    print(f"✓ Wrote extended metadata to {output_path}")
    
    # Also create a simple version for easy haul metadata
    simple_metadata_path = "./simulated_hauls_simple_metadata.txt"
    print(f"\nWriting simple metadata to {simple_metadata_path}...")
    with open(simple_metadata_path, 'w') as f:
        f.write("VCF_ID\tHaul_ID\n")
        for sample_id in sorted(haul_map.keys()):
            haul_id = haul_map[sample_id]
            f.write(f"{sample_id}\t{haul_id}\n")
    print(f"✓ Wrote simple metadata to {simple_metadata_path}")


if __name__ == "__main__":
    main()
