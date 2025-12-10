# Simulated Hauls Experiment Folder

This folder contains simulated herring hauls with **known population proportions** for testing and validating the PCA-based classification pipeline.

## Overview

The goal is to create synthetic hauls by mixing individuals from different populations at specified proportions, then use the pipeline to see how accurately it can recover the true population assignments.

## Files Generated

### Core Output Files (Minimal)

**You only need this ONE file:**

1. **`simulated_hauls_metadata.txt`** (2 columns)
   - Format: `sample_id\thaul_id`
   - Maps each individual fish to its simulated haul
   - **Use this as the metadata file for the pipeline**
   - Each haul contains 30 individuals
   - Only 35 KB (not a separate 21 MB VCF)

**And use the existing original VCF:**

- Point pipeline to: `../simulations/Bioinformatics_Course_2025_Herring_Sample_Subset.vcf`
- Pipeline automatically extracts only the simulated individuals

**For validation:**

2. **`haul_proportions.txt`** (10 columns)

   - Format: `haul_id\tproportion_name\tAutumn_%\tNorth_%\tCentral_%\tSouth_%\tActual_A_count\tActual_N_count\tActual_C_count\tActual_S_count`
   - Documents the **true proportion** for each simulated haul
   - Shows both expected percentages and actual individual counts
   - Essential for evaluating pipeline accuracy

3. **`simulated_individuals_list.txt`**

   - Simple list of all 1,102 sample IDs used in simulated hauls (one per line)
   - Reference only (not needed for pipeline)

4. **`simulated_hauls_extended_metadata.txt`**

   - All columns from original metadata PLUS: `Simulated_Haul_ID` and `True_Proportion`
   - Preserves original metadata while adding haul assignment
   - Useful for detailed analysis

5. **`simulated_hauls_simple_metadata.txt`**
   - Minimal format: `VCF_ID\tHaul_ID`
   - Alternative to the main metadata file

## Generated Hauls

**Total: 50 simulated hauls** (5 different proportions × 10 replicates each)

### Proportions

| Haul IDs       | Composition                        | Replicates |
| -------------- | ---------------------------------- | ---------- |
| haul_0001–0010 | 50% Autumn, 50% South              | 10         |
| haul_0011–0020 | 60% Autumn, 40% North              | 10         |
| haul_0021–0030 | 50% North, 50% Central             | 10         |
| haul_0031–0040 | 33% Autumn, 33% North, 33% Central | 10         |
| haul_0041–0050 | 25% each population (equal 4-way)  | 10         |

Each haul contains **30 individuals** sampled randomly from the specified populations.

## How to Use in the Pipeline

### Option 1: Use with `main_simulation_motor.py`

1. Copy metadata file to simulations directory:

   ```bash
   cp simulated_hauls_metadata.txt ../simulations/
   ```

2. Run the pipeline:

   ```bash
   cd ../simulations
   source ../.venv/bin/activate
   python main_simulation_motor.py
   ```

3. When asked for a VCF file, provide: `Bioinformatics_Course_2025_Herring_Sample_Subset.vcf`

4. When asked for metadata, provide: `simulated_hauls_metadata.txt`

5. The pipeline will:
   - Load the original VCF
   - Read the metadata file
   - Automatically extract only the 1,102 simulated individuals
   - QC the hauls and build a reference PCA

### Option 2: Use with `main_pca_mixes_experiments.py`

1. Generate reference PCA from real data (using `main_simulation_motor.py`)
2. Save `GM_all_after_QC.npz`
3. Run:
   ```bash
   python main_pca_mixes_experiments.py
   ```
4. The script will let you load the reference and project these simulated hauls into PCA space
5. Compare the true proportions (from `haul_proportions.txt`) with the pipeline's classification

## Validation

To check if the pipeline works correctly:

1. **Accuracy by proportion**: Do hauls with the same known proportion cluster together in PCA space?
2. **Classification accuracy**: Does the rule-based classifier (from `classification.py`) correctly assign population percentages?
3. **Robustness**: Do the 10 replicates of each proportion show consistent behavior despite different random individuals?

## Metadata Column Reference

### `simulated_hauls_metadata.txt`

- **sample_id**: Individual fish ID from VCF (e.g., `Gav17_001.CEL`)
- **haul_id**: Simulated haul ID (e.g., `haul_0001`)

### `haul_proportions.txt`

- **haul_id**: Simulated haul ID
- **proportion_name**: Human-readable proportion string (e.g., `50A_0N_0C_50S`)
- **Autumn\_%**, **North\_%**, **Central\_%**, **South\_%**: Expected percentages
- **Actual_A_count**, **Actual_N_count**, **Actual_C_count**, **Actual_S_count**: Actual individual counts (may differ slightly from percentages due to rounding)

## Scripts

### `generate_simulated_hauls.py`

Generates the simulated hauls by:

1. Loading original metadata and sample-to-population mapping
2. Defining desired proportions
3. Randomly sampling individuals from each population
4. Creating 10 replicates per proportion
5. Writing output metadata files

**Usage:**

```bash
python generate_simulated_hauls.py
```

### `create_simulated_metadata.py`

Creates extended metadata files that map individuals to hauls and proportions.

**Usage:**

```bash
python create_simulated_metadata.py
```

## Customization

To create different proportions, edit `generate_simulated_hauls.py` and modify the `proportions_list`:

```python
proportions_list = [
    {"Autumn": 50, "North": 0, "Central": 0, "South": 50},
    {"Autumn": 60, "North": 40, "Central": 0, "South": 0},
    # Add more...
]
```

Then rerun the generation script.

## Notes

- **Sampling with replacement**: If a proportion requires more individuals from a population than available, sampling is done with replacement
- **Deterministic**: Scripts use `seed=42` for reproducibility
- **Rounding**: Due to integer counts, actual counts may differ slightly from expected percentages
- **All 4 populations**: When a population has 0% in a proportion, no individuals are sampled from it

## Questions or Issues?

The metadata files are straightforward TSV files that can be opened in any text editor or spreadsheet software. The VCF follows standard VCF4.2 format.
