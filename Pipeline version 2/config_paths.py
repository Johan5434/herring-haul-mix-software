"""Central place to configure input/output paths for Pipeline v2 runs.

Edit these values to point to your data; test_simulated_hauls.py will import them.
Paths are relative to the Pipeline version 2 directory unless you provide absolute paths.

REQUIRED inputs:
  - DEFAULT_VCF: Original VCF file with all genotype data
  - DEFAULT_METADATA: Simulated hauls metadata file (sample_id → haul_id mapping)
  - DEFAULT_MODEL_DIR: Directory containing reference_pca.json and optionally spring_pca.json

OPTIONAL inputs:
  - DEFAULT_PROPORTIONS: Ground-truth haul proportions file. If missing, pipeline runs without
                         error metrics or summary comparisons (prediction-only mode).

OUTPUTS:
  - DEFAULT_OUTPUT_DIR: Directory where CSV results and plots are saved
"""

# ===== REQUIRED INPUTS =====
DEFAULT_VCF = "../Pipeline version 1/Bioinformatics_Course_2025_Herring_Sample_Subset.vcf"
DEFAULT_METADATA = "simulated_hauls_metadata.txt"
DEFAULT_MODEL_DIR = "reference_model"

# ===== OPTIONAL INPUTS =====
# Set to None or empty string to skip (pipeline will proceed without ground-truth comparisons)
DEFAULT_PROPORTIONS = "haul_proportions.txt"

# ===== OUTPUTS =====
DEFAULT_OUTPUT_DIR = "results"
