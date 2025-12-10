#!/bin/bash
# Example: How to Use Simulated Hauls with the Pipeline

# This script shows the recommended workflow for testing the classification
# pipeline with simulated hauls of known composition.

set -e  # Exit on error

echo "=========================================="
echo "Simulated Hauls Pipeline Workflow"
echo "=========================================="

# Define paths
PROJECT_DIR="/Users/elsarosenblad/Documents/Applied Bioinf/herring-haul-mix-software"
SIMULATIONS_DIR="$PROJECT_DIR/simulations"
SIMULATED_EXPS_DIR="$PROJECT_DIR/simulated_experiments"
VENV_PATH="$PROJECT_DIR/.venv/bin/activate"

# Step 1: Copy simulated files to simulations directory
echo -e "\n[1/4] Copying simulated data to simulations folder..."
cp "$SIMULATED_EXPS_DIR/simulated_hauls.vcf" "$SIMULATIONS_DIR/"
cp "$SIMULATED_EXPS_DIR/simulated_hauls_metadata.txt" "$SIMULATIONS_DIR/"
cp "$SIMULATED_EXPS_DIR/haul_proportions.txt" "$SIMULATIONS_DIR/"
echo "✓ Files copied"

# Step 2: Activate virtual environment and run pipeline
echo -e "\n[2/4] Running main_simulation_motor.py..."
cd "$SIMULATIONS_DIR"
source "$VENV_PATH"

# Note: The following is interactive - you'll need to answer prompts
python main_simulation_motor.py << EOF
y
simulated_hauls.vcf
simulated_hauls_metadata.txt
n
n
n
n
y
n
y
EOF

echo "✓ Pipeline complete"

# Step 3: Save reference PCA
echo -e "\n[3/4] Checking for GM_all_after_QC.npz..."
if [ -f "GM_all_after_QC.npz" ]; then
    echo "✓ Reference PCA saved"
else
    echo "✗ Warning: GM_all_after_QC.npz not found"
fi

# Step 4: Evaluate results
echo -e "\n[4/4] Generating evaluation report..."
python << 'PYTHON_EVAL'
import pandas as pd
import numpy as np

# Load the true proportions
try:
    props = pd.read_csv("haul_proportions.txt", sep="\t")
    print("\n" + "="*60)
    print("SIMULATED HAULS - TRUE PROPORTIONS")
    print("="*60)
    
    # Group by proportion and show statistics
    grouped = props.groupby("proportion_name")
    for prop_name, group in grouped:
        print(f"\n{prop_name} ({len(group)} hauls):")
        print(f"  Expected: {group['Autumn_%'].iloc[0]}% A, "
              f"{group['North_%'].iloc[0]}% N, "
              f"{group['Central_%'].iloc[0]}% C, "
              f"{group['South_%'].iloc[0]}% S")
        print(f"  Range of actual counts:")
        print(f"    Autumn:  {group['Actual_A_count'].min()}-{group['Actual_A_count'].max()}")
        print(f"    North:   {group['Actual_N_count'].min()}-{group['Actual_N_count'].max()}")
        print(f"    Central: {group['Actual_C_count'].min()}-{group['Actual_C_count'].max()}")
        print(f"    South:   {group['Actual_S_count'].min()}-{group['Actual_S_count'].max()}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Run main_pca_mixes_experiments.py")
    print("  2. Load GM_all_after_QC.npz")
    print("  3. Project simulated hauls into PCA space")
    print("  4. Compare results with haul_proportions.txt")
    print("="*60 + "\n")
    
except FileNotFoundError:
    print("✗ haul_proportions.txt not found")
PYTHON_EVAL

echo -e "\n✓ Workflow complete!"
echo "You can now use main_pca_mixes_experiments.py to evaluate classification accuracy."
