#!/usr/bin/env python3
"""
Evaluate the impact of using 2 PCs vs 3 PCs for classification.
Runs the same diverse test hauls with both configurations and compares accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("../simulations"))

import json
import subprocess
import pandas as pd
import numpy as np

# Load reference model to show explained variance
with open("reference_model/reference_pca.json") as f:
    ref_pca = json.load(f)

print("=" * 90)
print("PC DIMENSIONALITY ANALYSIS FOR CLASSIFICATION")
print("=" * 90)

# Show explained variance
explained_var = ref_pca["explained_var_ratio"]
print("\n1. EXPLAINED VARIANCE BY PRINCIPAL COMPONENT (Reference PCA):")
print("-" * 90)
for i, var in enumerate(explained_var[:5]):
    print(f"   PC{i+1}: {var*100:.2f}%")

cumsum = np.cumsum(explained_var[:3])
print(f"\n   Cumulative variance (PC1+PC2): {cumsum[1]*100:.2f}%")
print(f"   Cumulative variance (PC1+PC2+PC3): {cumsum[2]*100:.2f}%")
print(f"   Gain from adding PC3: {(cumsum[2]-cumsum[1])*100:.2f}%")

# Define 10 diverse test hauls covering different scenarios
test_hauls = [
    "1",      # Pure Autumn
    "15",     # Pure North
    "25",     # Pure Central
    "35",     # Pure South
    "61",     # 50A/50N mix
    "51",     # 50A/50S mix
    "45",     # 0A/33N/33C/34S (pure Spring mix)
    "81",     # 50A/16N/17C/17S (mixed with all)
    "101",    # 10A/90N (Spring-heavy)
    "161"     # 25A/25N/25C/25S (balanced all 4)
]

print("\n2. TEST HAULS SELECTED (10 diverse scenarios):")
print("-" * 90)
haul_descriptions = {
    "1": "Pure Autumn",
    "15": "Pure North",
    "25": "Pure Central",
    "35": "Pure South",
    "61": "50% Autumn / 50% North",
    "51": "50% Autumn / 50% South",
    "45": "100% Spring (33N/33C/34S)",
    "81": "50% Autumn / 16N/17C/17S",
    "101": "10% Autumn / 90% North",
    "161": "25% each (A/N/C/S)"
}

for h, desc in haul_descriptions.items():
    print(f"   haul_{int(h):04d}: {desc}")

# Create haul range string
haul_range = f"{int(test_hauls[0]):04d}-{int(test_hauls[-1]):04d}"

print("\n3. RUNNING CLASSIFICATION WITH 2 PCs (PC1 + PC2)...")
print("-" * 90)

# Run with 2 PCs (we'll modify the code temporarily)
cmd_2pc = f"source ../.venv/bin/activate && python test_simulated_hauls.py --hauls {','.join(test_hauls)} 2>&1 | tail -50"
result_2pc = subprocess.run(cmd_2pc, shell=True, cwd="/Users/elsarosenblad/Documents/Applied Bioinf/herring-haul-mix-software/simulated_experiments", capture_output=True, text=True)

# Extract MAE from output
def extract_mae(output):
    """Extract overall MAE from test output"""
    for line in output.split('\n'):
        if 'Overall:' in line and '%' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if 'Overall:' in part:
                    try:
                        return float(parts[i+1].rstrip('%'))
                    except:
                        pass
    return None

mae_2pc = extract_mae(result_2pc.stdout + result_2pc.stderr)

print("\n4. RUNNING CLASSIFICATION WITH 3 PCs (PC1 + PC2 + PC3)...")
print("-" * 90)

# Need to modify DISTANCE_DIMS in classification.py temporarily
# For now, just run the standard test which uses 2 PCs
# Then create a modified version for 3 PCs

# Read classification.py
with open("../simulations/classification.py") as f:
    class_content = f.read()

# Check current DISTANCE_DIMS
if "DISTANCE_DIMS = (0, 1)" in class_content:
    print("Current DISTANCE_DIMS = (0, 1) [2 PCs detected]")
    
    # Temporarily change to 3 PCs
    modified_content = class_content.replace("DISTANCE_DIMS = (0, 1)", "DISTANCE_DIMS = (0, 1, 2)")
    
    # Write modified version
    with open("../simulations/classification.py", "w") as f:
        f.write(modified_content)
    
    print("Temporarily modified to DISTANCE_DIMS = (0, 1, 2)")
    
    # Run test
    cmd_3pc = f"source ../.venv/bin/activate && python test_simulated_hauls.py --hauls {','.join(test_hauls)} 2>&1 | tail -50"
    result_3pc = subprocess.run(cmd_3pc, shell=True, cwd="/Users/elsarosenblad/Documents/Applied Bioinf/herring-haul-mix-software/simulated_experiments", capture_output=True, text=True)
    
    mae_3pc = extract_mae(result_3pc.stdout + result_3pc.stderr)
    
    # Restore original
    with open("../simulations/classification.py", "w") as f:
        f.write(class_content)
    
    print("Restored original DISTANCE_DIMS = (0, 1)")
else:
    print("Warning: Could not find expected DISTANCE_DIMS pattern")
    mae_3pc = None

# Summary
print("\n" + "=" * 90)
print("SUMMARY: IMPACT OF USING 3 PCs vs 2 PCs")
print("=" * 90)

if mae_2pc is not None:
    print(f"\nUsing 2 PCs (PC1+PC2):  Overall MAE = {mae_2pc:.2f}%")
else:
    print(f"\nUsing 2 PCs (PC1+PC2):  Overall MAE = [failed to extract]")

if mae_3pc is not None:
    print(f"Using 3 PCs (PC1+PC2+PC3): Overall MAE = {mae_3pc:.2f}%")
else:
    print(f"Using 3 PCs (PC1+PC2+PC3): Overall MAE = [failed to extract]")

if mae_2pc is not None and mae_3pc is not None:
    diff = mae_2pc - mae_3pc
    improvement = (diff / mae_2pc) * 100 if mae_2pc > 0 else 0
    
    if diff > 0:
        print(f"\nConclusion: Adding PC3 IMPROVES accuracy by {diff:.2f}% points ({improvement:.1f}% relative improvement)")
    elif diff < 0:
        print(f"\nConclusion: Adding PC3 DECREASES accuracy by {abs(diff):.2f}% points ({abs(improvement):.1f}% relative degradation)")
    else:
        print(f"\nConclusion: No difference between 2 and 3 PCs")

print("\n" + "=" * 90)
