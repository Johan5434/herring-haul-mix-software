#!/usr/bin/env python3
"""
Evaluate Spring PCA dimensionality: compare using 2 PCs vs 3 PCs for Step 2 (N/C/S) classification.
Also report Spring PCA explained variance for PCs used.

Assumptions:
- Run from simulated_experiments working directory.
- reference_model/reference_pca.json and reference_model/spring_pca.json exist.
- test_simulated_hauls.py prints Step 2 N/C/S MAE and Overall MAE.
"""

import os
import json
import subprocess
import numpy as np

WORKDIR = "/Users/elsarosenblad/Documents/Applied Bioinf/herring-haul-mix-software/simulated_experiments"

print("=" * 90)
print("SPRING PCA DIMENSIONALITY ANALYSIS (Step 2: N/C/S)")
print("=" * 90)

# Load Spring PCA explained variance if available
spring_pca_path = os.path.join("reference_model", "spring_pca.json")
if os.path.exists(spring_pca_path):
    with open(spring_pca_path) as f:
        spring_pca = json.load(f)
    ev = spring_pca.get("explained_var_ratio", [])
    print("\n1. EXPLAINED VARIANCE (Spring PCA):")
    print("-" * 90)
    if ev:
        for i, var in enumerate(ev[:5]):
            print(f"   PC{i+1}: {var*100:.2f}%")
        cumsum = np.cumsum(ev[:3] if len(ev) >= 3 else ev)
        if len(cumsum) >= 2:
            print(f"\n   Cumulative (PC1+PC2): {cumsum[1]*100:.2f}%")
        if len(cumsum) >= 3:
            print(f"   Cumulative (PC1+PC2+PC3): {cumsum[2]*100:.2f}%")
            print(f"   Gain from adding PC3: {(cumsum[2]-cumsum[1])*100:.2f}%")
    else:
        print("   No explained variance array present in spring_pca.json")
else:
    print("\n[warn] spring_pca.json not found; Step 2 may fall back to equal split.")

# Choose a set of hauls that are Spring-relevant to ensure Step 2 runs.
# Use pure Spring and Spring-heavy mixes to trigger Spring ≥ 90%.
# If your thresholds differ, adjust this list.
haul_ids = [
    "15",  # Pure North
    "25",  # Pure Central
    "35",  # Pure South
    "101", # 10A/90N (Spring-heavy)
]

print("\n2. TEST HAULS FOR STEP 2 (intended Spring ≥ 90% cases):")
print("-" * 90)
for h in haul_ids:
    print(f"   haul_{int(h):04d}")

# Utility to extract MAEs

def extract_step2_mae(output: str):
    mae = {"North": None, "Central": None, "South": None}
    overall = None
    lines = output.splitlines()
    # Capture Step 2 MAE block
    in_step2 = False
    for line in lines:
        if "STEP 2: SPRING N/C/S BREAKDOWN" in line:
            in_step2 = True
        elif in_step2 and "Mean Absolute Error (MAE) for N/C/S classification:" in line:
            continue
        elif in_step2 and line.strip().startswith("North:"):
            try:
                mae["North"] = float(line.split()[1].rstrip('%'))
            except Exception:
                pass
        elif in_step2 and line.strip().startswith("Central:"):
            try:
                mae["Central"] = float(line.split()[1].rstrip('%'))
            except Exception:
                pass
        elif in_step2 and line.strip().startswith("South:"):
            try:
                mae["South"] = float(line.split()[1].rstrip('%'))
            except Exception:
                pass
        elif in_step2 and line.strip().startswith("No hauls qualified for Step 2"):
            # No spring-relevant hauls
            break
    # Capture overall
    for line in lines:
        if line.strip().startswith("Overall:"):
            try:
                overall = float(line.split()[1].rstrip('%'))
            except Exception:
                pass
    return mae, overall

# Run with 2 PCs in Spring classification by temporarily editing SPRING_CLASSIFICATION_DIMS
class_path = os.path.join(WORKDIR, "../simulations/classification.py")
with open(class_path) as f:
    original = f.read()

print("\n3. RUNNING STEP 2 WITH 2 PCs (Spring PC1+PC2)...")
print("-" * 90)
modified_2pc = original.replace("SPRING_CLASSIFICATION_DIMS = (0, 1, 2)", "SPRING_CLASSIFICATION_DIMS = (0, 1)")
with open(class_path, "w") as f:
    f.write(modified_2pc)

cmd_2pc = f"source ../.venv/bin/activate && python test_simulated_hauls.py --hauls {','.join(haul_ids)}"
res_2pc = subprocess.run(cmd_2pc, shell=True, cwd=WORKDIR, capture_output=True, text=True)
mae_step2_2pc, overall_2pc = extract_step2_mae(res_2pc.stdout + res_2pc.stderr)

print("\n4. RUNNING STEP 2 WITH 3 PCs (Spring PC1+PC2+PC3)...")
print("-" * 90)
with open(class_path, "w") as f:
    f.write(original)  # restore 3D (0,1,2)

cmd_3pc = f"source ../.venv/bin/activate && python test_simulated_hauls.py --hauls {','.join(haul_ids)}"
res_3pc = subprocess.run(cmd_3pc, shell=True, cwd=WORKDIR, capture_output=True, text=True)
mae_step2_3pc, overall_3pc = extract_step2_mae(res_3pc.stdout + res_3pc.stderr)

# Restore original file in case of early exit
with open(class_path, "w") as f:
    f.write(original)

print("\n" + "=" * 90)
print("SUMMARY: SPRING PCA – 2 PCs vs 3 PCs (Step 2 N/C/S)")
print("=" * 90)

print("\nUsing 2 PCs (PC1+PC2):")
print(f"  Step 2 MAE → North:   {mae_step2_2pc['North'] if mae_step2_2pc['North'] is not None else 'n/a'}%")
print(f"                Central: {mae_step2_2pc['Central'] if mae_step2_2pc['Central'] is not None else 'n/a'}%")
print(f"                South:   {mae_step2_2pc['South'] if mae_step2_2pc['South'] is not None else 'n/a'}%")
print(f"  Overall: {overall_2pc if overall_2pc is not None else 'n/a'}%")

print("\nUsing 3 PCs (PC1+PC2+PC3):")
print(f"  Step 2 MAE → North:   {mae_step2_3pc['North'] if mae_step2_3pc['North'] is not None else 'n/a'}%")
print(f"                Central: {mae_step2_3pc['Central'] if mae_step2_3pc['Central'] is not None else 'n/a'}%")
print(f"                South:   {mae_step2_3pc['South'] if mae_step2_3pc['South'] is not None else 'n/a'}%")
print(f"  Overall: {overall_3pc if overall_3pc is not None else 'n/a'}%")

# Simple conclusion
if all(v is not None for v in mae_step2_2pc.values()) and all(v is not None for v in mae_step2_3pc.values()):
    avg_2 = np.mean([mae_step2_2pc['North'], mae_step2_2pc['Central'], mae_step2_2pc['South']])
    avg_3 = np.mean([mae_step2_3pc['North'], mae_step2_3pc['Central'], mae_step2_3pc['South']])
    if avg_3 < avg_2:
        print(f"\nConclusion: 3 PCs yield lower average N/C/S MAE ({avg_3:.2f}% vs {avg_2:.2f}%).")
    elif avg_3 > avg_2:
        print(f"\nConclusion: 2 PCs yield lower average N/C/S MAE ({avg_2:.2f}% vs {avg_3:.2f}%).")
    else:
        print(f"\nConclusion: No difference (avg MAE {avg_2:.2f}%).")
else:
    print("\nConclusion: Could not compute Step 2 MAE (no Spring-relevant hauls or parsing issue).")

print("\n" + "=" * 90)
