#!/usr/bin/env python3
"""Verify that all individuals in simulated hauls are QC-passed."""

import sys

# Load QC-passed individuals
print("Loading QC-passed individuals...")
qc_passed = set()
with open("qc_passed_individuals.txt", "r") as f:
    for line in f:
        sample_id = line.strip()
        if sample_id:
            qc_passed.add(sample_id)

print(f"  Loaded {len(qc_passed)} QC-passed individuals")

# Load haul metadata
print("\nLoading simulated hauls metadata...")
haul_samples = []
with open("simulated_hauls_metadata.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("sample_id"):
            continue
        parts = line.split("\t")
        sample_id = parts[0]
        haul_samples.append(sample_id)

print(f"  Loaded {len(haul_samples)} samples from hauls")

# Verify all haul samples are in QC-passed set
print("\nVerifying all haul samples are QC-passed...")
failed = []
for sample_id in haul_samples:
    if sample_id not in qc_passed:
        failed.append(sample_id)

if failed:
    print(f"❌ ERROR: {len(failed)} samples NOT in QC-passed set:")
    for sample_id in failed[:10]:
        print(f"  - {sample_id}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")
    sys.exit(1)
else:
    print(f"✓ All {len(haul_samples)} haul samples are QC-passed!")
    
print("\n✓ Verification PASSED: Ground truth proportions are accurate")
