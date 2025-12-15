#!/usr/bin/env python3
"""
Debug script: inspect actual distances and ratios from the test run predictions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("../simulations"))

import json
import pandas as pd
import glob

# Load reference model to understand the centroid positions
with open("reference_model/reference_pca.json") as f:
    ref_pca = json.load(f)

# Load the last predictions CSV from hauls 11-40
predictions_file = "results/predictions_0011-0040_20251212_075331.csv"
if os.path.exists(predictions_file):
    predictions = pd.read_csv(predictions_file)
    
    print("=" * 90)
    print("SUMMARY: Pure Spring hauls (0% Autumn true) - Autumn vs Spring classification")
    print("=" * 90)
    print("\nHauls 11-20 (Pure North, should be 0% Autumn):\n")
    
    for idx, row in predictions.iloc[:10].iterrows():
        haul_id = row["haul_id"]
        true_a = row["true_Autumn"]
        pred_a = row["pred_Autumn"]
        
        print(f"{haul_id}:  True Autumn {true_a:5.1f}%  →  Pred Autumn {pred_a:5.1f}%  (error: {abs(pred_a - true_a):5.1f}%)")
    
    print("\nHauls 21-30 (Pure Central, should be 0% Autumn):\n")
    
    for idx, row in predictions.iloc[10:20].iterrows():
        haul_id = row["haul_id"]
        true_a = row["true_Autumn"]
        pred_a = row["pred_Autumn"]
        
        print(f"{haul_id}:  True Autumn {true_a:5.1f}%  →  Pred Autumn {pred_a:5.1f}%  (error: {abs(pred_a - true_a):5.1f}%)")
    
    print("\nHauls 31-40 (Pure South, should be 0% Autumn):\n")
    
    for idx, row in predictions.iloc[20:30].iterrows():
        haul_id = row["haul_id"]
        true_a = row["true_Autumn"]
        pred_a = row["pred_Autumn"]
        
        print(f"{haul_id}:  True Autumn {true_a:5.1f}%  →  Pred Autumn {pred_a:5.1f}%  (error: {abs(pred_a - true_a):5.1f}%)")
    
    print("\n" + "=" * 90)
    print("\nOBSERVATION:")
    print("If hauls are VISUALLY CLOSE to Spring centroid on PCA plots,")
    print("why are they predicted 8-20% Autumn instead of closer to 0%?")
    print("\nPossible reasons:")
    print("1. The ratio formula r = d_autumn / (d_autumn + d_spring)")
    print("   gives non-zero Autumn based on RELATIVE distance, not absolute.")
    print("   Even if Spring is closer, if Autumn isn't VERY far, Autumn gets a percentage.")
    print("\n2. The Autumn centroid might be closer to Spring hauls than expected.")
    print("\n3. The PCA space might have high dimensionality issues where relative")
    print("   positioning in 2D plots (PC1 vs PC2) differs from actual distances.")
    print("=" * 90)

else:
    print(f"Predictions file not found: {predictions_file}")
    print("\nSearching for available predictions files...")
    files = sorted(glob.glob("results/predictions_*.csv"))
    for f in files[:5]:
        print(f"  - {f}")
