# simulations/classification.py

import numpy as np
from pca_hauls import build_spring_pca_from_ref, project_individuals_to_PCs
# ===========================
#  CONFIGURABLE CONSTANTS
# ===========================

SPRING_CLASSIFICATION_DIMS = (0, 1, 2)
PURE_THRESHOLD_SEASON = 0.90
PURE_THRESHOLD_SPRING = 0.90
DISTANCE_DIMS = (0, 1)

# Distance threshold for 100% pure classification
# If a haul is within this distance of a pure population centroid, classify as 100% that population
PURE_DISTANCE_THRESHOLD = 2.0  # Adjust this value based on your PCA space scale

AUTUMN_SPRING_RULE_TABLE = [
    (0.00, (100, 0)),
    (0.01, (99, 1)),
    (0.02, (98, 2)),
    (0.03, (97, 3)),
    (0.04, (96, 4)),
    (0.05, (95, 5)),
    (0.06, (94, 6)),
    (0.07, (93, 7)),
    (0.08, (92, 8)),
    (0.09, (91, 9)),
    (0.10, (90, 10)),
    (0.11, (89, 11)),
    (0.12, (88, 12)),
    (0.13, (87, 13)),
    (0.14, (86, 14)),
    (0.15, (85, 15)),
    (0.16, (84, 16)),
    (0.17, (83, 17)),
    (0.18, (82, 18)),
    (0.19, (81, 19)),
    (0.20, (80, 20)),
    (0.21, (79, 21)),
    (0.22, (78, 22)),
    (0.23, (77, 23)),
    (0.24, (76, 24)),
    (0.25, (75, 25)),
    (0.26, (74, 26)),
    (0.27, (73, 27)),
    (0.28, (72, 28)),
    (0.29, (71, 29)),
    (0.30, (70, 30)),
    (0.31, (69, 31)),
    (0.32, (68, 32)),
    (0.33, (67, 33)),
    (0.34, (66, 34)),
    (0.35, (65, 35)),
    (0.36, (64, 36)),
    (0.37, (63, 37)),
    (0.38, (62, 38)),
    (0.39, (61, 39)),
    (0.40, (60, 40)),
    (0.41, (59, 41)),
    (0.42, (58, 42)),
    (0.43, (57, 43)),
    (0.44, (56, 44)),
    (0.45, (55, 45)),
    (0.46, (54, 46)),
    (0.47, (53, 47)),
    (0.48, (52, 48)),
    (0.49, (51, 49)),
    (0.50, (50, 50)),
    (0.51, (49, 51)),
    (0.52, (48, 52)),
    (0.53, (47, 53)),
    (0.54, (46, 54)),
    (0.55, (45, 55)),
    (0.56, (44, 56)),
    (0.57, (43, 57)),
    (0.58, (42, 58)),
    (0.59, (41, 59)),
    (0.60, (40, 60)),
    (0.61, (39, 61)),
    (0.62, (38, 62)),
    (0.63, (37, 63)),
    (0.64, (36, 64)),
    (0.65, (35, 65)),
    (0.66, (34, 66)),
    (0.67, (33, 67)),
    (0.68, (32, 68)),
    (0.69, (31, 69)),
    (0.70, (30, 70)),
    (0.71, (29, 71)),
    (0.72, (28, 72)),
    (0.73, (27, 73)),
    (0.74, (26, 74)),
    (0.75, (25, 75)),
    (0.76, (24, 76)),
    (0.77, (23, 77)),
    (0.78, (22, 78)),
    (0.79, (21, 79)),
    (0.80, (20, 80)),
    (0.81, (19, 81)),
    (0.82, (18, 82)),
    (0.83, (17, 83)),
    (0.84, (16, 84)),
    (0.85, (15, 85)),
    (0.86, (14, 86)),
    (0.87, (13, 87)),
    (0.88, (12, 88)),
    (0.89, (11, 89)),
    (0.90, (10, 90)),
    (0.91, (9, 91)),
    (0.92, (8, 92)),
    (0.93, (7, 93)),
    (0.94, (6, 94)),
    (0.95, (5, 95)),
    (0.96, (4, 96)),
    (0.97, (3, 97)),
    (0.98, (2, 98)),
    (0.99, (1, 99)),
    (1.00, (0, 100)),
]
# 3D rule table: (rN, rC, rS)  ->  (N%, C%, S%)
SPRING_R3D_RULE_TABLE = [
    ((1.00, 0.00, 0.00), (100, 0,   0)),   # pure North
    ((0.90, 0.10, 0.00), (90,  10,  0)),
    ((0.90, 0.00, 0.10), (90,  0,   10)),
    ((0.80, 0.20, 0.00), (80,  20,  0)),
    ((0.80, 0.10, 0.10), (80,  10,  10)),
    ((0.80, 0.00, 0.20), (80,  0,   20)),
    ((0.70, 0.30, 0.00), (70,  30,  0)),
    ((0.70, 0.20, 0.10), (70,  20,  10)),
    ((0.70, 0.10, 0.20), (70,  10,  20)),
    ((0.70, 0.00, 0.30), (70,  0,   30)),
    ((0.60, 0.40, 0.00), (60,  40,  0)),
    ((0.60, 0.30, 0.10), (60,  30,  10)),
    ((0.60, 0.20, 0.20), (60,  20,  20)),
    ((0.60, 0.10, 0.30), (60,  10,  30)),
    ((0.60, 0.00, 0.40), (60,  0,   40)),
    ((0.50, 0.50, 0.00), (50,  50,  0)),
    ((0.50, 0.40, 0.10), (50,  40,  10)),
    ((0.50, 0.30, 0.20), (50,  30,  20)),
    ((0.50, 0.20, 0.30), (50,  20,  30)),
    ((0.50, 0.10, 0.40), (50,  10,  40)),
    ((0.50, 0.00, 0.50), (50,  0,   50)),
    ((0.40, 0.60, 0.00), (40,  60,  0)),
    ((0.40, 0.50, 0.10), (40,  50,  10)),
    ((0.40, 0.40, 0.20), (40,  40,  20)),
    ((0.40, 0.30, 0.30), (40,  30,  30)),
    ((0.40, 0.20, 0.40), (40,  20,  40)),
    ((0.40, 0.10, 0.50), (40,  10,  50)),
    ((0.40, 0.00, 0.60), (40,  0,   60)),
    ((0.30, 0.70, 0.00), (30,  70,  0)),
    ((0.30, 0.60, 0.10), (30,  60,  10)),
    ((0.30, 0.50, 0.20), (30,  50,  20)),
    ((0.30, 0.40, 0.30), (30,  40,  30)),
    ((0.30, 0.30, 0.40), (30,  30,  40)),
    ((0.30, 0.20, 0.50), (30,  20,  50)),
    ((0.30, 0.10, 0.60), (30,  10,  60)),
    ((0.30, 0.00, 0.70), (30,  0,   70)),
    ((0.20, 0.80, 0.00), (20,  80,  0)),
    ((0.20, 0.70, 0.10), (20,  70,  10)),
    ((0.20, 0.60, 0.20), (20,  60,  20)),
    ((0.20, 0.50, 0.30), (20,  50,  30)),
    ((0.20, 0.40, 0.40), (20,  40,  40)),
    ((0.20, 0.30, 0.50), (20,  30,  50)),
    ((0.20, 0.20, 0.60), (20,  20,  60)),
    ((0.20, 0.10, 0.70), (20,  10,  70)),
    ((0.20, 0.00, 0.80), (20,  0,   80)),
    ((0.10, 0.90, 0.00), (10,  90,  0)),
    ((0.10, 0.80, 0.10), (10,  80,  10)),
    ((0.10, 0.70, 0.20), (10,  70,  20)),
    ((0.10, 0.60, 0.30), (10,  60,  30)),
    ((0.10, 0.50, 0.40), (10,  50,  40)),
    ((0.10, 0.40, 0.50), (10,  40,  50)),
    ((0.10, 0.30, 0.60), (10,  30,  60)),
    ((0.10, 0.20, 0.70), (10,  20,  70)),
    ((0.10, 0.10, 0.80), (10,  10,  80)),
    ((0.10, 0.00, 0.90), (10,  0,   90)),
    ((0.00, 1.00, 0.00), (0,   100, 0)),
    ((0.00, 0.90, 0.10), (0,   90,  10)),
    ((0.00, 0.80, 0.20), (0,   80,  20)),
    ((0.00, 0.70, 0.30), (0,   70,  30)),
    ((0.00, 0.60, 0.40), (0,   60,  40)),
    ((0.00, 0.50, 0.50), (0,   50,  50)),
    ((0.00, 0.40, 0.60), (0,   40,  60)),
    ((0.00, 0.30, 0.70), (0,   30,  70)),
    ((0.00, 0.20, 0.80), (0,   20,  80)),
    ((0.00, 0.10, 0.90), (0,   10,  90)),
    ((0.00, 0.00, 1.00), (0,   0,   100)),  # pure South
]


def _apply_r3d_rule_table(r3d, table=SPRING_R3D_RULE_TABLE):
    """
    Given r3d = {"North": rN, "Central": rC, "South": rS} (summing ≈ 1),
    choose the row in SPRING_R3D_RULE_TABLE whose r3d point is closest
    in Euclidean sense.

    Returns (pred_NCS) as a dict {"North": pN, "Central": pC, "South": pS}.
    """
    import numpy as np

    # target vector
    target = np.array([
        float(r3d.get("North", 0.0)),
        float(r3d.get("Central", 0.0)),
        float(r3d.get("South", 0.0)),
    ])

    best = None
    best_dist = None

    for (rN, rC, rS), (pN, pC, pS) in table:
        center = np.array([rN, rC, rS], dtype=float)
        dist = np.sum((center - target) ** 2)

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (pN, pC, pS)

    if best is None:
        return {"North": 0, "Central": 0, "South": 0}

    pN, pC, pS = best
    return {"North": pN, "Central": pC, "South": pS}


SPRING_RELEVANT_MIN_SPRING_FRAC = 0.90
SPRING_RELEVANT_MAX_AUTUMN_FRAC = 0.10


def _subspace(pc, dims=DISTANCE_DIMS):
    """Extract PC1/PC2 (or other dimensions) for distance calculation."""
    pc = np.asarray(pc, dtype=float)
    return pc[list(dims)]


def _euclidean(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def compute_population_centroids(ref_pca):
    """
    Build pop->centroid for Autumn, North, Central, South etc.
    """
    centroids = ref_pca["centroids"]   # (n_hauls, k)
    haul_pops = ref_pca["haul_pops"]   # (n_hauls,)

    pops = np.unique(haul_pops)
    pop_to_centroid = {}
    for pop in pops:
        mask = (haul_pops == pop)
        pop_to_centroid[pop] = centroids[mask].mean(axis=0)
    return pop_to_centroid


def compute_season_centroids(ref_pca):
    """
    Autumn vs Spring centroids in the same PC space.

    Spring = {North, Central, South} merged.
    """
    centroids = ref_pca["centroids"]
    haul_pops = ref_pca["haul_pops"]

    mask_aut = (haul_pops == "Autumn")
    mask_spr = (haul_pops == "North") | (haul_pops == "Central") | (haul_pops == "South")

    if not np.any(mask_aut):
        raise RuntimeError("No Autumn hauls found in the reference.")
    if not np.any(mask_spr):
        raise RuntimeError("No Spring hauls (North/Central/South) found in the reference.")

    autumn_centroid = centroids[mask_aut].mean(axis=0)
    spring_centroid = centroids[mask_spr].mean(axis=0)

    return {
        "Autumn": autumn_centroid,
        "Spring": spring_centroid,
    }


def _distances_to_centroids(pc, label_to_centroid, dims=DISTANCE_DIMS):
    """
    Compute Euclidean distance from a point pc to each centroid
    in label_to_centroid, in the PC dimensions specified by dims.
    """
    pc = np.asarray(pc, dtype=float)
    pc_sub = pc[list(dims)]

    dists = {}
    for label, c in label_to_centroid.items():
        c = np.asarray(c, dtype=float)
        c_sub = c[list(dims)]
        dists[label] = _euclidean(pc_sub, c_sub)
    return dists


def _autumn_spring_ratio(d_autumn, d_spring):
    """
    Define a ratio r that we use for the rules.

    Example:
      r = d_spring / (d_autumn + d_spring)

    r ~ 0  => much closer to Autumn
    r ~ 1  => much closer to Spring
    """
    denom = d_autumn + d_spring
    if denom <= 0:
        return 0.5  # degenerate fallback
    return d_autumn / denom


def _apply_rule_table_for_ratio(r, table=AUTUMN_SPRING_RULE_TABLE):
    """
    Given ratio r and a rule table (list of (max_ratio, (A%,S%))),
    return (Autumn%, Spring%) according to the first row where r <= max_ratio.
    """
    for max_r, (a_perc, s_perc) in table:
        if r <= max_r:
            return {"Autumn": a_perc, "Spring": s_perc}
    # fallback if something is wrong
    max_r, (a_perc, s_perc) = table[-1]
    return {"Autumn": a_perc, "Spring": s_perc}


def _true_autumn_spring(perc_dict):
    """
    perc_dict: e.g. {"Autumn": 70, "North": 30, "Central": 0, "South": 0}

    Returns true Autumn/Spring distribution:
      true_A, true_S
    """
    a = float(perc_dict.get("Autumn", 0.0))
    spring = float(perc_dict.get("North", 0.0)) \
           + float(perc_dict.get("Central", 0.0)) \
           + float(perc_dict.get("South", 0.0))
    return a, spring


def _true_spring_subpops(perc_dict):
    """
    Extract true distribution within Spring (N,C,S), normalized to 100%.

    If total Spring = 0 → return all zeros.
    """
    n = float(perc_dict.get("North", 0.0))
    c = float(perc_dict.get("Central", 0.0))
    s = float(perc_dict.get("South", 0.0))
    total = n + c + s
    if total <= 0:
        return {"North": 0.0, "Central": 0.0, "South": 0.0}

    return {
        "North": 100.0 * n / total,
        "Central": 100.0 * c / total,
        "South": 100.0 * s / total,
    }


# ======================================================================
#  STEP 1: Autumn vs Spring – CENTRAL RULE FUNCTION FOR MIXEDNESS
# ======================================================================

def rule_step1_autumn_vs_spring(pc, season_to_centroid):
    """
    LOW-LEVEL rule for Autumn vs Spring (used by apply_step1_rules_for_haul).

    pc: 1D array, haul centroid in PC space.
    season_to_centroid: dict {"Autumn": cA, "Spring": cS}

    Returns a dict:
      {
        "pred_season_perc": {"Autumn": x, "Spring": y},  # 10% steps, according to rules
        "major_season": "Autumn" or "Spring",
        "mixed_season": bool,   # True if max < PURE_THRESHOLD_SEASON
        "dists": {"Autumn": dA, "Spring": dS},
        "ratio": r,             # helper value, can be interesting to inspect
      }
    """
    dists = _distances_to_centroids(pc, season_to_centroid)
    dA = dists["Autumn"]
    dS = dists["Spring"]

    r = _autumn_spring_ratio(dA, dS)
    pred_perc = _apply_rule_table_for_ratio(r, AUTUMN_SPRING_RULE_TABLE)
    major_season = max(pred_perc, key=pred_perc.get)
    max_frac = pred_perc[major_season] / 100.0
    mixed = max_frac < PURE_THRESHOLD_SEASON

    return {
        "pred_season_perc": pred_perc,
        "major_season": major_season,
        "mixed_season": mixed,
        "dists": dists,
        "ratio": r,
    }


def apply_step1_rules_for_haul(pc, true_mix, season_to_centroid):
    """
    *** STEP 1 – CENTRAL RULE FUNCTION ***

    This collects ALL hard-coded rules for:
      - Autumn vs Spring mixedness
      - Approx. A/S proportions
      - Which hauls are SPRING-RELEVANT (for STEP 2 later)

    This function is meant to be expanded when you add more
    rules/thresholds for Steps 2 and 3.

    Parameters
    ----------
    pc : np.ndarray
        The haul centroid coordinate in the REFERENCE PC space.

    true_mix : dict
        TRUE mix on A/N/C/S, e.g. {"Autumn": 50, "North": 0, "Central": 50, "South": 0}.

    season_to_centroid : dict
        {"Autumn": cA, "Spring": cS} from compute_season_centroids.

    Returns
    ----------
    res : dict with e.g.:
      - "true_mix_perc"
      - "true_season_perc"
      - "pred_season_perc"
      - "outer_category" ∈ {"pure_autumn", "pure_spring", "mixed"}
      - "mixed_season"  (True if outer mixed)
      - "spring_relevant" (True if haul can proceed to Spring PCA in Step 2)
      - "season_dists"
      - "season_ratio"
    """
    # TRUE Autumn vs Spring
    true_A, true_S = _true_autumn_spring(true_mix)
    true_season = {"Autumn": true_A, "Spring": true_S}

    # Low-level rule (ratio + table)
    step1_raw = rule_step1_autumn_vs_spring(pc, season_to_centroid)
    pred = step1_raw["pred_season_perc"]

    # Outer category based on predicted A/S distribution
    a_pred = pred["Autumn"] / 100.0
    s_pred = pred["Spring"] / 100.0

    if a_pred >= PURE_THRESHOLD_SEASON:
        outer_category = "pure_autumn"
    elif s_pred >= PURE_THRESHOLD_SEASON:
        outer_category = "pure_spring"
    else:
        outer_category = "mixed"

    mixed_season = (outer_category == "mixed")

    # Error in percentage points vs TRUE (for traceability)

    # Which hauls are Spring relevant? (for potential Step 2)
    spring_relevant = (
        s_pred >= SPRING_RELEVANT_MIN_SPRING_FRAC
        or a_pred <= SPRING_RELEVANT_MAX_AUTUMN_FRAC
    )

    res = {
        # ID/mix_name is added later in classify_batch_with_rules
        "true_mix_perc": true_mix,
        "true_season_perc": true_season,

        "pred_season_perc": pred,
        "outer_category": outer_category,   # "pure_autumn"/"pure_spring"/"mixed"
        "mixed_season": mixed_season,
        "spring_relevant": spring_relevant,
        "season_dists": step1_raw["dists"],
        "season_ratio": step1_raw["ratio"],
    }

    return res


# ===========================================================
#  STEP 2 & 3: within Spring  (NOT USED IN STEP 1 RIGHT NOW)
# ===========================================================
# These functions remain as tools for when you implement
# Spring PCA and inner Spring rules in Steps 2/3.

def rule_step2_and_3_within_spring(pc, pop_to_centroid, dims=DISTANCE_DIMS):
    """
    Rule for classification WITHIN Spring (N/C/S), based on r3d
    which is looked up in SPRING_R3D_RULE_TABLE.
    """
    pops = [p for p in ["North", "Central", "South"] if p in pop_to_centroid]
    if not pops:
        return {
            "r3d": {},
            "pred_sub_perc": {},
            "major_pop": None,
            "mixed_spring": False,
            "dists": {},
        }

    # 1) Distances in chosen PC subspace
    dists = _distances_to_centroids(
        pc,
        {p: pop_to_centroid[p] for p in pops},
        dims=dims,
    )
    vals = np.array([dists[p] for p in pops], dtype=float)

    # 2) Inverse distance → r3d
    eps = 1e-9
    inv = 1.0 / (vals + eps)
    weights = inv / inv.sum()

    r3d = {
        "North": float(weights[pops.index("North")]) if "North" in pops else 0.0,
        "Central": float(weights[pops.index("Central")]) if "Central" in pops else 0.0,
        "South": float(weights[pops.index("South")]) if "South" in pops else 0.0,
    }
    #### - HI, HERE You can add some extra rules if needed- BELOW IS AN EXAMPLE OF MORE SPECIALSED TUNING 

    # # --- EXTRA HAND-TUNED RULES ON r3d --------------------------
    # rN = r3d["North"]
    # rC = r3d["Central"]
    # rS = r3d["South"]

    # # Exempelregel: om Central är dominerande men South inte helt obetydlig,
    # # så vill du öka South lite och minska Central.
    # if rC > 0.6 and rS < 0.2:
    #     boost = 0.15  # 15%-enheter i r3d-världen
    #     rC = max(0.0, rC - boost)
    #     rS = min(1.0, rS + boost)

    # # Normalisera så att rN + rC + rS ≈ 1 igen
    # total = rN + rC + rS
    # if total > 0:
    #     rN /= total
    #     rC /= total
    #     rS /= total

    # r3d = {"North": rN, "Central": rC, "South": rS}
    # # ------------------------------------------------    

    ####
    # 3) Look up in the table – EXACTLY as in the AUTUMN_SPRING_RULE_TABLE case
    pred_sub_perc = _apply_r3d_rule_table(r3d)

    # 4) Majority class + mixedness
    major_pop = max(pred_sub_perc, key=pred_sub_perc.get)
    max_frac = pred_sub_perc[major_pop] / 100.0
    mixed_spring = max_frac < PURE_THRESHOLD_SPRING

    return {
        "r3d": r3d,
        "pred_sub_perc": pred_sub_perc,
        "major_pop": major_pop,
        "mixed_spring": mixed_spring,
        "dists": dists,
    }



# ===========================================================
#  STEP 1: MAIN CLASSIFICATION FOR A BATCH
# ===========================================================

def classify_batch_with_rules(ref_pca, PC_new, sim_info):

    season_to_centroid = compute_season_centroids(ref_pca)
    results = []

    for i, haul_info in enumerate(sim_info):
        pc = PC_new[i, :]
        haul_id = haul_info["haul_id"]
        mix_name = haul_info["mix_name"]
        true_mix = haul_info["perc"]

        step1 = apply_step1_rules_for_haul(pc, true_mix, season_to_centroid)

        res = {
            "haul_id": haul_id,
            "mix_name": mix_name,

            # Step-1 info (preds, mixedness, relevance etc.)
            **step1,

            # 🔥 Add index, this is absolutely the right place
            "index": i
        }

        results.append(res)

    return results



def run_rule_based_classification_on_batch(ref_pca, PC_new, sim_info):
    """
    STEP 1 – RUN RULES AND PRINT REPORT.

    This is only STEP 1 (outer mixedness & A/S proportions).
    Step 2 (Spring PCA) and Step 3 (N/C/S rules) will appear
    above with separate functions later.
    """
    results = classify_batch_with_rules(ref_pca, PC_new, sim_info)

    print("\n=== Step 1: Autumn vs Spring – rule-based mixedness for this batch ===")
    for r in results:
        haul_id = r["haul_id"]
        mix_name = r["mix_name"]

        true_mix = r["true_mix_perc"]
        true_season = r["true_season_perc"]

        pred_season = r["pred_season_perc"]
        outer_category = r["outer_category"]
        mixed_season = r["mixed_season"]
        spring_relevant = r["spring_relevant"]

        def fmt_perc(d):
            return ", ".join(f"{k}:{v:.0f}%" for k, v in d.items())

        def fmt_err(d):
            return ", ".join(f"{k}:±{v:.1f}p" for k, v in d.items())

        print(f"\nHaul {haul_id} ({mix_name})")

        # TRUE values (for traceability)
        print(f"  TRUE mix (A/N/C/S)        : {fmt_perc(true_mix)}")
        print(f"  TRUE Autumn vs Spring     : {fmt_perc(true_season)}")

        # Predictions from Step 1
        print(f"  PRED Autumn vs Spring     : {fmt_perc(pred_season)}")

        if outer_category == "pure_autumn":
            print("    → No mixedness detected. Over 90% likely Autumn.")
        elif outer_category == "pure_spring":
            print("    → No mixedness detected. Over 90% likely Spring.")
        else:
            print("    → Mixedness detected. (No side reaches 90%.)")

        print(f"    spring-relevant haul?   : {'YES' if spring_relevant else 'NO'}")

    # Here you can later add aggregates, e.g. list all spring-relevant hauls.

    return results


import numpy as np



from prompt import ask_yes_no  # keep the import

from prompt import ask_yes_no

from prompt import ask_yes_no  # once is enough

from prompt import ask_yes_no  # once is enough

SPRING_CLASSIFICATION_DIMS = (0, 1, 2)  # PC1/PC2/PC3 for Spring rules

def run_spring_detailed_classification(ref_pca, PC_new, sim_info, spring_results):
    """
    Step 2–3: detailed classification within Spring (N/C/S) based on r3d.

    Design:
      - Classification (r3d + rule table) is ALWAYS done in 3D: PC1/PC2/PC3.
      - Plotting can be done in 2D (PC1/PC2) or 3D (PC1/PC2/PC3) independently.
    """

    dims_class = SPRING_CLASSIFICATION_DIMS  # (0, 1, 2)

    print("\n=== Step 2–3: Detailed Spring classification (r3d-based, 3D) ===\n")

    pop_to_centroid = compute_population_centroids(ref_pca)

    spring_points = []   # full PC vectors for the sim hauls
    spring_labels = []   # e.g. "sim_000"
    spring_percs = []    # discrete N/C/S percentages (10%-steps) for labels

    for r in spring_results:
        idx = r["index"]
        haul_id = r["haul_id"]
        mix_name = r["mix_name"]
        true_mix = r["true_mix_perc"]

        H = PC_new[idx, :]  # full PC vector in reference PC space

        # --- 3D classification within Spring ---
        step2 = rule_step2_and_3_within_spring(
            pc=H,
            pop_to_centroid=pop_to_centroid,
            dims=dims_class,
        )

        r3d = step2["r3d"]
        pred_sub_perc = step2["pred_sub_perc"]
        major_pop = step2["major_pop"]
        mixed = step2["mixed_spring"]

        spring_points.append(H)
        spring_labels.append(haul_id)
        spring_percs.append(pred_sub_perc)

        # ---------- NICE, ALIGNED PRINT ----------
        print(f"Haul {haul_id} ({mix_name})")

        # TRUE Spring N/C/S (renormalised within Spring in your logic)
        print(
            "  TRUE Spring (N/C/S)        : "
            f"North:{true_mix.get('North', 0):.0f}%, "
            f"Central:{true_mix.get('Central', 0):.0f}%, "
            f"South:{true_mix.get('South', 0):.0f}%"
        )

        # Continuous weights
        print(
            "  r3d weights (3D, contin.)  : "
            f"North:{r3d.get('North', 0):.2f}, "
            f"Central:{r3d.get('Central', 0):.2f}, "
            f"South:{r3d.get('South', 0):.2f}"
        )

        # Discretised 10%-step prediction
        print(
            "  PRED Spring (N/C/S, 10%)   : "
            f"North:{pred_sub_perc.get('North', 0):.0f}%, "
            f"Central:{pred_sub_perc.get('Central', 0):.0f}%, "
            f"South:{pred_sub_perc.get('South', 0):.0f}%"
        )

        # Final line about mixedness / purity
        if major_pop is None:
            print("    → Cannot classify (missing Spring centroids).")
        elif mixed:
            print("    → Mixedness detected within Spring.")
        else:
            print(f"    → No mixedness detected. Likely {major_pop} (≥ 90%).")

        print()  # blank line between hauls
        # ----------------------------------------

    # ----- plotting section (unchanged in substance) -----
    if ask_yes_no("Do you want to plot Spring PCA with r3d classification?"):
        from pca_plotting import (
            plot_spring_classification_2d,
            plot_spring_classification_3d,
        )

        mode = input(
            "\nHow do you want to plot Spring PCA? "
            "[2 = 2D (PC1/PC2), 3 = 3D (PC1/PC2/PC3), default 2]: "
        ).strip()
        use_3d = (mode == "3")

        if use_3d:
            filename = "spring_r3d_classification_3d.png"
            plot_spring_classification_3d(
                ref_pca=ref_pca,
                pop_to_centroid=pop_to_centroid,
                spring_points=spring_points,
                spring_labels=spring_labels,
                spring_percs=spring_percs,
                dims=(0, 1, 2),
                savepath=filename,
                show=True,
            )
        else:
            filename = "spring_r3d_classification_2d.png"
            plot_spring_classification_2d(
                ref_pca=ref_pca,
                pop_to_centroid=pop_to_centroid,
                spring_points=spring_points,
                spring_labels=spring_labels,
                spring_percs=spring_percs,
                dims=(0, 1),
                savepath=filename,
                show=True,
            )

        print(f"  Saved Spring PCA plot to '{filename}'.")
