# simulations/mix_selection.py

import numpy as np


POPS_ORDER = ["Autumn", "North", "Central", "South"]


def ask_for_mixes():
    """
    Let the user input one or more mixes in percent for
    [Autumn, North, Central, South].

    Example input:
        20 20 0 60
        10 40 0 50
        q

    or simply:
        100 0 0 0
        q

    Requirements:
      - exactly 4 numbers per line
      - sum to 100
      - (recommended) multiples of 10, since we use 0..10 steps.

    Returns:
      mixes: list of dicts, e.g.
        {
          "name": "mix_20A_20N_0C_60S",
          "perc": {"Autumn": 20, "North": 20, "Central": 0, "South": 60},
          "steps": {"Autumn": 2, "North": 2, "Central": 0, "South": 6},
        }
    """
    print("=== Enter mixes (percent for Autumn, North, Central, South) ===")
    print("Write four numbers per line, e.g. '20 20 0 60'.")
    print("Finish with 'q' or an empty line.\n")

    mixes = []

    while True:
        line = input("Mix (A N C S) or 'q': ").strip()
        if not line or line.lower() == "q":
            break

        parts = line.replace(",", " ").split()
        if len(parts) != 4:
            print("  -> Please enter exactly 4 numbers (A N C S). Try again.")
            continue

        try:
            a_perc, n_perc, c_perc, s_perc = map(float, parts)
        except ValueError:
            print("  -> Could not interpret all values as numbers. Try again.")
            continue

        total = a_perc + n_perc + c_perc + s_perc
        if abs(total - 100.0) > 1e-6:
            print(f"  -> The sum must be 100. It was {total:.1f}. Try again.")
            continue

        # Convert to "steps" (0..10) – here we assume multiples of 10
        steps_a = a_perc / 10.0
        steps_n = n_perc / 10.0
        steps_c = c_perc / 10.0
        steps_s = s_perc / 10.0

        # check that we end up at integers (multiples of 10 %)
        steps_arr = np.array([steps_a, steps_n, steps_c, steps_s])
        if not np.allclose(steps_arr, np.round(steps_arr)):
            print("  -> Currently we require multiples of 10% (e.g. 0, 10, 20, ...). Try again.")
            continue

        steps_a, steps_n, steps_c, steps_s = map(int, np.round(steps_arr))
        if steps_a + steps_n + steps_c + steps_s != 10:
            print("  -> Internal steps must sum to 10. Something went wrong. Try again.")
            continue

        perc_dict = {
            "Autumn": a_perc,
            "North": n_perc,
            "Central": c_perc,
            "South": s_perc,
        }
        steps_dict = {
            "Autumn": steps_a,
            "North": steps_n,
            "Central": steps_c,
            "South": steps_s,
        }

        name = f"mix_{int(a_perc)}A_{int(n_perc)}N_{int(c_perc)}C_{int(s_perc)}S"

        mixes.append(
            {
                "name": name,
                "perc": perc_dict,
                "steps": steps_dict,
            }
        )

        print(f"  -> Added mix: {name}")

    print(f"\nTotal number of mixes: {len(mixes)}")
    return mixes


def _counts_from_steps(steps, n_individuals):
    """
    Internal helper:
      steps: list/array with integers in [0,10] for [A, N, C, S] where the sum = 10
      n_individuals: number of individuals in the haul

    Returns:
      counts: np.array with number of individuals per population (A,N,C,S)
              that sums to n_individuals.
    """
    steps_arr = np.array(steps, dtype=float)
    fractions = steps_arr / 10.0
    raw_counts = fractions * n_individuals
    counts = np.floor(raw_counts).astype(int)
    rest = int(n_individuals - counts.sum())

    if rest > 0:
        frac_parts = raw_counts - counts
        order = np.argsort(-frac_parts)  # largest remaining fraction first
        for k in order:
            if rest == 0:
                break
            if steps_arr[k] > 0:
                counts[k] += 1
                rest -= 1

    return counts


def simulate_single_haul(G, M, mix, n_individuals, pops_order=None, rng=None):
    """
    Simulate ONE haul based on G, M and a mix structure.

    Parameters
    ----------
    G : np.ndarray
        Genotype matrix (n_samples, n_snps).

    M : np.ndarray
        Metadata matrix (n_samples, 3): [sample_id, haul, population].

    mix : dict
        Mix structure from ask_for_mixes, e.g.:
          {
            "name": "mix_20A_20N_0C_60S",
            "perc": {...},
            "steps": {"Autumn":2, "North":2, "Central":0, "South":6},
          }

    n_individuals : int
        Number of individuals in this simulated haul.

    pops_order : sequence of str, optional
        Order of populations, default = ["Autumn","North","Central","South"].

    rng : np.random.Generator or None
        If you want to control randomness via an RNG object; otherwise np.random is used.

    Returns
    -------
    haul_info : dict with at least:
      {
        "haul_id": "sim_000",
        "mix_name": mix["name"],
        "perc": mix["perc"],
        "steps": mix["steps"],
        "indices": np.array([...]),   # indices into original G/M
        "G_sub": G_sub,               # (n_individuals, n_snps)
        "M_sub": M_sub,               # (n_individuals, 3)
      }
    """
    if pops_order is None:
        pops_order = POPS_ORDER

    if rng is None:
        rng = np.random

    pops = M[:, 2]
    pop_to_idx = {pop: np.where(pops == pop)[0] for pop in pops_order}

    # steps in correct order (A,N,C,S)
    steps = [mix["steps"].get(pop, 0) for pop in pops_order]
    counts = _counts_from_steps(steps, n_individuals)
    # counts[0] = Autumn, counts[1] = North, etc.

    chosen_indices = []

    for pop_name, count in zip(pops_order, counts):
        if count <= 0:
            continue
        idx_pool = pop_to_idx.get(pop_name, None)
        if idx_pool is None or idx_pool.size == 0:
            # if the mix requires a population that does not exist in the data → raise
            raise RuntimeError(
                f"Mix {mix['name']} requires population '{pop_name}', "
                "but no individuals were found in M."
            )
        chosen_indices.extend(
            rng.choice(idx_pool, size=count, replace=True)
        )

    chosen_indices = np.array(chosen_indices)

    if chosen_indices.size != n_individuals:
        raise RuntimeError(
            f"Error in simulate_single_haul: selected {chosen_indices.size} individuals, "
            f"but n_individuals = {n_individuals}."
        )

    # submatrices
    G_sub = G[chosen_indices].copy()
    M_sub = M[chosen_indices].copy()

    # haul_id is set by the caller (simulate_mixes), so here only partial info:
    haul_info = {
        "haul_id": None,  # set later in simulate_mixes
        "mix_name": mix["name"],
        "perc": mix["perc"],
        "steps": mix["steps"],
        "indices": chosen_indices,
        "G_sub": G_sub,
        "M_sub": M_sub,
    }

    return haul_info


def simulate_mixes(G, M, mixes, n_individuals, pops_order=None, seed=42):
    """
    Simulate one haul per mix in 'mixes', based on G and M.

    Parameters
    ----------
    G, M : see simulate_single_haul.

    mixes : list of mix dicts from ask_for_mixes.

    n_individuals : int
        Number of individuals per simulated haul.

    pops_order : sequence of str or None
        Order of populations, default = ["Autumn","North","Central","South"].

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sim_info : list of dicts, e.g.
      [
        {
          "haul_id": "sim_000",
          "mix_name": "...",
          "perc": {...},
          "steps": {...},
          "indices": np.array([...]),
          "G_sub": G_sub,
          "M_sub": M_sub,
        },
        ...
      ]
    """
    if pops_order is None:
        pops_order = POPS_ORDER

    rng = np.random.default_rng(seed)
    sim_info = []

    for i, mix in enumerate(mixes):
        haul_info = simulate_single_haul(
            G, M, mix, n_individuals, pops_order=pops_order, rng=rng
        )
        haul_info["haul_id"] = f"sim_{i:03d}"
        sim_info.append(haul_info)

    print(f"Simulated {len(sim_info)} hauls from {len(mixes)} mixes.")
    return sim_info
