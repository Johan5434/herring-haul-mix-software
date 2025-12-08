# simulations/main_pca_mixes_experiments.py

import os
import numpy as np

from mix_selection import ask_for_mixes, simulate_mixes
from pca_hauls import (
    build_reference_pca_from_GM,
    project_individuals_to_PCs
)
from pca_plotting import (
    plot_pca_2d,
    plot_pca_3d,
    plot_classification_2d
)

from prompt import ask_yes_no, ask_experiment_action
from classification import (
    run_rule_based_classification_on_batch,
    compute_population_centroids,   # possibly kept if used elsewhere
    run_spring_detailed_classification,
)


def load_GM_npz(path: str = "GM_all_after_QC.npz"):
    """
    Load G_all / M_all from a previously saved npz.
    These should come from the main_simulation_motor pipeline.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find '{path}'. First run your full pipeline and save G_all/M_all."
        )
    data = np.load(path, allow_pickle=True)
    G_all = data["G_all"]
    M_all = data["M_all"]
    print(f"Loaded G_all/M_all from '{path}':")
    print(f"  G_all shape: {G_all.shape}")
    print(f"  M_all shape: {M_all.shape}")
    return G_all, M_all


def maybe_plot_reference_pca(ref_pca):
    """
    Ask if you want to see a 2D/3D plot of the reference map
    in this experiment session.
    """
    centroids = ref_pca["centroids"]
    haul_ids = ref_pca["haul_ids"]
    haul_pops = ref_pca["haul_pops"]

    do_2d = ask_yes_no("\nDo you want to plot 2D PCA (PC1 vs PC2) for haul centroids?")
    if do_2d:
        plot_pca_2d(
            centroids,
            haul_ids,
            haul_pops,
            pc_x=1,
            pc_y=2,
            title="Reference PCA – haul centroids (PC1 vs PC2)",
            show=True,
            savepath="reference_pca_hauls_2d_experiments.png",
        )
        print("  Also saved 2D plot to 'reference_pca_hauls_2d_experiments.png'.")

    if centroids.shape[1] >= 3:
        do_3d = ask_yes_no("Do you also want to plot 3D PCA (PC1, PC2, PC3)?")
        if do_3d:
            plot_pca_3d(
                centroids,
                haul_ids,
                haul_pops,
                title="Reference PCA – haul centroids (PC1, PC2, PC3)",
                show=True,
                savepath="reference_pca_hauls_3d_experiments.png",
            )
            print("  Also saved 3D plot to 'reference_pca_hauls_3d_experiments.png'.")
    else:
        print("Fewer than 3 PCA components – skipping 3D plot.")


def maybe_plot_simulated_batch(
    ref_pca,
    PC_new,
    sim_ids,
):
    """
    Ask if you want to plot the simulated haul centroids (PC_new)
    together with the reference haul centroids in the same PCA space.
    """
    centroids_ref = ref_pca["centroids"]
    haul_ids_ref = ref_pca["haul_ids"]
    haul_pops_ref = ref_pca["haul_pops"]

    # Combine reference + simulated
    all_coords = np.vstack([centroids_ref, PC_new])
    all_ids = np.concatenate([haul_ids_ref, np.array(sim_ids, dtype=object)])
    # Give all simulated hauls the same "population" tag, e.g. "SIM"
    sim_pops = np.array(["SIM"] * len(sim_ids), dtype=object)
    all_pops = np.concatenate([haul_pops_ref, sim_pops])

    # 2D plot (PC1 vs PC2)
    if ask_yes_no(
        "\nDo you want to plot these simulated hauls in PCA space (2D, PC1 vs PC2)?"
    ):
        plot_pca_2d(
            all_coords,
            all_ids,
            all_pops,
            pc_x=1,
            pc_y=2,
            title="Reference hauls + simulated hauls (PC1 vs PC2)",
            show=True,
            savepath="pca_mixes_with_ref_2d.png",  # overwritten each batch
        )
        print("  Saved 2D plot to 'pca_mixes_with_ref_2d.png'.")

    # 3D plot (PC1, PC2, PC3) – if there are at least 3 components
    if PC_new.shape[1] >= 3 and ask_yes_no(
        "Do you also want to plot 3D PCA (PC1, PC2, PC3) with simulated hauls?"
    ):
        plot_pca_3d(
            all_coords,
            all_ids,
            all_pops,
            title="Reference hauls + simulated hauls (PC1, PC2, PC3)",
            show=True,
            savepath="pca_mixes_with_ref_3d.png",  # overwritten each batch
        )
        print("  Saved 3D plot to 'pca_mixes_with_ref_3d.png'.")


# mapping from short code to population name
CODE_TO_POP = {
    "A": "Autumn",
    "N": "North",
    "C": "Central",
    "S": "South",
}


def _parse_pop_selection(pop_to_centroid):
    """
    Ask the user how the classification should be done:

      - 'd' or empty line: use all populations with centroids
      - e.g. 'A N'       : use only Autumn and North
      - e.g. 'A N C'     : Autumn, North, Central

    (Support for 'season' will come later, but we reserve the name.)
    """
    print("\n[Classification] Which populations should be included in the comparison?")
    print("  - 'd' or empty line: all (Autumn, North, Central, South if available)")
    print("  - e.g. 'A N'      : only Autumn vs North")
    print("  - e.g. 'A N C'    : Autumn, North, Central")
    choice = input("  Choice (default: d): ").strip()

    if not choice or choice.lower() == "d":
        # all populations that exist in centroids
        pops = [p for p in ["Autumn", "North", "Central", "South"]
                if p in pop_to_centroid]
        if not pops:
            pops = sorted(pop_to_centroid.keys())
        return pops

    tokens = choice.replace(",", " ").split()
    pops = []
    for t in tokens:
        t_upper = t.upper()
        if t_upper in CODE_TO_POP:
            pop_name = CODE_TO_POP[t_upper]
            if pop_name in pop_to_centroid:
                pops.append(pop_name)
            else:
                print(f"  Warning: '{pop_name}' does not exist in the reference, skipping.")
        else:
            print(f"  Unknown code '{t}', skipping.")

    if not pops:
        print("  No valid populations selected, using default (all).")
        pops = [p for p in ["Autumn", "North", "Central", "South"]
                if p in pop_to_centroid]
        if not pops:
            pops = sorted(pop_to_centroid.keys())

    return pops


def run_centroid_classification_experiment(ref_pca, PC_new, sim_info):
    """
    Run centroid-distance-based classification for a batch of simulated hauls.

    - Ask which populations should be included
    - Compute predicted mixes per haul
    - Print short text for correct class, more details for misclassified
    """
    # we only need population centroids here → the classification module does the rest

    pop_to_centroid = compute_population_centroids(ref_pca)
    pops = _parse_pop_selection(pop_to_centroid)

    print("\nClassifying batch based on centroid distances...")
    results = run_rule_based_classification_on_batch(
        ref_pca=ref_pca,
        PC_new=PC_new,
        sim_info=sim_info,
        pops=pops,
        step=10,   # 10% steps, as described
    )

    print("\nClassification results (rounded to 10% steps):")
    for res in results:
        haul_id = res["haul_id"]
        mix_name = res["mix_name"]
        true_step = res["true_step"]
        pred_step = res["pred_step"]
        match = res["match"]
        dists = res["dists"]

        # format percentage string
        def fmt_perc(d):
            return ", ".join(f"{pop}:{p}%" for pop, p in d.items())

        if match:
            # minimal text
            print(
                f"  {haul_id} ({mix_name}) → "
                f"{fmt_perc(pred_step)}  [OK: matches true mix]"
            )
        else:
            # more detail, incl. true mix + distances
            print(
                f"  {haul_id} ({mix_name}) → MISCLASSIFIED"
            )
            print(f"    True mix     : {fmt_perc(true_step)}")
            print(f"    Predicted    : {fmt_perc(pred_step)}")

            dist_str = ", ".join(f"{pop}:{dists[pop]:.3f}" for pop in res["pops"])
            print(f"    Distances    : {dist_str}")
    if ask_yes_no("\nDo you want to plot the classification in PCA space (with lines)?"):
        plot_classification_2d(
            ref_pca=ref_pca,
            PC_new=PC_new,
            results=results,
            pop_to_centroid=pop_to_centroid,
            pc_x=1,
            pc_y=2,
            title="Classification: simulated hauls + population centroids (PC1 vs PC2)",
            show=True,
            savepath="classification_2d.png",
        )
        print("  Also saved classification plot to 'classification_2d.png'.")


def run_mix_experiment_loop(G_all, M_all, ref_pca):
    """
    Interactive loop for testing many different mixes and projecting them
    into the same PCA space that the individual PCA is trained on.

    For each batch:
      - the user specifies mixes
      - simulate individuals
      - project to PC space
      - compute centroid per simulated haul
      - then an inner menu:
          [1] plot simulated hauls + reference
          [2] STEP 1: Autumn/Spring rule + (optional) STEP 2–3 Spring detail
          [3] new batch
          [q] exit everything
    """

    while True:
        print("\n=== New batch of mixes ===")
        mixes = ask_for_mixes()  # 'q' or empty line → abort

        if not mixes:
            print("No mixes specified. Aborting experiment loop.")
            break

        n_individuals = int(
            input("Number of individuals per simulated haul (e.g. 30): ").strip()
        )

        # 1) simulate hauls for these mixes
        sim_info = simulate_mixes(
            G_all,
            M_all,
            mixes,
            n_individuals=n_individuals,
            seed=42,
        )

        # 2) project individuals for each simulated haul → centroid in PC space
        centroids_sim = []
        sim_ids = []
        sim_mix_names = []
        sim_perc = []

        for haul_info in sim_info:
            G_sub = haul_info["G_sub"]  # (n_ind_in_haul, n_snps)

            PCs_ind_sim = project_individuals_to_PCs(
                G_sub,
                ref_pca["mean_snps"],
                ref_pca["std_snps"],
                ref_pca["valid_snps"],
                ref_pca["components"],
            )

            centroid = PCs_ind_sim.mean(axis=0)
            centroids_sim.append(centroid)

            sim_ids.append(haul_info["haul_id"])
            sim_mix_names.append(haul_info["mix_name"])
            sim_perc.append(haul_info["perc"])

        PC_new = np.vstack(centroids_sim)  # (n_sim_hauls, n_components)

        # 3) short summary in the terminal
        print("\nResults for this batch (centroids in PC space):")
        for i, mix_name in enumerate(sim_mix_names):
            perc_str = ", ".join(f"{pop}:{p:.0f}%" for pop, p in sim_perc[i].items())
            coords_str = ", ".join(
                f"PC{d+1}={PC_new[i, d]:.3f}" for d in range(min(PC_new.shape[1], 3))
            )
            print(f"  {sim_ids[i]} ({mix_name}) [{perc_str}] → {coords_str}")

        # 4) inner menu for this batch
        while True:
            choice = ask_experiment_action()  # "1", "2", "3", "q"

            if choice == "1":
                # Plot reference + simulated hauls (2D/3D)
                maybe_plot_simulated_batch(ref_pca, PC_new, sim_ids)

            elif choice == "2":
                # ==========================
                # STEP 1: Autumn/Spring rules
                # ==========================
                results = run_rule_based_classification_on_batch(
                    ref_pca,
                    PC_new,
                    sim_info,
                )

                # Extract spring-relevant hauls (according to your rules)
                spring_results = [
                    r for r in results if r.get("spring_relevant", False)
                ]

                if spring_results:
                    print(
                        "\nThese hauls are spring-relevant and can be further analysed "
                        "within Spring (North/Central/South):"
                    )
                    for r in spring_results:
                        print(f"  - {r['haul_id']} ({r['mix_name']})")

                    if ask_yes_no(
                        "\nDo you want to continue with detailed Spring classification "
                        "using r3d rules (steps 2–3)?"
                    ):
                        # Steps 2–3: r3d-based Spring analysis (incl. choice of PC dims + plotting)
                        run_spring_detailed_classification(
                            ref_pca,
                            PC_new,
                            sim_info,
                            spring_results,
                        )
                    else:
                        print("Skipping detailed Spring classification.")
                else:
                    print(
                        "\n(No Spring-relevant hauls according to step-1 rules – "
                        "no further Spring analysis possible.)"
                    )

            elif choice == "3":
                print("Proceeding to next batch of mixes.")
                break  # out of inner while → back to outer while (new batch)

            elif choice == "q":
                print("Exiting experiment loop.")
                return

        # end of inner while → continue with new batch

    print("Experiment loop finished.")


def main():
    print("=== PCA mix experiments (without QC/VCF) ===\n")
    default_npz = "GM_all_after_QC.npz"
    npz_path = input(
        f"Enter npz file with G_all/M_all (default {default_npz}): "
    ).strip() or default_npz

    G_all, M_all = load_GM_npz(npz_path)

    # Build reference PCA once via pca_hauls.build_reference_pca_from_GM
    ref_pca = build_reference_pca_from_GM(G_all, M_all, n_components=3)

    # Print explained variance (same style as before)
    var_ratio = ref_pca["explained_var_ratio"]
    print("\nReference PCA (individual level) – explained variance:")
    for i, v in enumerate(var_ratio[:3]):
        print(f"  PC{i+1}: {v*100:.1f}%")

    print("Haul centroid matrix shape:", ref_pca["centroids"].shape)

    # Optionally plot the reference map
    maybe_plot_reference_pca(ref_pca)

    # Ask if you want to enter the mix experiment loop
    if ask_yes_no("\nDo you want to continue and test mix simulations in this PCA space?"):
        run_mix_experiment_loop(G_all, M_all, ref_pca)
    else:
        print("OK, only the reference PCA was run in this session.")


if __name__ == "__main__":
    main()
