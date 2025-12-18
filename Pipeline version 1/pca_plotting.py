# simulations/pca_plotting.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D

# colors per population – adjust if you want
POP_COLORS = {
    "Autumn": "#B557D1",  # Medium purple
    "North": "#5E89B8",   # Medium deep blue
    "Central": "#E07550", # Medium orange-red
    "South": "#66B366",   # Medium vibrant green
}


def plot_pca_2d(PC_ref, haul_ids, haul_pops,
                pc_x=1, pc_y=2,
                title="Reference PCA (hauls)",
                show=True,
                savepath=None):
    """
    Simple 2D plot of reference PCA at haul level.

    PC_ref: (n_hauls, k)
    haul_ids: list[str]
    haul_pops: list[str]
    """
    pc_x -= 1  # convert 1-based -> 0-based
    pc_y -= 1

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot per population for a nice legend
    unique_pops = sorted(set(haul_pops))
    for pop in unique_pops:
        mask = [p == pop for p in haul_pops]
        xs = PC_ref[mask, pc_x]
        ys = PC_ref[mask, pc_y]
        color = POP_COLORS.get(pop, "gray")
        ax.scatter(xs, ys, label=pop, alpha=0.8, edgecolor="k", s=40, c=color)

    ax.set_xlabel(f"PC{pc_x+1}")
    ax.set_ylabel(f"PC{pc_y+1}")
    ax.set_title(title)
    ax.legend(title="Population", fontsize=9)
    ax.grid(True, alpha=0.3)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pca_3d(PC_ref, haul_ids, haul_pops,
                title="Reference PCA (hauls, 3D)",
                show=True,
                savepath=None):
    """
    3D plot of reference PCA at haul level (PC1, PC2, PC3).
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    unique_pops = sorted(set(haul_pops))
    for pop in unique_pops:
        mask = [p == pop for p in haul_pops]
        xs = PC_ref[mask, 0]
        ys = PC_ref[mask, 1]
        zs = PC_ref[mask, 2]
        color = POP_COLORS.get(pop, "gray")
        ax.scatter(xs, ys, zs, label=pop, alpha=0.8, edgecolor="k", s=40, c=color)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(title="Population", fontsize=9)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_classification_2d(
    ref_pca,
    PC_new,
    results,
    pop_to_centroid,
    pc_x=1,
    pc_y=2,
    title="Classification view (PC1 vs PC2)",
    show=True,
    savepath=None,
):
    """
    2D plot for classification mode:

      - Reference haul centroids in the background (colored by population)
      - Population centroids as large X markers
      - Simulated hauls as stars (colored by dominant predicted population)
      - Dashed lines from each simulated haul to all used population centroids

    Parameters
    ----------
    ref_pca : dict
        From build_reference_pca_from_GM, must contain:
          - "centroids" (n_hauls, k)
          - "haul_pops" (n_hauls,)

    PC_new : np.ndarray
        Simulated haul centroids in PC space, shape (n_sim, k).

    results : list of dicts
        From classification.classify_batch. Each element must have:
          - "haul_id"
          - "pops"       (list with population names used)
          - "pred_step"  (dict pop->percent)
          - "match"      (bool)
          - "dists"      (dict pop->dist)
        Assumed to have the same order as the rows in PC_new.

    pop_to_centroid : dict
        {pop_name: centroid_vector} for all populations used.

    pc_x, pc_y : int
        Which PC axes (1-based) to plot on x and y, respectively.
    """
    import numpy as np

    if PC_new is None or len(PC_new) == 0:
        # no simulated hauls → just plot the reference?
        print("plot_classification_2d: no simulated hauls, skipping lines.")
    PC_new = np.asarray(PC_new, dtype=float)

    pc_x -= 1  # 1-based -> 0-based
    pc_y -= 1

    centroids_ref = ref_pca["centroids"]
    haul_pops_ref = ref_pca["haul_pops"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- 1) Reference haul centroids (background) -----------------------
    unique_pops = sorted(set(haul_pops_ref))
    for pop in unique_pops:
        mask = [p == pop for p in haul_pops_ref]
        xs = centroids_ref[mask, pc_x]
        ys = centroids_ref[mask, pc_y]
        color = POP_COLORS.get(pop, "gray")
        ax.scatter(
            xs,
            ys,
            label=f"{pop} hauls",
            alpha=0.3,
            edgecolor="none",
            s=25,
            c=color,
        )

    # --- 2) Population centroids (big X markers) ------------------------
    # take populations from the first result (all have the same list)
    if results:
        pops_used = results[0]["pops"]
    else:
        pops_used = []

    for pop in pops_used:
        if pop not in pop_to_centroid:
            continue
        c = pop_to_centroid[pop]
        x = c[pc_x]
        y = c[pc_y]
        color = POP_COLORS.get(pop, "gray")
        ax.scatter(
            [x],
            [y],
            marker="X",
            s=160,
            c=color,
            edgecolor="k",
            linewidth=1.0,
            label=f"{pop} centroid",
        )

    # --- 3) Simulated hauls (stars, colored by dominant predicted pop) ---
    if len(PC_new) > 0 and results:
        # group by dominant predicted population
        sim_by_pop = {pop: {"x": [], "y": [], "ids": []} for pop in pops_used}

        for point, res in zip(PC_new, results):
            pred_step = res["pred_step"]
            # dominant predicted population
            main_pop = max(pred_step, key=lambda k: pred_step[k])
            if main_pop not in sim_by_pop:
                sim_by_pop[main_pop] = {"x": [], "y": [], "ids": []}
            sim_by_pop[main_pop]["x"].append(point[pc_x])
            sim_by_pop[main_pop]["y"].append(point[pc_y])
            sim_by_pop[main_pop]["ids"].append(res["haul_id"])

        for pop, data in sim_by_pop.items():
            if not data["x"]:
                continue
            color = POP_COLORS.get(pop, "black")
            ax.scatter(
                data["x"],
                data["y"],
                marker="*",
                s=120,
                c=color,
                edgecolor="k",
                linewidth=0.5,
                label=f"Sim hauls (pred {pop})",
            )

        # --- 4) Dashed lines from each sim haul to each pop centroid -----
        for point, res in zip(PC_new, results):
            for pop in res["pops"]:
                if pop not in pop_to_centroid:
                    continue
                c = pop_to_centroid[pop]
                ax.plot(
                    [point[pc_x], c[pc_x]],
                    [point[pc_y], c[pc_y]],
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.3,
                    color="k",
                )

    ax.set_xlabel(f"PC{pc_x+1}")
    ax.set_ylabel(f"PC{pc_y+1}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # merge legend entries (there can be many, but it's informative)
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates via a simple dict
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


import numpy as np

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # already imported at the top


def plot_spring_classification_2d(
    ref_pca,
    pop_to_centroid,
    spring_points,
    spring_labels,
    spring_percs,
    dims=(0, 1),
    title="Spring-PCA r3d classification (PC1 vs PC2)",
    show=True,
    savepath=None,
):
    """
    2D plot for Spring r3d classification:

      - All Spring hauls (North/Central/South) as small points.
      - Population centroids (N/C/S) as large circles.
      - Simulated hauls as black stars with text labels (N/C/S %).
      - Pink dashed lines from each simulated haul to the N/C/S centroids.
    """
    centroids_ref = ref_pca["centroids"]
    haul_pops_ref = ref_pca["haul_pops"]

    ix, iy = dims[0], dims[1]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1) All Spring hauls (N/C/S) as small points
    for pop in ["North", "Central", "South"]:
        mask = (haul_pops_ref == pop)
        if not np.any(mask):
            continue
        xs = centroids_ref[mask, ix]
        ys = centroids_ref[mask, iy]
        color = POP_COLORS.get(pop, "gray")
        ax.scatter(
            xs,
            ys,
            s=15,
            alpha=0.3,
            c=color,
            edgecolor="none",
            label=f"{pop} haul",
        )

    # 2) Population centroids (large circles)
    for pop in ["North", "Central", "South"]:
        if pop not in pop_to_centroid:
            continue
        c = pop_to_centroid[pop]
        color = POP_COLORS.get(pop, "gray")
        x = c[ix]
        y = c[iy]
        ax.scatter(
            [x],
            [y],
            s=200,
            marker="o",
            edgecolor="k",
            linewidth=1.2,
            c=color,
            label=f"{pop} centroid",
        )

    # 3) Simulated Spring hauls (stars + text)
    for point, label, perc in zip(spring_points, spring_labels, spring_percs):
        x = point[ix]
        y = point[iy]
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=140,
            edgecolor="k",
            linewidth=0.8,
            c="black",
            label=f"{label} (sim)",
        )
        txt = (
            f"{label}\n"
            f"N:{perc.get('North', 0):.0f}% "
            f"C:{perc.get('Central', 0):.0f}% "
            f"S:{perc.get('South', 0):.0f}%"
        )
        ax.text(x, y, txt, fontsize=8)

        # 4) Pink dashed lines from sim haul to each N/C/S centroid
        for pop in ["North", "Central", "South"]:
            if pop not in pop_to_centroid:
                continue
            c = pop_to_centroid[pop]
            cx = c[ix]
            cy = c[iy]
            ax.plot(
                [x, cx],
                [y, cy],
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                color="hotpink",
            )

    ax.set_xlabel(f"PC{ix+1}")
    ax.set_ylabel(f"PC{iy+1}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_spring_classification_3d(
    ref_pca,
    pop_to_centroid,
    spring_points,
    spring_labels,
    spring_percs,
    dims=(0, 1, 2),
    title="Spring-PCA r3d classification (3D)",
    show=True,
    savepath=None,
):
    """
    3D plot for Spring r3d classification:

      - Spring hauls as small points.
      - Population centroids (N/C/S) as large circles.
      - Simulated hauls as black stars + text.
      - Pink dashed lines from each simulated haul to N/C/S centroids.
    """
    centroids_ref = ref_pca["centroids"]
    haul_pops_ref = ref_pca["haul_pops"]

    ix, iy, iz = dims[0], dims[1], dims[2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 1) Spring hauls (N/C/S) as small points
    for pop in ["North", "Central", "South"]:
        mask = (haul_pops_ref == pop)
        if not np.any(mask):
            continue
        xs = centroids_ref[mask, ix]
        ys = centroids_ref[mask, iy]
        zs = centroids_ref[mask, iz]
        color = POP_COLORS.get(pop, "gray")
        ax.scatter(
            xs,
            ys,
            zs,
            s=15,
            alpha=0.3,
            c=color,
            edgecolor="none",
            label=f"{pop} haul",
        )

    # 2) Population centroids
    for pop in ["North", "Central", "South"]:
        if pop not in pop_to_centroid:
            continue
        c = pop_to_centroid[pop]
        color = POP_COLORS.get(pop, "gray")
        x = c[ix]
        y = c[iy]
        z = c[iz]
        ax.scatter(
            [x],
            [y],
            [z],
            s=200,
            marker="o",
            edgecolor="k",
            linewidth=1.2,
            c=color,
            label=f"{pop} centroid",
        )

    # 3) Simulated Spring hauls
    for point, label, perc in zip(spring_points, spring_labels, spring_percs):
        x = point[ix]
        y = point[iy]
        z = point[iz]
        ax.scatter(
            [x],
            [y],
            [z],
            marker="*",
            s=140,
            edgecolor="k",
            linewidth=0.8,
            c="black",
            label=f"{label} (sim)",
        )
        txt = (
            f"{label}\n"
            f"N:{perc.get('North', 0):.0f}% "
            f"C:{perc.get('Central', 0):.0f}% "
            f"S:{perc.get('South', 0):.0f}%"
        )
        ax.text(x, y, z, txt, fontsize=8)

        # Pink dashed lines to N/C/S centroids
        for pop in ["North", "Central", "South"]:
            if pop not in pop_to_centroid:
                continue
            c = pop_to_centroid[pop]
            cx = c[ix]
            cy = c[iy]
            cz = c[iz]
            ax.plot(
                [x, cx],
                [y, cy],
                [z, cz],
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                color="hotpink",
            )

    ax.set_xlabel(f"PC{ix+1}")
    ax.set_ylabel(f"PC{iy+1}")
    ax.set_zlabel(f"PC{iz+1}")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8)

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
