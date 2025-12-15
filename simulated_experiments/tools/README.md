Tools (optional utilities)

This folder contains helper scripts that are useful for diagnostics and analysis, but are not required for the core pipeline.

- evaluate_pc_dimensions.py: Compares Step 1 (Autumn vs Spring) classification accuracy when using 2 PCs vs 3 PCs in the Reference PCA. Run from simulated_experiments.

  Example:
  
  - source ../.venv/bin/activate
  - python tools/evaluate_pc_dimensions.py

  Notes: Reads reference_model/reference_pca.json and runs test_simulated_hauls.py.

- evaluate_spring_pca_dimensions.py: Compares Step 2 (N/C/S) classification accuracy using 2 vs 3 PCs in the Spring PCA, and reports Spring PCA explained variance.

  Example:
  
  - source ../.venv/bin/activate
  - python tools/evaluate_spring_pca_dimensions.py

  Notes: Temporarily toggles SPRING_CLASSIFICATION_DIMS in simulations/classification.py and runs test_simulated_hauls.py on Spring-relevant hauls.

- debug_distances.py: Prints Autumn vs Spring predictions for a given predictions CSV to inspect non-zero Autumn on Spring-only hauls.

  Example:
  
  - source ../.venv/bin/activate
  - python tools/debug_distances.py

  Notes: Update the predictions_file variable inside the script to point to the CSV you want to inspect.
