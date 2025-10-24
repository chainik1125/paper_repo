## Simplex MATS Work Sample (Cartesian Belief Geometry)

This repository contains the work sample delivered for the **Simplex MATS** stream. It reuses the original paper codebase with the minimal modifications needed to evaluate tensor-factor regressions on the MM3×Bloch cartesian product model.

The main entry point for reviewers is the notebook:

```
notebooks/cartesian_analysis.ipynb
```

Running that notebook (from the repo root, with WandB credentials configured) reproduces the key diagnostics:

1. Fig. 2-style visualisation of the MM3 belief geometry recovered from the cartesian model.
2. Mirror visualisation for the Bloch belief geometry.
3. The full regression benchmark (factorised, augmented, operator-Schmidt, direct, and random baseline), producing the JSON metrics used in the tables.

Outside that notebook, the repository remains close to the original paper repo—only the pieces required for the analysis (e.g. `fig2_combined_bloch.py`, `plot_mm3_svd_component.py`) were modified.
