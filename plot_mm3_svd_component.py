"""
Visualise how the top Kronecker-SVD component acts on MM3 belief geometry.

This script reuses the regression utilities from fig2_combined_bloch.py to
download a cartesian-product checkpoint, fit the MM3/Bloch regressors, obtain
the augmented Kronecker coefficient matrix, and then extract the first (or
user-specified) Kronecker component from its SVD.  Keeping the Bloch belief
fixed to token index 0 (or a user-specified index), it maps the MM3 belief
states through that component and plots the resulting geometry alongside the
ground-truth MM3 beliefs.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import mean_squared_error, r2_score

from fig2_combined_bloch import (
    download_artifact,
    instantiate_model,
    prepare_process_data,
    fit_weighted_ridge,
    predict_with_regressor,
    sample_sequences,
    build_final_belief_lookup,
)
from minimal_impl.cartesian import _infer_vocab_size
from minimal_impl.mm3 import generate_mm3_transformer_data
from minimal_impl.bloch import generate_bloch_transformer_data
from minimal_impl.regression import deduplicate_tensor, _combine_layer_activations
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS
from scripts.activation_analysis.data_loading import ActivationExtractor


def project_to_simplex_3(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D probability vectors onto a 2D simplex."""
    assert points.shape[1] == 3
    x = points[:, 1] + 0.5 * points[:, 2]
    y = np.sqrt(3) / 2 * points[:, 2]
    return x, y


def compute_component_geometry(
    artifact_path: str,
    component_idx: int | str,
    bloch_index: int,
    samples: int,
    ridge_alpha: float | None,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ground_truth_mm3, component_mm3) belief arrays."""
    with tempfile.TemporaryDirectory(prefix="mm3_svd_") as tmpdir:
        workdir = Path(tmpdir)
        checkpoint_path, run_cfg = download_artifact(artifact_path, workdir)

        device_t = torch.device(device)
        model = instantiate_model(checkpoint_path, run_cfg, device)
        extractor = ActivationExtractor(device=device_t)

        process_cfg = run_cfg.get("process_config", {})
        if process_cfg.get("mode") != "mm3_bloch_product":
            raise ValueError("Run config does not describe an mm3_bloch_product experiment.")

        mm3_params = dict(process_cfg["mm3"])
        bloch_params = dict(process_cfg["bloch"])

        base_config = {
            "model_config": dict(run_cfg["model_config"]),
            "train_config": dict(run_cfg.get("train_config", {})),
        }
        base_config["model_config"].setdefault("n_ctx", base_config["model_config"].get("n_ctx"))

        mm3_data = prepare_process_data(base_config, "mess3", mm3_params)
        bloch_data = prepare_process_data(base_config, "tom_quantum", bloch_params)

        mm3_dataset = generate_mm3_transformer_data(
            n_ctx=base_config["model_config"]["n_ctx"],
            bos=base_config["train_config"].get("bos", False),
            device="cpu",
            as_numpy=False,
            **mm3_params,
        )
        bloch_dataset = generate_bloch_transformer_data(
            n_ctx=base_config["model_config"]["n_ctx"],
            bos=base_config["train_config"].get("bos", False),
            device="cpu",
            as_numpy=False,
            **bloch_params,
        )

        vocab_bloch = _infer_vocab_size(bloch_dataset.transformer_inputs)

        mm3_base_idx = torch.argmax(mm3_dataset.probabilities).item()
        bloch_base_idx = torch.argmax(bloch_dataset.probabilities).item()
        mm3_base_seq = mm3_dataset.transformer_inputs[mm3_base_idx, :-1].to(device_t)
        bloch_base_seq = bloch_dataset.transformer_inputs[bloch_base_idx, :-1].to(device_t)

        combined_mm3_inputs = mm3_data.nn_inputs.to(device_t) * vocab_bloch + bloch_base_seq
        cache_mm3 = extractor.extract_activations(
            model,
            combined_mm3_inputs.long(),
            "transformer",
            relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
        )
        activations_mm3 = {layer: acts.detach() for layer, acts in cache_mm3.items()}
        activations_mm3["combined"] = _combine_layer_activations(activations_mm3)
        combined_mm3 = activations_mm3["combined"].to(device_t)
        dedup_acts_mm3, _ = deduplicate_tensor(mm3_data.prefix_map, combined_mm3)

        mm3_model, _ = fit_weighted_ridge(
            dedup_acts_mm3,
            mm3_data.dedup_beliefs,
            mm3_data.dedup_probs,
            n_splits=0,
            alpha_override=ridge_alpha,
        )

        combined_bloch_inputs = mm3_base_seq * vocab_bloch + bloch_data.nn_inputs.to(device_t)
        cache_bloch = extractor.extract_activations(
            model,
            combined_bloch_inputs.long(),
            "transformer",
            relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
        )
        activations_bloch = {layer: acts.detach() for layer, acts in cache_bloch.items()}
        activations_bloch["combined"] = _combine_layer_activations(activations_bloch)
        combined_bloch = activations_bloch["combined"].to(device_t)
        dedup_acts_bloch, _ = deduplicate_tensor(bloch_data.prefix_map, combined_bloch)

        bloch_model, _ = fit_weighted_ridge(
            dedup_acts_bloch,
            bloch_data.dedup_beliefs,
            bloch_data.dedup_probs,
            n_splits=0,
            alpha_override=ridge_alpha,
        )

        mm3_samples, bloch_samples, sample_weights = sample_sequences(
            mm3_dataset,
            bloch_dataset,
            samples,
            device_t,
        )
        combined_samples = mm3_samples * vocab_bloch + bloch_samples

        cache_joint = extractor.extract_activations(
            model,
            combined_samples.long(),
            "transformer",
            relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
        )
        activations_joint = {layer: acts.detach() for layer, acts in cache_joint.items()}
        activations_joint["combined"] = _combine_layer_activations(activations_joint)
        joint_resid_full = activations_joint["combined"].to(device_t)
        joint_resid = joint_resid_full[:, -1, :]

        mm3_pred = predict_with_regressor(mm3_model, joint_resid)
        bloch_pred = predict_with_regressor(bloch_model, joint_resid)

        mm3_dim = mm3_pred.shape[1]
        bloch_dim = bloch_pred.shape[1]

        Z = np.einsum("bi,bj->bij", mm3_pred, bloch_pred).reshape(samples, mm3_dim * bloch_dim)
        ones_col = np.ones((samples, 1), dtype=np.float64)
        F = np.concatenate([Z, mm3_pred, bloch_pred, ones_col], axis=1)

        weights_norm = sample_weights / sample_weights.sum()
        alpha_aug = ridge_alpha if ridge_alpha is not None else 1e-10 * np.trace((F.T * weights_norm) @ F)
        from sklearn.linear_model import Ridge  # local import to avoid unused warning earlier

        ridge_aug = Ridge(alpha=alpha_aug, fit_intercept=False)

        mm3_lookup = build_final_belief_lookup(mm3_data)
        bloch_lookup = build_final_belief_lookup(bloch_data)
        mm3_true_sample = np.stack(
            [mm3_lookup[tuple(seq.cpu().tolist())] for seq in mm3_samples],
            axis=0,
        )
        bloch_true_sample = np.stack(
            [bloch_lookup[tuple(seq.cpu().tolist())] for seq in bloch_samples],
            axis=0,
        )

        joint_true = np.einsum("bi,bj->bij", mm3_true_sample, bloch_true_sample).reshape(samples, -1)

        ridge_aug.fit(F, joint_true, sample_weight=sample_weights)
        coef_aug = ridge_aug.coef_
        idx_z_end = mm3_dim * bloch_dim
        coef_z = coef_aug[:, :idx_z_end]

        U, S, Vt = np.linalg.svd(coef_z, full_matrices=False)
        baseline = np.zeros(bloch_dim)
        baseline[bloch_index] = 1.0
        mm3_pred_dedup = predict_with_regressor(mm3_model, dedup_acts_mm3)

        if isinstance(component_idx, str):
            comp_lower = component_idx.lower()
            if comp_lower == "all":
                idx_range = range(len(S))
            else:
                import re
                match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", comp_lower)
                if not match:
                    raise ValueError("component must be an integer, 'all', or 'start-end'")
                start = int(match.group(1))
                end = int(match.group(2))
                idx_range = range(start, min(end, len(S) - 1) + 1)

            approx_mm3 = np.zeros((mm3_pred_dedup.shape[0], mm3_dim))
            for i in idx_range:
                if i >= len(S):
                    raise ValueError(f"Component index {i} out of range (max {len(S)-1}).")
                V_matrix = Vt[i, :].reshape(mm3_dim, bloch_dim)
                w = V_matrix @ baseline
                scalar = mm3_pred_dedup @ w
                joint_template = (S[i] * U[:, i]).reshape(mm3_dim, bloch_dim)
                approx_joint = np.einsum("b,ij->bij", scalar, joint_template)
                approx_mm3 += approx_joint.sum(axis=2)
        elif isinstance(component_idx, tuple):
            start, end = component_idx
            idx_range = range(start, min(end, len(S) - 1) + 1)
            approx_mm3 = np.zeros((mm3_pred_dedup.shape[0], mm3_dim))
            for i in idx_range:
                if i >= len(S):
                    raise ValueError(f"Component index {i} out of range (max {len(S)-1}).")
                V_matrix = Vt[i, :].reshape(mm3_dim, bloch_dim)
                w = V_matrix @ baseline
                scalar = mm3_pred_dedup @ w
                joint_template = (S[i] * U[:, i]).reshape(mm3_dim, bloch_dim)
                approx_joint = np.einsum("b,ij->bij", scalar, joint_template)
                approx_mm3 += approx_joint.sum(axis=2)
        else:
            if component_idx >= len(S):
                raise ValueError(f"Component index {component_idx} out of range (max {len(S)-1}).")
            V_matrix = Vt[component_idx, :].reshape(mm3_dim, bloch_dim)
            w = V_matrix @ baseline
            scalar = mm3_pred_dedup @ w
            joint_template = (S[component_idx] * U[:, component_idx]).reshape(mm3_dim, bloch_dim)
            approx_joint = np.einsum("b,ij->bij", scalar, joint_template)
            approx_mm3 = approx_joint.sum(axis=2)

        approx_mm3 = np.clip(approx_mm3, 1e-12, None)
        approx_mm3 = approx_mm3 / approx_mm3.sum(axis=1, keepdims=True)

        ground_truth = mm3_data.dedup_beliefs.cpu().numpy()
        return ground_truth, approx_mm3


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MM3 belief geometry inferred from the first SVD component.")
    parser.add_argument("--artifact", required=True, help="WandB artifact path (entity/project/run-model:version)")
    parser.add_argument("--component", default="0", help="SVD component index or 'all' to use full reconstruction")
    parser.add_argument("--bloch-index", type=int, default=0, help="Bloch token index to hold fixed")
    parser.add_argument("--samples", type=int, default=10000, help="Number of joint samples for regression")
    parser.add_argument("--ridge-alpha", type=float, default=1e-6, help="Ridge regularisation strength")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (e.g. cpu or cuda:0)")
    parser.add_argument("--output", type=Path, default=Path("figs/mm3_svd_component.png"), help="Output figure path")
    args = parser.parse_args()

    comp_arg = args.component
    if comp_arg.lower() == "all":
        comp_idx = "all"
    else:
        import re
        match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", comp_arg)
        if match:
            comp_idx = (int(match.group(1)), int(match.group(2)))
        else:
            comp_idx = int(comp_arg)

    ground_truth, approx = compute_component_geometry(
        artifact_path=args.artifact,
        component_idx=comp_idx,
        bloch_index=args.bloch_index,
        samples=args.samples,
        ridge_alpha=args.ridge_alpha,
        device=args.device,
    )

    r2 = r2_score(ground_truth, approx)
    rmse = np.sqrt(mean_squared_error(ground_truth, approx))

    x_gt, y_gt = project_to_simplex_3(ground_truth)
    x_est, y_est = project_to_simplex_3(approx)

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(x_gt, y_gt, s=8, alpha=0.7)
    ax1.set_title("Ground Truth MM3 Beliefs")
    ax1.set_axis_off()

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(x_est, y_est, s=8, alpha=0.7, color="tab:orange")
    ax2.set_title(f"SVD Component {args.component} Projection")
    ax2.set_axis_off()

    plt.suptitle(f"MM3 Geometry vs SVD Component (R²={r2:.3f}, RMSE={rmse:.3e})", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
