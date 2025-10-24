"""
Analyse a cartesian-product checkpoint via tensor-factor regression.

Given a trained MM3×Bloch transformer stored as a WandB artifact, this script:
  1. downloads the checkpoint and its run config
  2. fits two linear probes on the joint activations while freezing the other
     process (baseline Bloch / baseline MM3) to recover MM3-only and Bloch-only
     belief readouts
  3. evaluates whether the Kronecker product of those two regressors reproduces
     the joint belief geometry by sampling from the product distribution
  4. reports regression metrics for each component and for the factorised vs
     directly-fitted joint probe.

The output is printed as a JSON blob (and optionally written to disk with
--output-json).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from minimal_impl.cartesian import _infer_vocab_size
from minimal_impl.mm3 import generate_mm3_transformer_data
from minimal_impl.bloch import generate_bloch_transformer_data
from minimal_impl.regression import deduplicate_data, deduplicate_tensor, _combine_layer_activations
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS
from scripts.activation_analysis.data_loading import ActivationExtractor
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, HookedTransformerConfig

try:  # WandB is optional for offline debugging
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


@dataclass
class ProcessBeliefData:
    nn_inputs: torch.Tensor
    nn_beliefs: torch.Tensor
    nn_probs: torch.Tensor
    dedup_probs: torch.Tensor
    dedup_beliefs: torch.Tensor
    dedup_indices: List[Tuple[int, int]]
    prefix_map: Dict[Tuple[int, ...], List[Tuple[int, int]]]


def download_artifact(artifact_path: str, workdir: Path) -> Tuple[Path, dict]:
    if wandb is None:
        raise RuntimeError("wandb is required to download artifacts")

    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = Path(artifact.download(root=str(workdir)))

    ckpt_files = list(artifact_dir.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found inside {artifact_path}")
    checkpoint_path = ckpt_files[0]

    run = artifact.logged_by()
    try:
        config_file = next(f for f in run.files() if f.name.endswith("run_config.yaml"))
        config_file.download(root=str(workdir), replace=True)
        run_config = yaml.safe_load((workdir / config_file.name).read_text())
    except StopIteration:
        # fallback to the WandB run config
        run_config = dict(run.config)

    return checkpoint_path, run_config


def instantiate_model(checkpoint_path: Path, run_config: dict, device: str) -> HookedTransformer:
    model_cfg = dict(run_config.get("model_config", {}))
    if "dtype" in model_cfg and isinstance(model_cfg["dtype"], str):
        model_cfg["dtype"] = getattr(torch, model_cfg["dtype"].split(".")[-1])
    else:
        model_cfg["dtype"] = torch.float32

    model_cfg.setdefault("device", device)

    state = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    if "d_vocab" not in model_cfg:
        model_cfg["d_vocab"] = state["embed.W_E"].shape[0]

    cfg = HookedTransformerConfig(**model_cfg)
    model = HookedTransformer(cfg)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def prepare_process_data(base_config: dict, process_name: str, params: dict) -> ProcessBeliefData:
    cfg = dict(base_config)
    cfg.setdefault("global_config", {}).setdefault("device", "cpu")
    cfg.setdefault("train_config", {})
    cfg["train_config"].setdefault("bos", False)
    model_cfg = dict(cfg["model_config"])

    cfg["process_config"] = dict(params)
    cfg["process_config"]["name"] = process_name
    cfg["model_config"] = model_cfg

    nn_inputs, nn_beliefs, _, nn_probs, _ = prepare_msp_data(cfg, cfg["model_config"])
    dedup_probs, dedup_beliefs, dedup_indices, prefix_map = deduplicate_data(nn_inputs, nn_probs, nn_beliefs)

    return ProcessBeliefData(
        nn_inputs=nn_inputs,
        nn_beliefs=nn_beliefs,
        nn_probs=nn_probs,
        dedup_probs=dedup_probs,
        dedup_beliefs=dedup_beliefs,
        dedup_indices=dedup_indices,
        prefix_map=prefix_map,
    )


def fit_weighted_ridge(
    activations: torch.Tensor,
    beliefs: torch.Tensor,
    weights: torch.Tensor,
    n_splits: int = 5,
    random_state: int = 42,
    alpha_override: float | None = None,
) -> Tuple[Ridge, dict]:
    if activations.ndim == 1:
        X = activations.unsqueeze(0).cpu().numpy()
    else:
        X = activations.reshape(activations.shape[0], -1).cpu().numpy()
    y = beliefs.cpu().numpy()
    w = weights.cpu().numpy()
    w = w / w.sum()

    predictions = np.zeros_like(y)
    use_cv = alpha_override is None and n_splits and n_splits > 1 and X.shape[0] >= n_splits

    if use_cv:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            w_train = w[train_idx]
            w_train = w_train / w_train.sum()

            alpha = 1e-10 * np.trace(X_train.T @ np.diag(w_train) @ X_train)
            reg = Ridge(alpha=alpha, fit_intercept=True)
            reg.fit(X_train, y_train, sample_weight=w_train)
            predictions[test_idx] = reg.predict(X_test)
    else:
        alpha = alpha_override if alpha_override is not None else 1e-10 * np.trace(X.T @ np.diag(w) @ X)
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(X, y, sample_weight=w)
        predictions = reg.predict(X)

    residuals = y - predictions
    mse = np.sum(w * (residuals**2).sum(axis=1))
    rmse = np.sqrt(mse)

    y_mean = np.average(y, axis=0, weights=w)
    ss_tot = np.sum(w * ((y - y_mean) ** 2).sum(axis=1))
    ss_res = np.sum(w * (residuals**2).sum(axis=1))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Refit on full data for downstream prediction
    alpha_full = alpha_override if alpha_override is not None else 1e-10 * np.trace(X.T @ np.diag(w) @ X)
    model = Ridge(alpha=alpha_full, fit_intercept=True)
    model.fit(X, y, sample_weight=w)

    metrics = {"rmse": float(rmse), "r2": float(r2)}
    return model, metrics


def predict_with_regressor(model: Ridge, activations: torch.Tensor) -> np.ndarray:
    if activations.ndim == 1:
        X = activations.unsqueeze(0).cpu().numpy()
    else:
        X = activations.reshape(activations.shape[0], -1).cpu().numpy()
    return model.predict(X)


def sample_sequences(
    mm3_dataset,
    bloch_dataset,
    num_samples: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    with torch.no_grad():
        mm3_inputs = mm3_dataset.transformer_inputs[:, :-1]  # drop target token
        bloch_inputs = bloch_dataset.transformer_inputs[:, :-1]
        mm3_probs = mm3_dataset.probabilities
        bloch_probs = bloch_dataset.probabilities

        idx_mm3 = torch.multinomial(mm3_probs, num_samples, replacement=True)
        idx_bloch = torch.multinomial(bloch_probs, num_samples, replacement=True)

        mm3_samples = mm3_inputs[idx_mm3].to(device)
        bloch_samples = bloch_inputs[idx_bloch].to(device)

        weights = (mm3_probs[idx_mm3] * bloch_probs[idx_bloch]).cpu().numpy()
        return mm3_samples, bloch_samples, weights


def build_final_belief_lookup(data: ProcessBeliefData) -> Dict[Tuple[int, ...], np.ndarray]:
    full_sequences = data.nn_inputs
    final_beliefs = data.nn_beliefs[:, -1, :]
    lookup = {}
    for seq, belief in zip(full_sequences, final_beliefs):
        lookup[tuple(seq.tolist())] = belief.cpu().numpy()
    return lookup


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    weights = weights / weights.sum()
    residuals = y_true - y_pred
    mse = np.sum(weights * (residuals**2).sum(axis=1))
    rmse = np.sqrt(mse)

    mean = np.average(y_true, axis=0, weights=weights)
    ss_tot = np.sum(weights * ((y_true - mean) ** 2).sum(axis=1))
    ss_res = np.sum(weights * (residuals**2).sum(axis=1))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": float(rmse), "r2": r2}


def main():
    parser = argparse.ArgumentParser(description="Tensor-factor regression analysis for MM3×Bloch checkpoints")
    parser.add_argument("--config", type=Path, help="YAML file describing one or more analyses")
    parser.add_argument("--artifact", help="WandB artifact path, e.g. entity/project/run-model-179200:v0")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write metrics JSON")
    parser.add_argument("--samples", type=int, default=50000, help="Number of joint samples for evaluation")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds for the component regressions")
    parser.add_argument("--device", type=str, default="cpu", help="torch device")
    parser.add_argument("--kronecker-rank", type=int, default=5, help="Rank for low-rank Kronecker correction (0 disables)")
    parser.add_argument("--operator-rank", type=int, default=5, help="Rank for operator Schmidt approximation (0 disables)")
    parser.add_argument("--ridge-alpha", type=float, default=None, help="Override ridge regularisation strength (disables CV)")
    parser.add_argument("--random-baseline", action="store_true", help="Evaluate a random initialised model as baseline")
    args = parser.parse_args()

    analyses: List[dict]
    if args.config:
        config_data = yaml.safe_load(args.config.read_text())
        if isinstance(config_data, dict) and "runs" in config_data:
            analyses = config_data["runs"]
        elif isinstance(config_data, list):
            analyses = config_data
        else:
            raise ValueError("Config file must contain either a list or a dict with key 'runs'")
    else:
        if args.artifact is None:
            parser.error("Either --config or --artifact must be provided")
        analyses = [{
            "artifact": args.artifact,
            "output_json": str(args.output_json) if args.output_json else None,
            "samples": args.samples,
            "n_splits": args.n_splits,
            "device": args.device,
            "kronecker_rank": args.kronecker_rank,
            "operator_rank": args.operator_rank,
            "ridge_alpha": args.ridge_alpha,
            "random_baseline": args.random_baseline,
        }]

    for entry in analyses:
        artifact_path = entry["artifact"]
        samples = int(entry.get("samples", args.samples))
        n_splits = int(entry.get("n_splits", args.n_splits))
        device = entry.get("device", args.device)
        output_path = entry.get("output_json")
        output_json = Path(output_path) if output_path else None
        rank = int(entry.get("kronecker_rank", args.kronecker_rank))
        operator_rank = int(entry.get("operator_rank", args.operator_rank))
        ridge_alpha = entry.get("ridge_alpha", args.ridge_alpha)
        ridge_alpha = float(ridge_alpha) if ridge_alpha is not None else None
        random_flag = bool(entry.get("random_baseline", args.random_baseline))

        print(f"=== Analysing artifact: {artifact_path} ===")
        results = run_single_analysis(
            artifact_path=artifact_path,
            samples=samples,
            n_splits=n_splits,
            device=device,
            output_json=output_json,
            kronecker_rank=rank,
            operator_rank=operator_rank,
            ridge_alpha=ridge_alpha,
            random_baseline=random_flag,
        )
        print(json.dumps(results, indent=2))


def run_single_analysis(
    artifact_path: str,
    samples: int,
    n_splits: int,
    device: str,
    output_json: Path | None = None,
    kronecker_rank: int = 5,
    operator_rank: int = 5,
    ridge_alpha: float | None = None,
    random_baseline: bool = False,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="fig2_combined_") as tmpdir:
        workdir = Path(tmpdir)
        checkpoint_path, run_cfg = download_artifact(artifact_path, workdir)

        device_t = torch.device(device)
        model = instantiate_model(checkpoint_path, run_cfg, device)
        extractor = ActivationExtractor(device=device_t)

        # Extract process configs
        process_cfg = run_cfg.get("process_config", {})
        if process_cfg.get("mode") != "mm3_bloch_product":
            raise ValueError("Run config does not describe an mm3_bloch_product experiment")

        mm3_params = dict(process_cfg["mm3"])
        bloch_params = dict(process_cfg["bloch"])

        # Prepare belief tables
        base_config = {
            "model_config": dict(run_cfg["model_config"]),
            "train_config": dict(run_cfg.get("train_config", {})),
        }
        base_config["model_config"].setdefault("n_ctx", base_config["model_config"].get("n_ctx"))

        mm3_data = prepare_process_data(base_config, "mess3", mm3_params)
        bloch_data = prepare_process_data(base_config, "tom_quantum", bloch_params)

        # Load enumerated datasets for sampling
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

        # Choose baseline sequences (highest prob)
        mm3_base_idx = torch.argmax(mm3_dataset.probabilities).item()
        bloch_base_idx = torch.argmax(bloch_dataset.probabilities).item()
        mm3_base_seq = mm3_dataset.transformer_inputs[mm3_base_idx, :-1].to(device_t)
        bloch_base_seq = bloch_dataset.transformer_inputs[bloch_base_idx, :-1].to(device_t)

        # --- MM3 regression (vary MM3, fix Bloch) ---
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
        mm3_model, mm3_metrics = fit_weighted_ridge(
            dedup_acts_mm3,
            mm3_data.dedup_beliefs,
            mm3_data.dedup_probs,
            n_splits=n_splits,
            alpha_override=ridge_alpha,
        )

        # --- Bloch regression (vary Bloch, fix MM3) ---
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
        bloch_model, bloch_metrics = fit_weighted_ridge(
            dedup_acts_bloch,
            bloch_data.dedup_beliefs,
            bloch_data.dedup_probs,
            n_splits=n_splits,
            alpha_override=ridge_alpha,
        )

        # --- Sample joint distribution
        num_samples = samples
        mm3_samples, bloch_samples, sample_weights = sample_sequences(mm3_dataset, bloch_dataset, num_samples, device_t)
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

        # Predictions from single regressors
        mm3_pred = predict_with_regressor(mm3_model, joint_resid)
        bloch_pred = predict_with_regressor(bloch_model, joint_resid)

        # True beliefs for samples
        mm3_lookup = build_final_belief_lookup(mm3_data)
        bloch_lookup = build_final_belief_lookup(bloch_data)
        mm3_true = np.stack([mm3_lookup[tuple(seq.tolist())] for seq in mm3_samples.cpu()], axis=0)
        bloch_true = np.stack([bloch_lookup[tuple(seq.tolist())] for seq in bloch_samples.cpu()], axis=0)

        joint_true = np.einsum("bi,bj->bij", mm3_true, bloch_true).reshape(num_samples, -1)
        joint_pred_fact = np.einsum("bi,bj->bij", mm3_pred, bloch_pred).reshape(num_samples, -1)

        fact_metrics = weighted_metrics(joint_true, joint_pred_fact, sample_weights)
        coef_z_singular_values: list[float] = []

        # Augmented regression on outer-product features (with optional low-rank truncation)
        mm3_dim = mm3_pred.shape[1]
        bloch_dim = bloch_pred.shape[1]
        Z = np.einsum("bi,bj->bij", mm3_pred, bloch_pred).reshape(num_samples, mm3_dim * bloch_dim)
        ones_col = np.ones((num_samples, 1), dtype=np.float64)
        F = np.concatenate([Z, mm3_pred, bloch_pred, ones_col], axis=1)
        weights_norm = sample_weights / sample_weights.sum()
        alpha_aug = ridge_alpha if ridge_alpha is not None else 1e-10 * np.trace((F.T * weights_norm) @ F)
        ridge_aug = Ridge(alpha=alpha_aug, fit_intercept=False)
        ridge_aug.fit(F, joint_true, sample_weight=sample_weights)
        coef_aug = ridge_aug.coef_

        idx_z_end = mm3_dim * bloch_dim
        idx_mm3_end = idx_z_end + mm3_dim
        idx_bloch_end = idx_mm3_end + bloch_dim
        coef_z = coef_aug[:, :idx_z_end]
        coef_mm3 = coef_aug[:, idx_z_end:idx_mm3_end]
        coef_bloch = coef_aug[:, idx_mm3_end:idx_bloch_end]
        coef_bias = coef_aug[:, idx_bloch_end]

        joint_pred_aug_full = (
            Z @ coef_z.T
            + mm3_pred @ coef_mm3.T
            + bloch_pred @ coef_bloch.T
            + coef_bias
        )
        augmented_metrics = weighted_metrics(joint_true, joint_pred_aug_full, sample_weights)

        coef_z_singular_values: list[float] = []
        if kronecker_rank and kronecker_rank > 0:
            U, S, Vt = np.linalg.svd(coef_z, full_matrices=False)
            coef_z_singular_values = S.tolist()
            if 0 < kronecker_rank < len(S):
                U_k = U[:, :kronecker_rank]
                S_k = S[:kronecker_rank]
                Vt_k = Vt[:kronecker_rank, :]
                coef_z_trunc = (U_k * S_k) @ Vt_k
            else:
                coef_z_trunc = coef_z
        else:
            coef_z_trunc = coef_z

        joint_pred_aug_lr = (
            Z @ coef_z_trunc.T
            + mm3_pred @ coef_mm3.T
            + bloch_pred @ coef_bloch.T
            + coef_bias
        )
        augmented_lr_metrics = weighted_metrics(joint_true, joint_pred_aug_lr, sample_weights)

        operator_metrics = {}
        operator_singular_values = []
        if operator_rank and operator_rank > 0:
            mm3_dim = mm3_pred.shape[1]
            bloch_dim = bloch_pred.shape[1]
            joint_dim = joint_true.shape[1]
            if joint_dim != mm3_dim * bloch_dim:
                raise ValueError("Joint belief dimension does not match mm3*bloch product")

            W_tensor = coef_z.reshape(mm3_dim, bloch_dim, mm3_dim, bloch_dim)
            W_AB = np.transpose(W_tensor, (0, 2, 1, 3)).reshape(mm3_dim * mm3_dim, bloch_dim * bloch_dim)
            U_op, S_op, Vt_op = np.linalg.svd(W_AB, full_matrices=False)
            operator_singular_values = S_op.tolist()
            k_op = min(operator_rank, len(S_op))
            Wk_AB = (U_op[:, :k_op] * S_op[:k_op]) @ Vt_op[:k_op, :]
            Wk_tensor = np.transpose(Wk_AB.reshape(mm3_dim, mm3_dim, bloch_dim, bloch_dim), (0, 2, 1, 3))
            Wk = Wk_tensor.reshape(joint_dim, mm3_dim * bloch_dim)
            joint_pred_op = (
                Z @ Wk.T
                + mm3_pred @ coef_mm3.T
                + bloch_pred @ coef_bloch.T
                + coef_bias
            )
            operator_metrics = weighted_metrics(joint_true, joint_pred_op, sample_weights)

        # Direct joint regression baseline
        alpha_joint = ridge_alpha if ridge_alpha is not None else 1e-10
        ridge_joint = Ridge(alpha=alpha_joint, fit_intercept=True)
        ridge_joint.fit(joint_resid.cpu().numpy(), joint_true, sample_weight=sample_weights)
        joint_pred_direct = ridge_joint.predict(joint_resid.cpu().numpy())
        direct_metrics = weighted_metrics(joint_true, joint_pred_direct, sample_weights)

        random_baseline_metrics = None
        if random_baseline:
            cfg_copy = run_cfg.get('model_config', {}).copy()
            cfg_copy.setdefault('act_fn', run_cfg.get('model_config', {}).get('act_fn', 'relu'))
            cfg_copy.setdefault('normalization_type', run_cfg.get('model_config', {}).get('normalization_type', 'LN'))
            cfg_copy.setdefault('n_layers', run_cfg['model_config']['n_layers'])
            cfg_copy.setdefault('n_heads', run_cfg['model_config']['n_heads'])
            cfg_copy.setdefault('d_model', run_cfg['model_config']['d_model'])
            cfg_copy.setdefault('d_mlp', run_cfg['model_config']['d_mlp'])
            cfg_copy.setdefault('n_ctx', run_cfg['model_config']['n_ctx'])
            max_token = int(combined_samples.max().item()) + 1
            cfg_copy['d_vocab'] = max_token
            cfg_copy['dtype'] = torch.float32
            cfg_copy['device'] = device
            random_model = HookedTransformer(HookedTransformerConfig(**cfg_copy))
            random_model.eval()

            with torch.no_grad():
                cache_mm3_rand = extractor.extract_activations(
                    random_model,
                    combined_mm3_inputs.long(),
                    'transformer',
                    relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
                )
            activations_mm3_rand = {layer: acts.detach() for layer, acts in cache_mm3_rand.items()}
            activations_mm3_rand['combined'] = _combine_layer_activations(activations_mm3_rand)
            combined_mm3_rand = activations_mm3_rand['combined'].to(device_t)
            dedup_acts_mm3_rand, _ = deduplicate_tensor(mm3_data.prefix_map, combined_mm3_rand)
            mm3_model_rand, mm3_metrics_rand = fit_weighted_ridge(
                dedup_acts_mm3_rand,
                mm3_data.dedup_beliefs,
                mm3_data.dedup_probs,
                n_splits=0,
                alpha_override=ridge_alpha,
            )

            with torch.no_grad():
                cache_bloch_rand = extractor.extract_activations(
                    random_model,
                    combined_bloch_inputs.long(),
                    'transformer',
                    relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
                )
            activations_bloch_rand = {layer: acts.detach() for layer, acts in cache_bloch_rand.items()}
            activations_bloch_rand['combined'] = _combine_layer_activations(activations_bloch_rand)
            combined_bloch_rand = activations_bloch_rand['combined'].to(device_t)
            dedup_acts_bloch_rand, _ = deduplicate_tensor(bloch_data.prefix_map, combined_bloch_rand)
            bloch_model_rand, bloch_metrics_rand = fit_weighted_ridge(
                dedup_acts_bloch_rand,
                bloch_data.dedup_beliefs,
                bloch_data.dedup_probs,
                n_splits=0,
                alpha_override=ridge_alpha,
            )

            with torch.no_grad():
                cache_joint_rand = extractor.extract_activations(
                    random_model,
                    combined_samples.long(),
                    'transformer',
                    relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
                )
            activations_joint_rand = {layer: acts.detach() for layer, acts in cache_joint_rand.items()}
            activations_joint_rand['combined'] = _combine_layer_activations(activations_joint_rand)
            joint_resid_rand_full = activations_joint_rand['combined'].to(device_t)
            joint_resid_rand = joint_resid_rand_full[:, -1, :]

            mm3_pred_rand = predict_with_regressor(mm3_model_rand, joint_resid_rand)
            bloch_pred_rand = predict_with_regressor(bloch_model_rand, joint_resid_rand)

            joint_pred_fact_rand = np.einsum("bi,bj->bij", mm3_pred_rand, bloch_pred_rand).reshape(samples, -1)
            fact_rand_metrics = weighted_metrics(joint_true, joint_pred_fact_rand, sample_weights)

            Z_rand = np.einsum("bi,bj->bij", mm3_pred_rand, bloch_pred_rand).reshape(samples, mm3_dim * bloch_dim)
            F_rand = np.concatenate([Z_rand, mm3_pred_rand, bloch_pred_rand, ones_col], axis=1)

            joint_pred_aug_rand = (
                Z_rand @ coef_z.T
                + mm3_pred_rand @ coef_mm3.T
                + bloch_pred_rand @ coef_bloch.T
                + coef_bias
            )
            aug_rand_metrics = weighted_metrics(joint_true, joint_pred_aug_rand, sample_weights)

            if kronecker_rank and kronecker_rank > 0:
                joint_pred_aug_lr_rand = (
                    Z_rand @ coef_z_trunc.T
                    + mm3_pred_rand @ coef_mm3.T
                    + bloch_pred_rand @ coef_bloch.T
                    + coef_bias
                )
                aug_lr_rand_metrics = weighted_metrics(joint_true, joint_pred_aug_lr_rand, sample_weights)
            else:
                aug_lr_rand_metrics = {}

            if operator_rank and operator_rank > 0:
                joint_pred_op_rand = (
                    Z_rand @ Wk.T
                    + mm3_pred_rand @ coef_mm3.T
                    + bloch_pred_rand @ coef_bloch.T
                    + coef_bias
                )
                operator_rand_metrics = weighted_metrics(joint_true, joint_pred_op_rand, sample_weights)
            else:
                operator_rand_metrics = {}

            ridge_rand = Ridge(alpha=alpha_joint, fit_intercept=True)
            ridge_rand.fit(joint_resid_rand.cpu().numpy(), joint_true, sample_weight=sample_weights)
            joint_pred_rand = ridge_rand.predict(joint_resid_rand.cpu().numpy())
            direct_rand_metrics = weighted_metrics(joint_true, joint_pred_rand, sample_weights)

            random_baseline_metrics = {
                "mm3_metrics": mm3_metrics_rand,
                "bloch_metrics": bloch_metrics_rand,
                "factorised_joint_metrics": fact_rand_metrics,
                "augmented_metrics": aug_rand_metrics,
                "augmented_low_rank_metrics": aug_lr_rand_metrics,
                "operator_schmidt_metrics": operator_rand_metrics,
                "direct_joint_metrics": direct_rand_metrics,
            }

        results = {
            "artifact": artifact_path,
            "mm3_metrics": mm3_metrics,
            "bloch_metrics": bloch_metrics,
            "factorised_joint_metrics": fact_metrics,
            "augmented_metrics": augmented_metrics,
            "augmented_low_rank_metrics": augmented_lr_metrics,
            "operator_schmidt_metrics": operator_metrics,
            "operator_singular_values": operator_singular_values,
            "kronecker_singular_values": coef_z_singular_values,
            "direct_joint_metrics": direct_metrics,
            "random_baseline_metrics": random_baseline_metrics,
            "num_samples": num_samples,
            "kronecker_rank": kronecker_rank,
            "operator_rank": operator_rank,
            "ridge_alpha": ridge_alpha,
        }

        # Save singular value diagnostics
        plot_dir = output_json.parent if output_json else Path("figs")
        plot_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = artifact_path.replace('/', '_').replace(':', '_')
        if coef_z_singular_values:
            plt.figure()
            plt.plot(np.arange(1, len(coef_z_singular_values) + 1), coef_z_singular_values, marker='o')
            plt.xlabel("Component index")
            plt.ylabel("Singular value")
            plt.title("Kronecker Singular Values")
            plt.grid(True, alpha=0.3)
            plt.savefig(plot_dir / f"{safe_prefix}_kronecker_singular_values.png", dpi=200, bbox_inches="tight")
            plt.close()
        if operator_singular_values:
            plt.figure()
            plt.plot(np.arange(1, len(operator_singular_values) + 1), operator_singular_values, marker='o', color='orange')
            plt.xlabel("Component index")
            plt.ylabel("Singular value")
            plt.title("Operator Schmidt Singular Values")
            plt.grid(True, alpha=0.3)
            plt.savefig(plot_dir / f"{safe_prefix}_operator_singular_values.png", dpi=200, bbox_inches="tight")
            plt.close()

        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(results, indent=2))
        return results


if __name__ == "__main__":
    main()
