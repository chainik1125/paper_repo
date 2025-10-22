"""
Utility helpers for constructing the Bloch Walk (Tom Quantum) process and enumerating the
exact distribution over transformer input sequences.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import tom_quantum as _tom_quantum_matrix
from epsilon_transformers.training.dataloader import generate_all_seqs
from minimal_impl.model import TransformerParams, create_hooked_transformer
from minimal_impl.utils import training_loop
from transformer_lens import HookedTransformer
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_nn_inputs

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class BlochDataset:
    """
    Container holding the complete Bloch Walk sequence distribution.

    Attributes:
        transformer_inputs: Array of shape (num_seqs, seq_len) with BOS tokens if requested.
        probabilities: Probability assigned to each sequence (sums to one).
        loss_lower_bound: Per-position lower bound on achievable loss (used for diagnostics).
        bos_token: Index reserved for BOS if one was added, otherwise ``None``.
        backend: ``"numpy"`` or ``"torch"`` to indicate storage format.
        device: Torch device when storing tensors (``None`` for NumPy).
    """

    transformer_inputs: ArrayLike
    probabilities: ArrayLike
    loss_lower_bound: ArrayLike
    bos_token: Optional[int]
    backend: str = "numpy"
    device: Optional[torch.device] = None

    def sample(
        self,
        rng: Optional[Union[np.random.Generator, torch.Generator]] = None,
    ) -> Tuple[ArrayLike, float]:
        """
        Sample a sequence according to the exact Bloch Walk distribution.

        Args:
            rng: Optional random generator. Use ``torch.Generator`` when ``backend == 'torch'``.

        Returns:
            Tuple containing the sampled sequence and its probability.
        """
        if self.backend == "torch":
            generator = rng if isinstance(rng, torch.Generator) else None
            probs = self.probabilities
            if probs.ndim != 1:
                probs = probs.flatten()
            idx = torch.multinomial(probs, 1, replacement=True, generator=generator).item()
            seq = self.transformer_inputs[idx]
            return seq, float(self.probabilities[idx].item())

        rng_np = np.random.default_rng() if rng is None else rng
        if not isinstance(rng_np, np.random.Generator):
            raise TypeError("Expected a numpy Generator when backend == 'numpy'.")
        idx = rng_np.choice(len(self.transformer_inputs), p=self.probabilities)
        return self.transformer_inputs[idx], float(self.probabilities[idx])


def _to_numpy(tensor: ArrayLike) -> np.ndarray:
    """Detach a tensor (if needed) and return it as a NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Resolve ``device`` specifiers, defaulting to CUDA when available."""
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def build_bloch_process(alpha: float = 1.0, beta: float = 7.14142842854285) -> TransitionMatrixGHMM:
    """
    Instantiate the Bloch Walk process using the canonical parameters from the paper.

    Args:
        alpha: First parameter for the Bloch Walk dynamics.
        beta: Second parameter for the Bloch Walk dynamics.

    Returns:
        A ``TransitionMatrixGHMM`` representing the Bloch Walk process.
    """
    process = TransitionMatrixGHMM(_tom_quantum_matrix(alpha=alpha, beta=beta))
    process.name = "tom_quantum"
    return process


def generate_bloch_transformer_data(
    n_ctx: int,
    *,
    bos: bool = True,
    alpha: float = 1.0,
    beta: float = 7.14142842854285,
    device: Optional[Union[str, torch.device]] = "auto",
    as_numpy: bool = True,
) -> BlochDataset:
    """
    Enumerate every possible Bloch Walk transformer input sequence and its probability.

    Args:
        n_ctx: Context length used by the transformer (tokens available to predict the next token).
        bos: Whether to prefix sequences with the BOS token (index equals vocab size).
        alpha: First parameter passed to the Bloch Walk transition matrix.
        beta: Second parameter passed to the Bloch Walk transition matrix.
        device: Torch device specifier. Use ``"auto"`` (default) to pick CUDA when available.
        as_numpy: When ``True`` (default), convert outputs to NumPy arrays. Set to ``False`` to keep tensors on ``device``.

    Returns:
        ``BlochDataset`` bundling the sequences, their probabilities, and auxiliary metadata.
    """
    if n_ctx < 1:
        raise ValueError("n_ctx must be at least 1.")

    process = build_bloch_process(alpha=alpha, beta=beta)
    torch_device = _resolve_device(device)

    # We add one extra position so (context, target) pairs have length n_ctx + 1.
    seq_len = n_ctx + 1
    transformer_inputs, probs, loss_lower_bound = generate_all_seqs(process, seq_len, bos=bos)

    transformer_inputs = transformer_inputs.to(torch_device)
    probs = probs.to(torch_device)
    loss_lower_bound = torch.as_tensor(loss_lower_bound, dtype=torch.float32, device=torch_device)

    bos_token = process.vocab_len if bos else None

    if as_numpy:
        return BlochDataset(
            transformer_inputs=_to_numpy(transformer_inputs),
            probabilities=_to_numpy(probs),
            loss_lower_bound=_to_numpy(loss_lower_bound),
            bos_token=bos_token,
            backend="numpy",
            device=None,
        )

    return BlochDataset(
        transformer_inputs=transformer_inputs,
        probabilities=probs,
        loss_lower_bound=loss_lower_bound,
        bos_token=bos_token,
        backend="torch",
        device=torch_device,
    )


def sample_bloch_sequence(
    n_ctx: int,
    *,
    bos: bool = True,
    alpha: float = 1.0,
    beta: float = 7.14142842854285,
    device: Optional[Union[str, torch.device]] = "auto",
    as_numpy: bool = True,
    rng: Optional[Union[np.random.Generator, torch.Generator]] = None,
) -> Tuple[ArrayLike, float]:
    """
    Convenience wrapper that samples a single sequence and returns its probability.

    Args:
        n_ctx: Context length used by the transformer.
        bos: Whether to prefix sequences with the BOS token.
        alpha: First parameter for the Bloch Walk dynamics.
        beta: Second parameter for the Bloch Walk dynamics.
        device: Torch device specifier.
        as_numpy: Whether to convert outputs to NumPy arrays (matches ``generate_bloch_transformer_data``).
        rng: Optional random generator.

    Returns:
        A tuple ``(sequence, probability)`` drawn from the Bloch Walk distribution.
    """
    dataset = generate_bloch_transformer_data(
        n_ctx,
        bos=bos,
        alpha=alpha,
        beta=beta,
        device=device,
        as_numpy=as_numpy,
    )
    return dataset.sample(rng=rng)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _init_wandb(config: Dict[str, Any]) -> Optional["wandb.sdk.wandb_run.Run"]:
    if wandb is None:
        return None

    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    init_kwargs: Dict[str, Any] = {}
    project = wandb_cfg.get("project")
    if project is None:
        raise ValueError("wandb.enabled is True but no project provided.")
    init_kwargs["project"] = project

    if wandb_cfg.get("entity"):
        init_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg.get("run_name"):
        init_kwargs["name"] = wandb_cfg["run_name"]

    mode = wandb_cfg.get("mode")
    init_kwargs["mode"] = mode or "online"

    return wandb.init(config=config, **init_kwargs)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: Optional[Dict[str, Any]]):
    if not cfg:
        return None
    if not isinstance(cfg, dict):
        raise ValueError("scheduler configuration must be a mapping when provided.")
    name = cfg.get("name")
    if not name:
        raise ValueError("scheduler configuration requires a 'name' field.")
    params = cfg.get("params", {})
    scheduler_cls = getattr(torch.optim.lr_scheduler, name)
    return scheduler_cls(optimizer, **params)


def _rmse(loss_tensor: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(loss_tensor.float() ** 2)).item()


def _save_checkpoint(
    model: HookedTransformer,
    checkpoint_path: Path,
    checkpoint_name: str,
    wandb_run: Optional["wandb.sdk.wandb_run.Run"] = None,
) -> None:
    """
    Save a model checkpoint to disk and optionally to wandb.

    Args:
        model: The HookedTransformer model to save.
        checkpoint_path: Directory where checkpoints should be saved.
        checkpoint_name: Name for this checkpoint (e.g., "initial_model.pt", "final_model.pt").
        wandb_run: Optional wandb run object for uploading the checkpoint.
    """
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    model_file = checkpoint_path / checkpoint_name

    # Save model state dict
    torch.save(model.state_dict(), model_file)
    print(f"Saved checkpoint: {model_file}")

    # Upload to wandb if enabled
    if wandb_run is not None:
        wandb.save(str(model_file), policy="now")
        print(f"Uploaded checkpoint to wandb: {checkpoint_name}")


def _find_duplicate_prefixes(contexts: torch.Tensor) -> Dict[Tuple[int, ...], List[Tuple[int, int]]]:
    prefix_map: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    batch, n_ctx = contexts.shape
    for seq_idx in range(batch):
        seq = contexts[seq_idx]
        for pos in range(n_ctx):
            prefix = tuple(int(token) for token in seq[: pos + 1].tolist())
            prefix_map.setdefault(prefix, []).append((seq_idx, pos))
    return prefix_map


def _deduplicate_data(
    contexts: torch.Tensor,
    activations: torch.Tensor,
    beliefs: torch.Tensor,
    probs: torch.Tensor,
    *,
    tolerance: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix_map = _find_duplicate_prefixes(contexts)
    unique_activations: List[torch.Tensor] = []
    unique_beliefs: List[torch.Tensor] = []
    unique_probs: List[float] = []

    max_act_diff = 0.0
    max_belief_diff = 0.0

    for locations in prefix_map.values():
        seq0, pos0 = locations[0]
        act_ref = activations[seq0, pos0].clone()
        belief_ref = beliefs[seq0, pos0].clone()
        prob_sum = probs[seq0, pos0].item()

        for seq_idx, pos in locations[1:]:
            act_cur = activations[seq_idx, pos]
            belief_cur = beliefs[seq_idx, pos]
            prob_sum += probs[seq_idx, pos].item()

            act_diff = torch.max(torch.abs(act_ref - act_cur)).item()
            belief_diff = torch.max(torch.abs(belief_ref - belief_cur)).item()
            max_act_diff = max(max_act_diff, act_diff)
            max_belief_diff = max(max_belief_diff, belief_diff)

            if act_diff > tolerance:
                act_ref = 0.5 * (act_ref + act_cur)
            if belief_diff > tolerance:
                belief_ref = 0.5 * (belief_ref + belief_cur)

        unique_activations.append(act_ref)
        unique_beliefs.append(belief_ref)
        unique_probs.append(prob_sum)

    if max_act_diff > tolerance or max_belief_diff > tolerance:
        print(
            f"[Bloch] Warning: observed activation diff {max_act_diff:.3e}, "
            f"belief diff {max_belief_diff:.3e} during deduplication."
        )

    activations_tensor = torch.stack(unique_activations).to(torch.float32)
    beliefs_tensor = torch.stack(unique_beliefs).to(torch.float32)
    probs_tensor = torch.tensor(unique_probs, dtype=torch.float32)
    probs_tensor = probs_tensor / probs_tensor.sum()

    return activations_tensor, beliefs_tensor, probs_tensor


def run_belief_regression(
    model: HookedTransformer,
    dataset: BlochDataset,
    *,
    bloch_cfg: Dict[str, Any],
    use_gpu: bool = False,
    regularization_rcond: float = 1e-4,
) -> Dict[str, float]:
    bos = bloch_cfg.get("bos", False)
    n_ctx = bloch_cfg.get("n_ctx", dataset.transformer_inputs.shape[1] - 1)
    process = build_bloch_process(
        alpha=bloch_cfg.get("alpha", 1.0),
        beta=bloch_cfg.get("beta", 7.14142842854285),
    )

    if use_gpu:
        print("[Bloch] GPU regression not available; using CPU implementation instead.")

    seq_len = n_ctx + 1
    msp_depth = seq_len + (1 if bos else 2)
    msp = process.derive_mixed_state_tree(depth=msp_depth)

    tree_paths = msp.paths
    tree_beliefs = msp.belief_states
    tree_unnormalized = msp.unnorm_belief_states
    path_probs = msp.path_probs

    msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in tree_beliefs]
    msp_belief_index = {tuple_belief: idx for idx, tuple_belief in enumerate(set(msp_beliefs))}
    probs_dict = {tuple(path): prob for path, prob in zip(tree_paths, path_probs)}

    contexts = dataset.transformer_inputs[:, :-1].to(torch.int64)
    beliefs_out = get_beliefs_for_nn_inputs(
        contexts.cpu(),
        msp_belief_index,
        tree_paths,
        tree_beliefs,
        tree_unnormalized,
        probs_dict,
    )
    belief_states = beliefs_out[0].to(torch.float32).cpu()
    probs_matrix = None
    if len(beliefs_out) >= 3 and isinstance(beliefs_out[2], torch.Tensor):
        probs_matrix = beliefs_out[2].to(torch.float32).cpu()
    if probs_matrix is None:
        probs_matrix = dataset.probabilities.unsqueeze(1).repeat(1, contexts.shape[1]).to(torch.float32).cpu()

    was_training = model.training
    model.eval()
    with torch.no_grad():
        inputs = contexts.to(model.cfg.device)

        n_layers = model.cfg.n_layers
        activation_keys = (
            ['blocks.0.hook_resid_pre']
            + [f'blocks.{i}.hook_resid_post' for i in range(n_layers)]
            + ['ln_final.hook_normalized']
        )

        _, cache = model.run_with_cache(
            inputs,
            names_filter=lambda name: name in activation_keys,
        )

        collected_activations: List[torch.Tensor] = []
        for key in activation_keys:
            if key in cache:
                collected_activations.append(cache[key].detach().to("cpu", dtype=torch.float64))
        if not collected_activations:
            raise ValueError("No activations collected for regression.")

        activations = torch.cat(collected_activations, dim=-1)
    if was_training:
        model.train()

    contexts_cpu = contexts.cpu()
    dedup_acts, dedup_beliefs, dedup_probs = _deduplicate_data(
        contexts_cpu,
        activations.cpu(),
        belief_states,
        probs_matrix,
    )

    rcond_value = regularization_rcond
    weights = dedup_probs.to(torch.float32)
    sqrt_w = torch.sqrt(weights).unsqueeze(1)
    X_w = dedup_acts * sqrt_w
    Y_w = dedup_beliefs * sqrt_w

    beta = torch.linalg.lstsq(X_w, Y_w, rcond=rcond_value).solution

    preds = dedup_acts @ beta
    residuals = preds - dedup_beliefs
    sample_sq_err = torch.sum(residuals.pow(2), dim=1)
    weighted_sq_err = torch.sum(weights * sample_sq_err)
    mse_val = float(weighted_sq_err / weights.sum())
    rmse_val = float(np.sqrt(mse_val))

    weighted_mean = torch.sum(weights.unsqueeze(1) * dedup_beliefs, dim=0) / weights.sum()
    sample_sq_total = torch.sum((dedup_beliefs - weighted_mean) ** 2, dim=1)
    weighted_total = torch.sum(weights * sample_sq_total)
    if weighted_total > 0:
        r_squared = float(1.0 - (weighted_sq_err / weighted_total).item())
    else:
        r_squared = float("nan")

    rank_value = torch.linalg.matrix_rank(
        dedup_acts * torch.sqrt(weights).unsqueeze(1)
    ).item()

    return {
        "belief_regression_rmse": rmse_val,
        "belief_regression_mse": mse_val,
        "belief_regression_rank": rank_value,
        "belief_regression_r2": r_squared,
    }


def main(config_path: str = "bloch_config.yaml") -> None:
    config = _load_config(Path(config_path))

    device_choice = config.get("device", "auto")
    bloch_defaults = {"n_ctx": 7, "bos": False, "alpha": 1.0, "beta": 7.14142842854285}
    bloch_defaults.update(config.get("bloch_walk", {}))
    dataset = generate_bloch_transformer_data(**bloch_defaults, device=device_choice, as_numpy=False)

    params = TransformerParams(**{**config.get("model", {}), "device": device_choice})
    model = create_hooked_transformer(dataset, params)

    training_cfg = config.get("training", {})
    adam_beta1 = float(training_cfg.get("beta1", 0.9))
    adam_beta2 = float(training_cfg.get("beta2", 0.999))
    adam_eps = float(training_cfg.get("eps", 1e-8))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.get("learning_rate", 1e-3),
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )
    scheduler = _build_scheduler(optimizer, training_cfg.get("scheduler"))

    generator = lambda: (
        dataset.transformer_inputs,
        dataset.probabilities,
        dataset.loss_lower_bound,
    )

    wandb_run = _init_wandb(config)

    # Checkpoint configuration
    checkpoint_cfg = config.get("checkpoints", {})
    save_checkpoints = bool(checkpoint_cfg.get("enabled", True))  # Default: enabled
    checkpoint_dir = Path(checkpoint_cfg.get("dir", "./checkpoints"))

    # Save initial model checkpoint (before training)
    if save_checkpoints:
        _save_checkpoint(model, checkpoint_dir, "initial_model.pt", wandb_run)

    batches_per_epoch = int(training_cfg.get("batches_per_epoch", 1))
    weighted_loss = bool(training_cfg.get("weighted_loss", False))
    num_epochs = int(training_cfg.get("num_epochs", 10))
    total_steps = num_epochs if weighted_loss else num_epochs * batches_per_epoch

    regression_cfg = config.get("belief_regression", {})
    regression_enabled = bool(regression_cfg.get("enabled", True))
    regression_intervals = int(regression_cfg.get("intervals", 0) or 0)
    regression_use_gpu = bool(regression_cfg.get("use_gpu", False))
    regression_rcond = float(regression_cfg.get("rcond", 1e-4))
    if not regression_enabled:
        regression_steps = []
    elif regression_intervals > 0 and total_steps > 0:
        regression_steps = sorted(
            {
                int(round(step))
                for step in np.linspace(0, max(total_steps - 1, 0), regression_intervals)
            }
        )
    else:
        regression_steps = []
    regression_steps_set = set(regression_steps)

    def step_callback(step: int, train_loss: torch.Tensor, val_loss: torch.Tensor) -> None:
        train_rmse = _rmse(train_loss)
        val_rmse = _rmse(val_loss)
        if step < 0:
            print(f"Initial RMSE -> train: {train_rmse:.6f}, val: {val_rmse:.6f}")
        wb_step = step + 1 if step >= 0 else 0

        # Prepare base metrics
        log_dict = {
            "actual_step": step,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        }

        # Add belief regression metrics if this is an evaluation step
        should_eval = (
            regression_enabled
            and (step == -1 or (regression_steps_set and step in regression_steps_set))
        )
        if should_eval:
            metrics = run_belief_regression(
                model,
                dataset,
                bloch_cfg=bloch_defaults,
                use_gpu=regression_use_gpu,
                regularization_rcond=regression_rcond,
            )
            tag = "initial" if step == -1 else f"step_{step}"
            print(f"Belief regression ({tag}): {metrics}")
            # Add belief metrics to the same log dict
            log_dict.update({f"belief_{k}": v for k, v in metrics.items()})

        # Single wandb.log() call with all metrics
        if wandb_run is not None:
            wandb.log(log_dict, step=wb_step)

    results = training_loop(
        model,
        optimizer,
        generator,
        num_epochs=num_epochs,
        scheduler=scheduler,
        weighted_loss=weighted_loss,
        batch_size=training_cfg.get("batch_size"),
        batches_per_epoch=batches_per_epoch,
        epoch_callback=step_callback,
    )

    final_train_rmse = _rmse(results.train_losses[-1])
    final_val_rmse = _rmse(results.val_losses[-1])
    print(f"Final RMSE -> train: {final_train_rmse:.6f}, validation: {final_val_rmse:.6f}")

    if regression_enabled:
        regression_metrics = run_belief_regression(
            model,
            dataset,
            bloch_cfg=bloch_defaults,
            use_gpu=regression_use_gpu,
            regularization_rcond=regression_rcond,
        )
        print("Belief regression metrics:", regression_metrics)
    else:
        regression_metrics = {}
        print("Belief regression disabled; skipping final regression.")

    # Save final model checkpoint (after training)
    if save_checkpoints:
        _save_checkpoint(model, checkpoint_dir, "final_model.pt", wandb_run)

    if wandb_run is not None:
        wandb_run.summary["train_rmse"] = final_train_rmse
        wandb_run.summary["val_rmse"] = final_val_rmse
        if regression_metrics:
            wandb.log(
                {f"belief_{k}": v for k, v in regression_metrics.items()},
                step=total_steps + 1,
            )
            for key, value in regression_metrics.items():
                wandb_run.summary[key] = value
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a HookedTransformer on Bloch Walk sequences.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("BLOCH_CONFIG", "bloch_config.yaml"),
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
