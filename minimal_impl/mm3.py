"""
Utility helpers for constructing the Mess3 (MM3) process and enumerating the
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
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import mess3 as _mess3_matrix
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
class MM3Dataset:
    """
    Container holding the complete MM3 sequence distribution.

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
        Sample a sequence according to the exact MM3 distribution.

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


def build_mm3_process(x: float = 0.15, a: float = 0.6) -> TransitionMatrixGHMM:
    """
    Instantiate the MM3 process using the canonical parameters from the paper.

    Args:
        x: Reset probability parameter.
        a: Self-transition parameter.

    Returns:
        A ``TransitionMatrixGHMM`` representing the MM3 process.
    """
    process = TransitionMatrixGHMM(_mess3_matrix(x=x, a=a))
    process.name = "mess3"
    return process


def generate_mm3_transformer_data(
    n_ctx: int,
    *,
    bos: bool = True,
    x: float = 0.15,
    a: float = 0.6,
    device: Optional[Union[str, torch.device]] = "auto",
    as_numpy: bool = True,
) -> MM3Dataset:
    """
    Enumerate every possible MM3 transformer input sequence and its probability.

    Args:
        n_ctx: Context length used by the transformer (tokens available to predict the next token).
        bos: Whether to prefix sequences with the BOS token (index equals vocab size).
        x: Reset probability parameter passed to the MM3 transition matrix.
        a: Self-transition parameter passed to the MM3 transition matrix.
        device: Torch device specifier. Use ``"auto"`` (default) to pick CUDA when available.
        as_numpy: When ``True`` (default), convert outputs to NumPy arrays. Set to ``False`` to keep tensors on ``device``.

    Returns:
        ``MM3Dataset`` bundling the sequences, their probabilities, and auxiliary metadata.
    """
    if n_ctx < 1:
        raise ValueError("n_ctx must be at least 1.")

    process = build_mm3_process(x=x, a=a)
    torch_device = _resolve_device(device)

    # We add one extra position so (context, target) pairs have length n_ctx + 1.
    seq_len = n_ctx + 1
    transformer_inputs, probs, loss_lower_bound = generate_all_seqs(process, seq_len, bos=bos)

    transformer_inputs = transformer_inputs.to(torch_device)
    probs = probs.to(torch_device)
    loss_lower_bound = torch.as_tensor(loss_lower_bound, dtype=torch.float32, device=torch_device)

    bos_token = process.vocab_len if bos else None

    if as_numpy:
        return MM3Dataset(
            transformer_inputs=_to_numpy(transformer_inputs),
            probabilities=_to_numpy(probs),
            loss_lower_bound=_to_numpy(loss_lower_bound),
            bos_token=bos_token,
            backend="numpy",
            device=None,
        )

    return MM3Dataset(
        transformer_inputs=transformer_inputs,
        probabilities=probs,
        loss_lower_bound=loss_lower_bound,
        bos_token=bos_token,
        backend="torch",
        device=torch_device,
    )


def sample_mm3_sequence(
    n_ctx: int,
    *,
    bos: bool = True,
    x: float = 0.15,
    a: float = 0.6,
    device: Optional[Union[str, torch.device]] = "auto",
    as_numpy: bool = True,
    rng: Optional[Union[np.random.Generator, torch.Generator]] = None,
) -> Tuple[ArrayLike, float]:
    """
    Convenience wrapper that samples a single sequence and returns its probability.

    Args:
        n_ctx: Context length used by the transformer.
        bos: Whether to prefix sequences with the BOS token.
        x: Reset probability parameter.
        a: Self-transition parameter.
        device: Torch device specifier.
        as_numpy: Whether to convert outputs to NumPy arrays (matches ``generate_mm3_transformer_data``).
        rng: Optional random generator.

    Returns:
        A tuple ``(sequence, probability)`` drawn from the MM3 distribution.
    """
    dataset = generate_mm3_transformer_data(
        n_ctx,
        bos=bos,
        x=x,
        a=a,
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


def run_belief_regression(
    model: HookedTransformer,
    dataset: MM3Dataset,
    *,
    mm3_cfg: Dict[str, Any],
) -> Dict[str, float]:
    bos = mm3_cfg.get("bos", False)
    n_ctx = mm3_cfg.get("n_ctx", dataset.transformer_inputs.shape[1] - 1)
    process = build_mm3_process(
        x=mm3_cfg.get("x", 0.15),
        a=mm3_cfg.get("a", 0.6),
    )

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
    belief_states = beliefs_out[0].to(model.cfg.device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        inputs = contexts.to(model.cfg.device)
        _, cache = model.run_with_cache(
            inputs,
            names_filter=lambda name: name == "ln_final.hook_normalized",
        )
        activations = cache["ln_final.hook_normalized"].detach()
    if was_training:
        model.train()

    acts_flat = activations.reshape(-1, activations.shape[-1]).float()
    beliefs_flat = belief_states.reshape(-1, belief_states.shape[-1]).float()

    lstsq_result = torch.linalg.lstsq(acts_flat, beliefs_flat)
    preds = acts_flat @ lstsq_result.solution
    residuals = preds - beliefs_flat
    mse = torch.mean(residuals.pow(2))
    rmse_val = torch.sqrt(mse).item()

    total_variance = torch.mean(
        (beliefs_flat - beliefs_flat.mean(dim=0, keepdim=True)).pow(2)
    )
    r_squared = 1.0 - (mse / total_variance) if total_variance > 0 else float("nan")

    rank_value = lstsq_result.rank
    if isinstance(rank_value, torch.Tensor):
        rank_value = (
            int(rank_value.item()) if rank_value.numel() == 1 else float("nan")
        )

    r_squared_value = (
        float(r_squared.item()) if torch.isfinite(r_squared) else float("nan")
    )

    return {
        "belief_regression_rmse": rmse_val,
        "belief_regression_mse": mse.item(),
        "belief_regression_rank": rank_value,
        "belief_regression_r2": r_squared_value,
    }


def main(config_path: str = "mm3_config.yaml") -> None:
    config = _load_config(Path(config_path))

    device_choice = config.get("device", "auto")
    mm3_defaults = {"n_ctx": 7, "bos": False, "x": 0.15, "a": 0.6}
    mm3_defaults.update(config.get("mm3", {}))
    dataset = generate_mm3_transformer_data(**mm3_defaults, device=device_choice, as_numpy=False)

    params = TransformerParams(**{**config.get("model", {}), "device": device_choice})
    model = create_hooked_transformer(dataset, params)

    training_cfg = config.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_cfg.get("learning_rate", 1e-3)
    )
    scheduler = _build_scheduler(optimizer, training_cfg.get("scheduler"))

    generator = lambda: (
        dataset.transformer_inputs,
        dataset.probabilities,
        dataset.loss_lower_bound,
    )

    wandb_run = _init_wandb(config)

    batches_per_epoch = int(training_cfg.get("batches_per_epoch", 1))
    weighted_loss = bool(training_cfg.get("weighted_loss", False))
    num_epochs = int(training_cfg.get("num_epochs", 10))
    total_steps = num_epochs if weighted_loss else num_epochs * batches_per_epoch

    regression_cfg = config.get("belief_regression", {})
    regression_intervals = int(regression_cfg.get("intervals", 0) or 0)
    if regression_intervals > 0 and total_steps > 0:
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
        if wandb_run is not None:
            wandb.log(
                {
                    "actual_step": step,
                    "train_rmse": train_rmse,
                    "val_rmse": val_rmse,
                },
                step=wb_step,
            )
        should_eval = step == -1 or (regression_steps_set and step in regression_steps_set)
        if should_eval:
            metrics = run_belief_regression(model, dataset, mm3_cfg=mm3_defaults)
            tag = "initial" if step == -1 else f"step_{step}"
            print(f"Belief regression ({tag}): {metrics}")
            if wandb_run is not None:
                wandb.log(
                    {f"belief_{k}": v for k, v in metrics.items()},
                    step=wb_step,
                )

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

    regression_metrics = run_belief_regression(
        model,
        dataset,
        mm3_cfg=mm3_defaults,
    )
    print("Belief regression metrics:", regression_metrics)

    if wandb_run is not None:
        wandb_run.summary["train_rmse"] = final_train_rmse
        wandb_run.summary["val_rmse"] = final_val_rmse
        wandb.log(
            {f"belief_{k}": v for k, v in regression_metrics.items()},
            step=total_steps + 1,
        )
        for key, value in regression_metrics.items():
            wandb_run.summary[key] = value
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a HookedTransformer on MM3 sequences.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("MM3_CONFIG", "mm3_config.yaml"),
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
