"""
Command-line entry point for training a HookedTransformer on the MM3 process
using the lightweight utilities in this directory.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parent.parent))

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
import numpy as np

from minimal_impl.mm3 import generate_mm3_transformer_data, build_mm3_process
from minimal_impl.model import TransformerParams, create_hooked_transformer
from minimal_impl.utils import TrainingResult, training_loop
from transformer_lens import HookedTransformer
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_nn_inputs

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is an optional dependency at runtime
    wandb = None  # type: ignore


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def rmse(loss_tensor: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(loss_tensor.float() ** 2)).item()


def maybe_init_wandb(config: Dict[str, Any]) -> Optional["wandb.sdk.wandb_run.Run"]:
    if wandb is None:
        return None

    wandb_cfg: Dict[str, Any] = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    init_kwargs: Dict[str, Any] = {}
    project = wandb_cfg.get("project")
    if project is None:
        raise ValueError("W&B project must be specified when wandb.enabled is True.")
    init_kwargs["project"] = project

    entity = wandb_cfg.get("entity")
    if entity:
        init_kwargs["entity"] = entity
    run_name = wandb_cfg.get("run_name")
    if run_name:
        init_kwargs["name"] = run_name

    mode = wandb_cfg.get("mode")
    if mode and mode != "auto":
        init_kwargs["mode"] = mode
    elif "WANDB_API_KEY" not in os.environ:
        init_kwargs["mode"] = "offline"

    return wandb.init(config=config, **init_kwargs)


def prepare_transformer_parameters(cfg: Dict[str, Any], device: str) -> TransformerParams:
    params_cfg = dict(cfg)
    params_cfg.setdefault("device", device)
    return TransformerParams(**params_cfg)


def run_belief_regression(
    model: HookedTransformer,
    dataset,
    *,
    mm3_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Fit a linear mapping from the model's final activations to MM3 belief states and report fit metrics.
    """
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
    # Returned tuple: beliefs, indices, probs, unnormalized beliefs
    belief_states = beliefs_out[0].to(model.cfg.device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model_inputs = contexts.to(model.cfg.device)
        _, cache = model.run_with_cache(
            model_inputs,
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


def run_training(config: Dict[str, Any]) -> TrainingResult:
    device_choice = config.get("device", "auto")

    mm3_cfg = config.get("mm3", {})
    dataset = generate_mm3_transformer_data(
        n_ctx=mm3_cfg.get("n_ctx", 7),
        bos=mm3_cfg.get("bos", False),
        x=mm3_cfg.get("x", 0.15),
        a=mm3_cfg.get("a", 0.6),
        device=device_choice,
        as_numpy=False,
    )

    model_cfg = prepare_transformer_parameters(config.get("model", {}), device_choice)
    model = create_hooked_transformer(dataset, model_cfg)

    torch.manual_seed(model_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_cfg.seed)

    training_cfg = config.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_cfg.get("learning_rate", 1e-3)
    )

    scheduler_cfg = training_cfg.get("scheduler")
    if isinstance(scheduler_cfg, dict):
        scheduler_name = scheduler_cfg.get("name")
        scheduler_params = scheduler_cfg.get("params", {})
        if not hasattr(torch.optim.lr_scheduler, scheduler_name):
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_cls(optimizer, **scheduler_params)
    else:
        scheduler = None

    sequence_tensors = (
        dataset.transformer_inputs,
        dataset.probabilities,
        dataset.loss_lower_bound,
    )

    def generator():
        return sequence_tensors

    batches_per_epoch = int(training_cfg.get("batches_per_epoch", 1))
    batch_size = training_cfg.get("batch_size")
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

    wandb_run = maybe_init_wandb(config)

    def step_callback(step: int, train_loss: torch.Tensor, val_loss: torch.Tensor) -> None:
        train_rmse = rmse(train_loss)
        val_rmse = rmse(val_loss)
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
            metrics = run_belief_regression(model, dataset, mm3_cfg=mm3_cfg)
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
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        epoch_callback=step_callback,
    )

    final_train_rmse = rmse(results.train_losses[-1])
    final_val_rmse = rmse(results.val_losses[-1])
    print(
        f"Final RMSE -> train: {final_train_rmse:.6f}, "
        f"validation: {final_val_rmse:.6f}"
    )

    regression_metrics = run_belief_regression(
        model,
        dataset,
        mm3_cfg=mm3_cfg,
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

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HookedTransformer on MM3 sequences.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("mm3_config.yaml"),
        help="Path to configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
