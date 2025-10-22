"""
Exact MM3 training loop inspired by the original epsilon-transformers repository.

This script aims to reproduce the Mess3 (quantum) transformer experiment as faithfully
as possible, including the optimiser, scheduler, and training schedule that were
used in the paper runs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from minimal_impl.mm3 import generate_mm3_transformer_data, run_belief_regression
from minimal_impl.model import TransformerParams, create_hooked_transformer
from minimal_impl.utils import training_loop, _rmse


@dataclass
class ExactConfig:
    num_epochs: int = 20_000
    batches_per_epoch: int = 200
    learning_rate: float = 1e-4
    batch_size: int = 128
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1000
    scheduler_cooldown: int = 200
    scheduler_threshold: float = 1e-6
    n_layers: int = 4
    n_heads: int = 4
    d_head: int = 16
    d_model: int = 64
    d_mlp: int = 256
    seed: int = 42
    n_ctx: int = 8
    bos: bool = False
    mess3_x: float = 0.15
    mess3_a: float = 0.6
    regression: bool = True


def load_config(path: Optional[Path]) -> ExactConfig:
    if path is None or not path.exists():
        return ExactConfig()
    with path.open("r") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = ExactConfig()
    for field in cfg.__dataclass_fields__:
        if field in raw:
            setattr(cfg, field, raw[field])
    return cfg


def run_exact_training(cfg: ExactConfig) -> None:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    dataset = generate_mm3_transformer_data(
        n_ctx=cfg.n_ctx,
        bos=cfg.bos,
        x=cfg.mess3_x,
        a=cfg.mess3_a,
        device="auto",
        as_numpy=False,
    )

    model_params = TransformerParams(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        d_model=cfg.d_model,
        d_mlp=cfg.d_mlp,
        seed=cfg.seed,
        device="auto",
    )
    model = create_hooked_transformer(dataset, model_params)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        cooldown=cfg.scheduler_cooldown,
        threshold=cfg.scheduler_threshold,
    )

    generator = lambda: (
        dataset.transformer_inputs,
        dataset.probabilities,
        dataset.loss_lower_bound,
    )

    def epoch_callback(step: int, train_loss: torch.Tensor, val_loss: torch.Tensor) -> None:
        train_rmse = _rmse(train_loss)
        val_rmse = _rmse(val_loss)
        if step % 100 == 0 or step < 0:
            print(
                f"[Step {step}] train RMSE={train_rmse:.6f} "
                f"val RMSE={val_rmse:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    results = training_loop(
        model,
        optimizer,
        generator,
        num_epochs=cfg.num_epochs,
        scheduler=scheduler,
        weighted_loss=False,
        batch_size=cfg.batch_size,
        batches_per_epoch=cfg.batches_per_epoch,
        epoch_callback=epoch_callback,
    )

    final_train_rmse = _rmse(results.train_losses[-1])
    final_val_rmse = _rmse(results.val_losses[-1])
    print(
        f"[Exact] Final RMSE -> train={final_train_rmse:.6f} "
        f"validation={final_val_rmse:.6f}"
    )

    if cfg.regression:
        metrics = run_belief_regression(
            model,
            dataset,
            mm3_cfg={
                "n_ctx": cfg.n_ctx,
                "bos": cfg.bos,
                "x": cfg.mess3_x,
                "a": cfg.mess3_a,
            },
            use_gpu=False,
        )
        print("[Exact] Belief regression metrics:", metrics)
    else:
        print("[Exact] Skipping belief regression (disabled).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact Mess3 transformer training.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML file to override default exact training parameters.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_exact_training(cfg)


if __name__ == "__main__":
    main()

