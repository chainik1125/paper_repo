"""
Light-weight utilities for training TransformerLens models on arbitrary
sequence/probability generators while reusing the project's existing training logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch.optim import Optimizer
from transformer_lens import HookedTransformer
import torch.nn.functional as F
from tqdm.auto import tqdm

from epsilon_transformers.training.dataloader import BatchGenerator
from scripts.train import train_epoch, train_epoch_all, validate_epoch_all

SequenceGenerator = Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass
class TrainingResult:
    """Container summarising losses collected during the training loop."""

    val_losses: list[torch.Tensor]
    train_losses: list[torch.Tensor]
    steps: list[int]


def _rmse_tensor(tensor: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(tensor.float() ** 2)).item()


def _wrap_generator_as_batch_generator(
    generator: Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    batch_size: Optional[int] = None,
    batches_per_epoch: int = 1,
) -> BatchGenerator:
    """
    Wrap a callable that returns ``(transformer_inputs, probs, loss_lower_bound)``
    into a ``BatchGenerator`` instance so we can reuse the project's training helpers.
    """
    transformer_inputs, probs, _ = generator()

    # Ensure tensors live on the designated device
    transformer_inputs = transformer_inputs.to(device)
    probs = probs.to(device)

    if batch_size is None:
        batch_size = transformer_inputs.shape[0]

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if batches_per_epoch <= 0:
        raise ValueError("batches_per_epoch must be positive")

    return BatchGenerator(transformer_inputs, probs, batches_per_epoch, batch_size, device)


def training_loop(
    model: HookedTransformer,
    optimizer: Optimizer,
    generator: SequenceGenerator,
    *,
    num_epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    weighted_loss: bool = False,
    batch_size: Optional[int] = None,
    batches_per_epoch: int = 1,
    epoch_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    normalize_by_lower_bound: bool = False,
    loss_lower_bound: Optional[torch.Tensor] = None,
) -> TrainingResult:
    """
    Train a ``HookedTransformer`` using sequences produced by ``generator``.

    Args:
        model: TransformerLens model to train.
        optimizer: Optimiser configured for the model.
        generator: Callable returning a tuple ``(transformer_inputs, probs, loss_lower_bound)``
            each time it is invoked. The tensors should be PT tensors on the same device
            expected by the model, and ``probs`` should sum to one.
        num_epochs: Number of training epochs to execute.
        scheduler: Optional learning-rate scheduler.
        weighted_loss: When ``True``, use the project's ``train_epoch_all`` routine that weights
            each sequence by its probability; otherwise, default to standard sampling.
        normalize_by_lower_bound: When ``True``, normalize losses by dividing by loss_lower_bound
            (following scripts/train.py approach). Default is False.
        loss_lower_bound: Per-position lower bound on achievable loss. Required if
            normalize_by_lower_bound is True.

    Returns:
        ``TrainingResult`` containing per-epoch training and validation losses.
    """
    device = next(model.parameters()).device
    model.to(device)

    dataloader = _wrap_generator_as_batch_generator(
        generator,
        device,
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
    )

    # Validate normalization parameters
    if normalize_by_lower_bound:
        if loss_lower_bound is None:
            raise ValueError("loss_lower_bound must be provided when normalize_by_lower_bound=True")
        loss_lower_bound = loss_lower_bound.to(device)

    train_losses: list[torch.Tensor] = []
    val_losses: list[torch.Tensor] = []
    steps: list[int] = []

    total_steps = num_epochs if weighted_loss else num_epochs * batches_per_epoch
    progress = tqdm(total=total_steps, desc="Training", unit="step")

    # Evaluate before any updates to capture random-initialization metrics.
    initial_val_loss = validate_epoch_all(model, dataloader, scheduler=None)
    if normalize_by_lower_bound:
        initial_val_loss = initial_val_loss / loss_lower_bound
    initial_val_loss = initial_val_loss.detach().cpu()
    val_losses.append(initial_val_loss)
    train_losses.append(initial_val_loss)
    steps.append(-1)
    progress.set_postfix(
        {
            "step": -1,
            "train_rmse": f"{_rmse_tensor(initial_val_loss):.4f}",
            "val_rmse": f"{_rmse_tensor(initial_val_loss):.4f}",
        }
    )
    if epoch_callback is not None:
        epoch_callback(-1, train_losses[-1], val_losses[-1])

    step_idx = 0
    for epoch in range(num_epochs):
        if weighted_loss:
            epoch_loss = train_epoch_all(model, optimizer, dataloader, scheduler)
            val_loss = validate_epoch_all(model, dataloader, scheduler=None)

            if normalize_by_lower_bound:
                epoch_loss = epoch_loss / loss_lower_bound
                val_loss = val_loss / loss_lower_bound

            epoch_loss = epoch_loss.detach().cpu()
            val_loss = val_loss.detach().cpu()

            train_losses.append(epoch_loss)
            val_losses.append(val_loss)
            steps.append(step_idx)

            progress.update(1)
            progress.set_postfix(
                {
                    "step": step_idx,
                    "train_rmse": f"{_rmse_tensor(epoch_loss):.4f}",
                    "val_rmse": f"{_rmse_tensor(val_loss):.4f}",
                }
            )
            if epoch_callback is not None:
                epoch_callback(step_idx, epoch_loss, val_loss)

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(_rmse_tensor(val_loss))
            elif scheduler:
                scheduler.step()

            step_idx += 1
            continue

        batch_iter = iter(dataloader)
        for _ in range(dataloader.batches_per_epoch):
            inputs, targets = next(batch_iter)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)

            batch_size_tensor, seq_length, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1).to(torch.int64)
            loss_matrix = F.cross_entropy(logits_flat, targets_flat, reduction="none").reshape(
                batch_size_tensor, seq_length
            )

            loss_matrix.mean().backward()
            optimizer.step()

            step_loss = loss_matrix.mean(dim=0)
            val_loss = validate_epoch_all(model, dataloader, scheduler=None)

            if normalize_by_lower_bound:
                step_loss = step_loss / loss_lower_bound
                val_loss = val_loss / loss_lower_bound

            step_loss_cpu = step_loss.detach().cpu()
            val_loss_cpu = val_loss.detach().cpu()

            train_losses.append(step_loss_cpu)
            val_losses.append(val_loss_cpu)
            steps.append(step_idx)

            progress.update(1)
            progress.set_postfix(
                {
                    "step": step_idx,
                    "train_rmse": f"{_rmse_tensor(step_loss_cpu):.4f}",
                    "val_rmse": f"{_rmse_tensor(val_loss_cpu):.4f}",
                }
            )

            if epoch_callback is not None:
                epoch_callback(step_idx, step_loss_cpu, val_loss_cpu)

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(_rmse_tensor(val_loss_cpu))
            elif scheduler:
                scheduler.step()

            step_idx += 1

    progress.close()

    return TrainingResult(val_losses=val_losses, train_losses=train_losses, steps=steps)


__all__ = ["training_loop", "TrainingResult"]
