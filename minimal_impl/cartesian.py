"""
Helpers for constructing composite training datasets (e.g., MM3 × Bloch cartesian products)
without touching the original training scripts.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from epsilon_transformers.training.dataloader import BatchGenerator
from minimal_impl.mm3 import generate_mm3_transformer_data
from minimal_impl.bloch import generate_bloch_transformer_data


def _as_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _make_batch_generator(
    transformer_inputs: torch.Tensor,
    probs: torch.Tensor,
    *,
    batches_per_epoch: int,
    batch_size: int,
    device: str | torch.device,
) -> BatchGenerator:
    torch_device = _as_device(device)
    transformer_inputs = transformer_inputs.to(torch_device)
    probs = probs.to(torch_device)
    return BatchGenerator(transformer_inputs, probs, batches_per_epoch, batch_size, torch_device)


def _infer_vocab_size(transformer_inputs: torch.Tensor) -> int:
    return int(transformer_inputs.max().item()) + 1


def _extract_mm3_params(cfg: Dict[str, float]) -> Dict[str, float]:
    params = {"x": cfg.get("x"), "a": cfg.get("a")}
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise ValueError(f"Missing MM3 parameters: {missing}")
    return params


def _extract_bloch_params(cfg: Dict[str, float]) -> Dict[str, float]:
    params = {"alpha": cfg.get("alpha"), "beta": cfg.get("beta")}
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise ValueError(f"Missing Bloch parameters: {missing}")
    return params


def generate_mm3_data(
    mm3_params: Dict[str, float],
    *,
    n_ctx: int,
    bos: bool,
    batches_per_epoch: int,
    batch_size: int,
    device: str | torch.device,
) -> Tuple[BatchGenerator, torch.Tensor, int]:
    params = _extract_mm3_params(mm3_params)
    dataset = generate_mm3_transformer_data(
        n_ctx=n_ctx,
        bos=bos,
        device="cpu",
        as_numpy=False,
        **params,
    )
    dataloader = _make_batch_generator(
        dataset.transformer_inputs,
        dataset.probabilities,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        device=device,
    )
    loss_lower_bound = dataset.loss_lower_bound.to(_as_device(device))
    d_vocab = _infer_vocab_size(dataset.transformer_inputs)
    return dataloader, loss_lower_bound, d_vocab


def generate_bloch_data(
    bloch_params: Dict[str, float],
    *,
    n_ctx: int,
    bos: bool,
    batches_per_epoch: int,
    batch_size: int,
    device: str | torch.device,
) -> Tuple[BatchGenerator, torch.Tensor, int]:
    params = _extract_bloch_params(bloch_params)
    dataset = generate_bloch_transformer_data(
        n_ctx=n_ctx,
        bos=bos,
        device="cpu",
        as_numpy=False,
        **params,
    )
    dataloader = _make_batch_generator(
        dataset.transformer_inputs,
        dataset.probabilities,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        device=device,
    )
    loss_lower_bound = dataset.loss_lower_bound.to(_as_device(device))
    d_vocab = _infer_vocab_size(dataset.transformer_inputs)
    return dataloader, loss_lower_bound, d_vocab


def generate_mm3_bloch_product_data(
    mm3_params: Dict[str, float],
    bloch_params: Dict[str, float],
    *,
    n_ctx: int,
    bos: bool,
    batches_per_epoch: int,
    batch_size: int,
    device: str | torch.device,
) -> Tuple[BatchGenerator, torch.Tensor, int]:
    """
    Build the exact cartesian product distribution over (MM3 token, Bloch token)
    pairs. Sequence probabilities factor as P(a, b) = P_mm3(a) * P_bloch(b).

    Warning: the number of sequences grows multiplicatively (|MM3| * |Bloch|).
    Use small context lengths to avoid exploding memory requirements.
    """
    torch_device = _as_device("cpu")  # work on CPU tensors first to avoid large GPU allocations

    mm3_params = _extract_mm3_params(mm3_params)
    bloch_params = _extract_bloch_params(bloch_params)

    mm3_dataset = generate_mm3_transformer_data(
        n_ctx=n_ctx,
        bos=bos,
        device=torch_device,
        as_numpy=False,
        **mm3_params,
    )
    bloch_dataset = generate_bloch_transformer_data(
        n_ctx=n_ctx,
        bos=bos,
        device=torch_device,
        as_numpy=False,
        **bloch_params,
    )

    inputs_a = mm3_dataset.transformer_inputs.to(torch.long)
    inputs_b = bloch_dataset.transformer_inputs.to(torch.long)

    if inputs_a.shape[1] != inputs_b.shape[1]:
        raise ValueError(
            "MM3 and Bloch datasets must have the same sequence length. "
            "Ensure n_ctx and bos match for both components."
        )

    num_a, seq_len = inputs_a.shape
    num_b = inputs_b.shape[0]

    inputs_a_exp = inputs_a.unsqueeze(1).expand(num_a, num_b, seq_len)
    inputs_b_exp = inputs_b.unsqueeze(0).expand(num_a, num_b, seq_len)

    base = _infer_vocab_size(inputs_b)
    combined_inputs = (inputs_a_exp * base + inputs_b_exp).reshape(num_a * num_b, seq_len)

    probs_a = mm3_dataset.probabilities.reshape(num_a, 1)
    probs_b = bloch_dataset.probabilities.reshape(1, num_b)
    combined_probs = (probs_a * probs_b).reshape(-1)

    dataloader = _make_batch_generator(
        combined_inputs,
        combined_probs,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        device=device,
    )

    loss_lower_bound = (mm3_dataset.loss_lower_bound + bloch_dataset.loss_lower_bound).to(_as_device(device))
    d_vocab = _infer_vocab_size(combined_inputs)
    return dataloader, loss_lower_bound, d_vocab
