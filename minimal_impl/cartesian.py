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
from torch.utils.data import IterableDataset


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


class ProductBatchGenerator(IterableDataset):
    """
    Sampling-based batch generator for cartesian product distributions.

    Instead of materializing all O(|A| × |B|) sequences, this generator:
    1. Stores only the component sequences (A and B)
    2. Samples from each component independently during iteration
    3. Constructs combined sequences on-the-fly

    Memory: O(|A| + |B|) instead of O(|A| × |B|)
    """

    def __init__(
        self,
        inputs_a: torch.Tensor,
        probs_a: torch.Tensor,
        inputs_b: torch.Tensor,
        probs_b: torch.Tensor,
        batches_per_epoch: int,
        batch_size: int,
        device: torch.device,
    ):
        self.inputs_a = inputs_a.to(device)
        self.probs_a = probs_a.to(device)
        self.inputs_b = inputs_b.to(device)
        self.probs_b = probs_b.to(device)
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.device = device

        # Infer parameters
        self.seq_len = inputs_a.shape[1]
        self.vocab_b = int(inputs_b.max().item()) + 1
        self.tokens_per_epoch = batches_per_epoch * batch_size * (self.seq_len - 1)

    def __iter__(self):
        """Sample batches by independently sampling from component distributions."""
        for _ in range(self.batches_per_epoch):
            # Sample indices from each component
            indices_a = torch.multinomial(self.probs_a, self.batch_size, replacement=True)
            indices_b = torch.multinomial(self.probs_b, self.batch_size, replacement=True)

            # Get the component sequences
            batch_a = self.inputs_a[indices_a]  # (batch_size, seq_len)
            batch_b = self.inputs_b[indices_b]  # (batch_size, seq_len)

            # Combine: encode as base*vocab_a + vocab_b
            combined = batch_a * self.vocab_b + batch_b  # (batch_size, seq_len)

            # Split into inputs and targets
            X = combined[:, :-1]
            Y = combined[:, 1:]

            yield X, Y

    def validation_data(self, max_samples=10000):
        """
        For validation with large product spaces, sample instead of enumerating.

        Args:
            max_samples: Maximum number of samples to use for validation.
                        If the product is smaller, use all sequences.
        """
        num_a = len(self.inputs_a)
        num_b = len(self.inputs_b)
        total_size = num_a * num_b

        if total_size <= max_samples:
            # Small enough to enumerate all sequences
            inputs_a_exp = self.inputs_a.unsqueeze(1).expand(num_a, num_b, self.seq_len)
            inputs_b_exp = self.inputs_b.unsqueeze(0).expand(num_a, num_b, self.seq_len)
            combined = (inputs_a_exp * self.vocab_b + inputs_b_exp).reshape(num_a * num_b, self.seq_len)

            # Compute product probabilities
            probs_a_exp = self.probs_a.unsqueeze(1).expand(num_a, num_b)
            probs_b_exp = self.probs_b.unsqueeze(0).expand(num_a, num_b)
            combined_probs = (probs_a_exp * probs_b_exp).reshape(-1)
        else:
            # Too large - sample instead
            print(f"[DEBUG] Product space too large ({total_size}), sampling {max_samples} for validation")
            indices_a = torch.multinomial(self.probs_a, max_samples, replacement=True)
            indices_b = torch.multinomial(self.probs_b, max_samples, replacement=True)

            batch_a = self.inputs_a[indices_a]
            batch_b = self.inputs_b[indices_b]
            combined = batch_a * self.vocab_b + batch_b

            # Compute probabilities for sampled sequences
            probs_a_sampled = self.probs_a[indices_a]
            probs_b_sampled = self.probs_b[indices_b]
            combined_probs = probs_a_sampled * probs_b_sampled

        X = combined[:, :-1]
        Y = combined[:, 1:]

        return X, Y, combined_probs


def generate_mm3_bloch_product_data_sampled(
    mm3_params: Dict[str, float],
    bloch_params: Dict[str, float],
    *,
    n_ctx: int,
    bos: bool,
    batches_per_epoch: int,
    batch_size: int,
    device: str | torch.device,
) -> Tuple[ProductBatchGenerator, torch.Tensor, int]:
    """
    Build a sampling-based cartesian product distribution over (MM3, Bloch) pairs.

    This version uses O(|MM3| + |Bloch|) memory instead of O(|MM3| × |Bloch|),
    making it feasible to train on longer context lengths (n_ctx=7).

    During training, sequences are sampled on-the-fly from the product distribution.
    During validation, all sequences are enumerated (this can be slow/memory-intensive).
    """
    print(f"[DEBUG] Starting generate_mm3_bloch_product_data_sampled with n_ctx={n_ctx}, bos={bos}")
    torch_device = _as_device(device)

    print(f"[DEBUG] Extracting parameters...")
    mm3_params = _extract_mm3_params(mm3_params)
    bloch_params = _extract_bloch_params(bloch_params)
    print(f"[DEBUG] MM3 params: {mm3_params}, Bloch params: {bloch_params}")

    # Generate component datasets (on CPU to save GPU memory)
    print(f"[DEBUG] About to generate MM3 dataset (expected size: 3^{n_ctx} = {3**n_ctx} sequences)...")
    mm3_dataset = generate_mm3_transformer_data(
        n_ctx=n_ctx,
        bos=bos,
        device="cpu",
        as_numpy=False,
        **mm3_params,
    )
    print(f"[DEBUG] MM3 dataset created: {len(mm3_dataset.transformer_inputs)} sequences")

    print(f"[DEBUG] About to generate Bloch dataset (expected size: 4^{n_ctx} = {4**n_ctx} sequences)...")
    bloch_dataset = generate_bloch_transformer_data(
        n_ctx=n_ctx,
        bos=bos,
        device="cpu",
        as_numpy=False,
        **bloch_params,
    )
    print(f"[DEBUG] Bloch dataset created: {len(bloch_dataset.transformer_inputs)} sequences")

    # Validate sequence lengths match
    if mm3_dataset.transformer_inputs.shape[1] != bloch_dataset.transformer_inputs.shape[1]:
        raise ValueError(
            "MM3 and Bloch datasets must have the same sequence length. "
            "Ensure n_ctx and bos match for both components."
        )

    # Create sampling-based batch generator
    print(f"Creating ProductBatchGenerator with:")
    print(f"  MM3 sequences: {len(mm3_dataset.transformer_inputs)}")
    print(f"  Bloch sequences: {len(bloch_dataset.transformer_inputs)}")
    print(f"  Total product size would be: {len(mm3_dataset.transformer_inputs) * len(bloch_dataset.transformer_inputs)}")
    print(f"  Memory for training: ~{(len(mm3_dataset.transformer_inputs) + len(bloch_dataset.transformer_inputs)) * 7 * 4 / 1024 / 1024:.2f} MB")

    dataloader = ProductBatchGenerator(
        inputs_a=mm3_dataset.transformer_inputs,
        probs_a=mm3_dataset.probabilities,
        inputs_b=bloch_dataset.transformer_inputs,
        probs_b=bloch_dataset.probabilities,
        batches_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        device=torch_device,
    )
    print("ProductBatchGenerator created successfully!")

    # Loss lower bound is additive for independent processes
    loss_lower_bound = (mm3_dataset.loss_lower_bound + bloch_dataset.loss_lower_bound).to(torch_device)

    # Vocabulary size is product of component vocabularies
    d_vocab = _infer_vocab_size(mm3_dataset.transformer_inputs) * _infer_vocab_size(bloch_dataset.transformer_inputs)

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
