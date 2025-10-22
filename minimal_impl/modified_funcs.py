"""
GPU-accelerated helpers for belief regression on the MM3 process.

These utilities keep intermediate tensors on the target device so that the
least-squares projection can execute on the GPU when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

@dataclass
class _LengthTable:
    """Lookup table for a particular prefix length."""

    sorted_codes: torch.Tensor
    beliefs: torch.Tensor
    indices: Optional[torch.Tensor]
    unnormalized: Optional[torch.Tensor]


def _encode_paths(paths: torch.Tensor, base: int) -> torch.Tensor:
    """Encode sequences of tokens into a single integer per row."""
    codes = torch.zeros(paths.shape[0], dtype=torch.int64, device=paths.device)
    for col in range(paths.shape[1]):
        codes = codes * base + paths[:, col].to(torch.int64)
    return codes


def _round_tensor(tensor: torch.Tensor, decimals: int = 5) -> torch.Tensor:
    factor = 10 ** decimals
    return torch.round(tensor * factor) / factor


def _prepare_length_tables(
    tree_paths: List[List[int]],
    tree_beliefs: List[Iterable[float]],
    tree_unnormalized: Optional[List[Iterable[float]]],
    msp_belief_index: Dict[Tuple[float, ...], int],
    max_length: int,
    *,
    base: int,
    device: torch.device,
) -> Dict[int, _LengthTable]:
    """
    Build torch lookup tables for every prefix length up to ``max_length``.
    """
    tables: Dict[int, _LengthTable] = {}

    for length in range(1, max_length + 1):
        indices_for_length = [
            idx for idx, path in enumerate(tree_paths) if len(path) == length
        ]
        if not indices_for_length:
            continue

        paths_tensor = torch.tensor(
            [tree_paths[idx] for idx in indices_for_length],
            device=device,
            dtype=torch.int64,
        )
        codes = _encode_paths(paths_tensor, base)
        sort_idx = torch.argsort(codes)
        sorted_codes = codes[sort_idx]

        beliefs_np = np.asarray(
            [np.squeeze(tree_beliefs[idx]) for idx in indices_for_length],
            dtype=np.float32,
        )
        beliefs_tensor = torch.as_tensor(beliefs_np, device=device, dtype=torch.float32)
        beliefs_tensor = _round_tensor(beliefs_tensor)
        beliefs_tensor = beliefs_tensor[sort_idx]

        if msp_belief_index:
            belief_indices = []
            for idx in indices_for_length:
                belief_tuple = tuple(
                    round(float(x), 5) for x in np.squeeze(tree_beliefs[idx])
                )
                belief_indices.append(msp_belief_index[belief_tuple])
            indices_tensor = torch.as_tensor(
                belief_indices, device=device, dtype=torch.int64
            )[sort_idx]
        else:
            indices_tensor = None

        if tree_unnormalized is not None:
            unnorm_np = np.asarray(
                [np.squeeze(tree_unnormalized[idx]) for idx in indices_for_length],
                dtype=np.float32,
            )
            unnorm_tensor = torch.as_tensor(
                unnorm_np, device=device, dtype=torch.float32
            )
            unnorm_tensor = unnorm_tensor[sort_idx]
        else:
            unnorm_tensor = None

        tables[length] = _LengthTable(
            sorted_codes=sorted_codes,
            beliefs=beliefs_tensor,
            indices=indices_tensor,
            unnormalized=unnorm_tensor,
        )

    return tables


def get_beliefs_for_nn_inputs_gpu(
    nn_inputs: torch.Tensor,
    msp_belief_index: Dict[Tuple[float, ...], int],
    tree_paths: List[List[int]],
    tree_beliefs: List[Iterable[float]],
    tree_unnormalized_beliefs: Optional[List[Iterable[float]]],
    probs_dict: Optional[Dict[Tuple[int, ...], float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised variant of ``get_beliefs_for_nn_inputs`` that keeps computations
    on the provided device whenever possible.
    """
    device = nn_inputs.device
    batch, n_ctx = nn_inputs.shape
    base = int(nn_inputs.max().item()) + 1

    tables = _prepare_length_tables(
        tree_paths=tree_paths,
        tree_beliefs=tree_beliefs,
        tree_unnormalized=tree_unnormalized_beliefs,
        msp_belief_index=msp_belief_index,
        max_length=n_ctx,
        base=base,
        device=device,
    )

    sample_length_table = next(iter(tables.values()))
    belief_dim = sample_length_table.beliefs.shape[1]

    beliefs = torch.zeros(batch, n_ctx, belief_dim, device=device, dtype=torch.float32)
    belief_indices = torch.zeros(batch, n_ctx, device=device, dtype=torch.int64)

    unnormalized_out = (
        torch.zeros(batch, n_ctx, belief_dim, device=device, dtype=torch.float32)
        if tree_unnormalized_beliefs is not None
        else None
    )

    for length, table in tables.items():
        prefixes = nn_inputs[:, :length]
        prefix_codes = _encode_paths(prefixes, base)

        match_idx = torch.searchsorted(table.sorted_codes, prefix_codes)
        match_idx = torch.clamp(match_idx, max=table.sorted_codes.numel() - 1)
        if not torch.all(table.sorted_codes[match_idx] == prefix_codes):
            raise KeyError(
                "Failed to match prefixes when constructing belief tensors."
            )

        beliefs[:, length - 1] = table.beliefs[match_idx]
        if table.indices is not None:
            belief_indices[:, length - 1] = table.indices[match_idx]
        if unnormalized_out is not None and table.unnormalized is not None:
            unnormalized_out[:, length - 1] = table.unnormalized[match_idx]

    if probs_dict is not None:
        # Probabilities depend only on the full sequence; broadcast across ctx.
        sequences = nn_inputs.detach().cpu().tolist()
        sample_probs = torch.tensor(
            [float(probs_dict[tuple(seq)]) for seq in sequences],
            dtype=torch.float32,
            device=device,
        )
        probs = sample_probs.unsqueeze(1).expand(-1, n_ctx)
    else:
        probs = None

    if probs is None and unnormalized_out is None:
        return beliefs, belief_indices
    if probs is None:
        return beliefs, belief_indices, unnormalized_out  # type: ignore[return-value]
    if unnormalized_out is None:
        return beliefs, belief_indices, probs  # type: ignore[return-value]
    return beliefs, belief_indices, probs, unnormalized_out  # type: ignore[return-value]


def run_belief_regression_gpu(
    model,
    dataset,
    *,
    process,
    mm3_cfg: Dict[str, Any],
    activation_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    GPU-enabled belief regression that mirrors ``mm3.run_belief_regression``.
    """
    bos = mm3_cfg.get("bos", False)
    n_ctx = mm3_cfg.get("n_ctx", dataset.transformer_inputs.shape[1] - 1)

    seq_len = n_ctx + 1
    msp_depth = seq_len + (1 if bos else 2)
    msp = process.derive_mixed_state_tree(depth=msp_depth)

    tree_paths = msp.paths
    tree_beliefs = msp.belief_states
    tree_unnormalized = msp.unnorm_belief_states
    path_probs = msp.path_probs

    msp_beliefs = [
        tuple(round(float(b), 5) for b in np.squeeze(belief))
        for belief in tree_beliefs
    ]
    msp_belief_index = {
        tuple_belief: idx for idx, tuple_belief in enumerate(set(msp_beliefs))
    }
    probs_dict = (
        {tuple(path): float(prob) for path, prob in zip(tree_paths, path_probs)}
        if path_probs is not None
        else None
    )

    device = torch.device(model.cfg.device)
    contexts = dataset.transformer_inputs[:, :-1].to(device=device, dtype=torch.int64)

    beliefs_out = get_beliefs_for_nn_inputs_gpu(
        contexts,
        msp_belief_index,
        tree_paths,
        tree_beliefs,
        tree_unnormalized,
        probs_dict,
    )
    belief_states = beliefs_out[0].to(device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        activations_to_use = activation_names or ["ln_final.hook_normalized"]
        _, cache = model.run_with_cache(
            contexts,
            names_filter=lambda name: name in activations_to_use,
        )
        collected = []
        for name in activations_to_use:
            if name not in cache:
                raise KeyError(f"Activation {name!r} not found in cache.")
            collected.append(cache[name].detach())
        activations = (
            torch.cat(collected, dim=-1) if len(collected) > 1 else collected[0]
        )
    if was_training:
        model.train()

    acts_flat = activations.reshape(-1, activations.shape[-1]).float()
    beliefs_flat = belief_states.reshape(-1, belief_states.shape[-1]).float()

    # Diagnostic checks
    if torch.isnan(acts_flat).any() or torch.isinf(acts_flat).any():
        print(f"WARNING: NaN or Inf in activations! NaN: {torch.isnan(acts_flat).sum()}, Inf: {torch.isinf(acts_flat).sum()}")
    if torch.isnan(beliefs_flat).any() or torch.isinf(beliefs_flat).any():
        print(f"WARNING: NaN or Inf in beliefs! NaN: {torch.isnan(beliefs_flat).sum()}, Inf: {torch.isinf(beliefs_flat).sum()}")

    print(f"Debug shapes - acts_flat: {acts_flat.shape}, beliefs_flat: {beliefs_flat.shape}")
    print(f"Debug stats - acts mean: {acts_flat.mean().item():.6f}, std: {acts_flat.std().item():.6f}")
    print(f"Debug stats - beliefs mean: {beliefs_flat.mean().item():.6f}, std: {beliefs_flat.std().item():.6f}")

    # Use ridge regression instead of plain lstsq to avoid numerical instability
    rcond = 1e-10  # Regularization strength
    lstsq_result = torch.linalg.lstsq(acts_flat, beliefs_flat, rcond=rcond)
    preds = acts_flat @ lstsq_result.solution
    residuals = preds - beliefs_flat
    mse = torch.mean(residuals.pow(2))
    rmse_val = torch.sqrt(mse).item()

    total_variance = torch.mean(
        (beliefs_flat - beliefs_flat.mean(dim=0, keepdim=True)).pow(2)
    )

    print(f"Debug regression - MSE: {mse.item():.6e}, Total Variance: {total_variance.item():.6e}")
    print(f"Debug regression - preds mean: {preds.mean().item():.6f}, preds std: {preds.std().item():.6f}")

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


__all__ = [
    "get_beliefs_for_nn_inputs_gpu",
    "run_belief_regression_gpu",
]
