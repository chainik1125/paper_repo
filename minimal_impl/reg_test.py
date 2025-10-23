"""
Test script to verify regression implementation by loading a model from wandb
and performing the same regression as in mm3.py.

This is a diagnostic tool to compare against the full regression pipeline.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig

try:
    import wandb
except ImportError:
    wandb = None

from epsilon_transformers.analysis.activation_analysis import prepare_msp_data


def load_wandb_checkpoint(
    entity: str,
    project: str,
    run_path: str,
    checkpoint_name: str = "final_model.pt",
    api_key: Optional[str] = None,
    artifact_path: Optional[str] = None,
) -> Path:
    """
    Download a checkpoint from wandb.

    Args:
        entity: WandB entity name
        project: WandB project name
        run_path: Full run path (entity/project/run_id)
        checkpoint_name: Name of checkpoint to download (e.g., "final_model.pt", "179200")
        api_key: Optional WandB API key
        artifact_path: Optional direct artifact path (e.g., "dmitry2-uiuc/epsilon-bloch-walk_manual/b9y0lvw8-model-179200:v0")

    Returns:
        Path to downloaded checkpoint
    """
    if wandb is None:
        raise RuntimeError("wandb is not installed")

    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    api = wandb.Api()

    # If direct artifact path is provided, use it
    if artifact_path:
        print(f"Downloading from artifact path: {artifact_path}")
        artifact = api.artifact(artifact_path)
        download_dir = Path(tempfile.mkdtemp(prefix="reg_test_"))
        artifact.download(root=str(download_dir))
        ckpt_files = list(download_dir.glob("*.pt"))
        if ckpt_files:
            print(f"Downloaded checkpoint from artifact: {ckpt_files[0]}")
            return ckpt_files[0]
        raise FileNotFoundError(f"No .pt file found in artifact {artifact_path}")

    run = api.run(run_path)

    # Look for checkpoint in artifacts by name matching
    print(f"Searching artifacts for: {checkpoint_name}")
    for artifact in run.logged_artifacts():
        if artifact.type != "model":
            continue

        artifact_name = artifact.name
        print(f"  Found artifact: {artifact_name}")

        # Check if checkpoint_name appears in the artifact name
        if checkpoint_name in artifact_name:
            print(f"  -> Matched! Downloading artifact: {artifact_name}")
            download_dir = Path(tempfile.mkdtemp(prefix="reg_test_"))
            artifact.download(root=str(download_dir))

            ckpt_files = list(download_dir.glob("*.pt"))
            if ckpt_files:
                print(f"Downloaded checkpoint: {ckpt_files[0]}")
                return ckpt_files[0]

    raise FileNotFoundError(f"No checkpoint found matching '{checkpoint_name}' in run {run_path}")


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def instantiate_model(run_cfg: Dict[str, Any], checkpoint_path: Path, device: str = "cpu") -> HookedTransformer:
    """
    Create a HookedTransformer model and load checkpoint weights.

    Args:
        run_cfg: Run configuration dict
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded HookedTransformer model
    """
    model_cfg = dict(run_cfg["model_config"])
    model_cfg.setdefault("device", device)

    # Ensure d_vocab is set
    if "d_vocab" not in model_cfg:
        process_cfg = run_cfg["process_config"]
        name = process_cfg["name"]
        if name == "tom_quantum":
            vocab_size = 4
        elif name == "mess3":
            vocab_size = 3
        else:
            vocab_size = 5
        model_cfg["d_vocab"] = vocab_size

    # Ensure d_head is set (required by HookedTransformerConfig)
    if "d_head" not in model_cfg:
        if "d_model" in model_cfg and "n_heads" in model_cfg:
            model_cfg["d_head"] = model_cfg["d_model"] // model_cfg["n_heads"]
        else:
            model_cfg["d_head"] = 64  # default fallback

    # Ensure act_fn is set
    if "act_fn" not in model_cfg:
        model_cfg["act_fn"] = "gelu"  # default activation function

    # Convert dtype string to torch dtype
    if "dtype" in model_cfg and isinstance(model_cfg["dtype"], str):
        model_cfg["dtype"] = getattr(torch, model_cfg["dtype"])

    # Create model
    hook_cfg = HookedTransformerConfig(**model_cfg)
    model = HookedTransformer(hook_cfg)

    # Load weights
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model


def run_simple_regression(
    model: HookedTransformer,
    nn_inputs: torch.Tensor,
    nn_beliefs: torch.Tensor,
    nn_probs: torch.Tensor,
    rcond: float = 1e-10,
    use_weights: bool = True,
) -> Dict[str, float]:
    """
    Perform simple regression as in mm3.py's run_belief_regression function.

    Args:
        model: HookedTransformer model
        nn_inputs: Input sequences (batch, seq_len)
        nn_beliefs: Target belief states (batch, seq_len, belief_dim)
        nn_probs: Sequence probabilities (batch, seq_len)
        rcond: Regularization parameter for lstsq
        use_weights: Whether to use weighted least squares (default: True)

    Returns:
        Dictionary with regression metrics
    """
    device = model.cfg.device

    # Use contexts (all but last token) for activation extraction
    contexts = nn_inputs[:, :-1].to(torch.int64).to(device)
    # nn_beliefs should match the contexts shape (without the last position)
    # If nn_beliefs is (batch, seq_len, belief_dim), we need (batch, seq_len-1, belief_dim)
    belief_states = nn_beliefs[:, :-1, :].to(device)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        # Extract activations from all layers
        n_layers = model.cfg.n_layers
        activation_keys = (
            ['blocks.0.hook_resid_pre'] +
            [f'blocks.{i}.hook_resid_post' for i in range(n_layers)] +
            ['ln_final.hook_normalized']
        )

        _, cache = model.run_with_cache(
            contexts,
            names_filter=lambda name: name in activation_keys,
        )

        # Combine activations from all layers by concatenating along feature dimension
        all_activations = []
        for key in activation_keys:
            all_activations.append(cache[key].detach())

        # Concatenate: (batch, seq, d_model) -> (batch, seq, d_model * n_layers)
        activations = torch.cat(all_activations, dim=-1)

    if was_training:
        model.train()

    # Flatten for regression
    acts_flat = activations.reshape(-1, activations.shape[-1]).float()
    beliefs_flat = belief_states.reshape(-1, belief_states.shape[-1]).float()
    probs_flat = nn_probs[:, :-1].reshape(-1).float()  # Match the context length

    print(f"Activation shape: {acts_flat.shape}")
    print(f"Belief shape: {beliefs_flat.shape}")
    print(f"Probs shape: {probs_flat.shape}")
    print(f"Activation stats: min={acts_flat.min():.4f}, max={acts_flat.max():.4f}, mean={acts_flat.mean():.4f}, std={acts_flat.std():.4f}")
    print(f"Belief stats: min={beliefs_flat.min():.4f}, max={beliefs_flat.max():.4f}, mean={beliefs_flat.mean():.4f}, std={beliefs_flat.std():.4f}")
    print(f"Probs stats: min={probs_flat.min():.4f}, max={probs_flat.max():.4f}, sum={probs_flat.sum():.4f}")
    print(f"Rcond: {rcond}")
    print(f"Use weights: {use_weights}")

    # Check for NaNs or Infs
    if torch.isnan(acts_flat).any() or torch.isinf(acts_flat).any():
        print("WARNING: Activations contain NaN or Inf values!")
    if torch.isnan(beliefs_flat).any() or torch.isinf(beliefs_flat).any():
        print("WARNING: Beliefs contain NaN or Inf values!")

    # Move to CPU for lstsq to avoid CUDA issues
    print("Moving data to CPU for lstsq computation...")
    acts_cpu = acts_flat.cpu()
    beliefs_cpu = beliefs_flat.cpu()
    probs_cpu = probs_flat.cpu()

    # Normalize probabilities
    probs_norm = probs_cpu / probs_cpu.sum()

    if use_weights:
        print("Using WEIGHTED least squares (matching full pipeline)")
        # Apply weights by multiplying by sqrt(weights)
        # This matches the full regression pipeline approach
        sqrt_weights = torch.sqrt(probs_norm).unsqueeze(1)
        acts_weighted = acts_cpu * sqrt_weights
        beliefs_weighted = beliefs_cpu * sqrt_weights
    else:
        print("Using UNWEIGHTED least squares (simple approach)")
        acts_weighted = acts_cpu
        beliefs_weighted = beliefs_cpu

    try:
        lstsq_result = torch.linalg.lstsq(acts_weighted, beliefs_weighted, rcond=rcond)
        print(f"lstsq solution shape: {lstsq_result.solution.shape}")
        print(f"lstsq rank type: {type(lstsq_result.rank)}")
        print(f"lstsq rank value: {lstsq_result.rank}")

        # Check solution for NaNs
        if torch.isnan(lstsq_result.solution).any():
            print("WARNING: Solution contains NaN values!")

        # Make predictions on UNWEIGHTED data
        preds = acts_cpu @ lstsq_result.solution
        print(f"Predictions shape: {preds.shape}")
        print(f"Predictions stats: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")

        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print("WARNING: Predictions contain NaN or Inf values!")
    except Exception as e:
        print(f"ERROR in lstsq: {e}")
        raise

    # Compute metrics
    if use_weights:
        # For weighted regression, compute weighted metrics
        residuals = preds - beliefs_cpu
        weighted_residuals = residuals * sqrt_weights

        # Weighted MSE
        mse = torch.mean(weighted_residuals.pow(2))
        rmse_val = torch.sqrt(mse).item()

        # Weighted R²
        total_variance = torch.mean((beliefs_weighted - beliefs_weighted.mean(dim=0, keepdim=True)).pow(2))
        residual_variance = torch.mean(weighted_residuals.pow(2))
        r_squared = 1.0 - (residual_variance / total_variance) if total_variance > 0 else float("nan")
    else:
        # Unweighted metrics
        residuals = preds - beliefs_cpu
        mse = torch.mean(residuals.pow(2))
        rmse_val = torch.sqrt(mse).item()

        total_variance = torch.mean(
            (beliefs_cpu - beliefs_cpu.mean(dim=0, keepdim=True)).pow(2)
        )
        r_squared = 1.0 - (mse / total_variance) if total_variance > 0 else float("nan")

    rank_value = lstsq_result.rank
    print(f"Rank before processing: {rank_value}, type: {type(rank_value)}")
    if isinstance(rank_value, torch.Tensor):
        if rank_value.numel() == 0:
            rank_value = float("nan")
        elif rank_value.numel() == 1:
            rank_value = int(rank_value.item())
        else:
            rank_value = float("nan")

    r_squared_value = float(r_squared.item()) if torch.isfinite(r_squared) else float("nan")

    return {
        "rmse": rmse_val,
        "mse": mse.item(),
        "rank": rank_value,
        "r2": r_squared_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test regression by loading model from wandb"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to run config YAML (e.g., mm3_config.yaml)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="WandB entity name",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="WandB project name",
    )
    parser.add_argument(
        "--run-path",
        type=str,
        required=True,
        help="Full WandB run path (entity/project/run_id)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final_model.pt",
        help="Checkpoint name to load (default: final_model.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--rcond",
        type=float,
        default=1e-10,
        help="Regularization parameter for lstsq (default: 1e-10)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="WandB API key (optional, can use WANDB_API_KEY env var)",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=None,
        help="Direct WandB artifact path (e.g., 'dmitry2-uiuc/epsilon-bloch-walk_manual/b9y0lvw8-model-179200:v0')",
    )
    parser.add_argument(
        "--test-random",
        action="store_true",
        help="Also test with a randomly initialized model (for comparison)",
    )

    args = parser.parse_args()

    # Load run config
    print(f"Loading config from {args.config}")
    run_cfg = load_config(args.config)
    run_cfg.setdefault("global_config", {})["device"] = args.device

    # Download checkpoint from wandb
    if args.artifact_path:
        print(f"Downloading checkpoint from artifact: {args.artifact_path}")
        checkpoint_path = load_wandb_checkpoint(
            entity=args.entity,
            project=args.project,
            run_path=args.run_path,
            checkpoint_name=args.checkpoint,
            api_key=args.api_key,
            artifact_path=args.artifact_path,
        )
    else:
        print(f"Downloading checkpoint from {args.run_path}")
        checkpoint_path = load_wandb_checkpoint(
            entity=args.entity,
            project=args.project,
            run_path=args.run_path,
            checkpoint_name=args.checkpoint,
            api_key=args.api_key,
        )

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = instantiate_model(run_cfg, checkpoint_path, device=args.device)
    print(f"Model loaded on {model.cfg.device}")

    # Prepare ground truth data
    print("Preparing ground truth belief states")
    nn_inputs, nn_beliefs, _, nn_probs, _ = prepare_msp_data(
        run_cfg,
        run_cfg["process_config"],
    )

    print(f"Number of sequences: {nn_inputs.shape[0]}")
    print(f"Sequence length: {nn_inputs.shape[1]}")
    print(f"Belief shape: {nn_beliefs.shape}")
    print(f"Belief dimension: {nn_beliefs.shape[-1]}")
    print(f"Sample beliefs[0, 0]: {nn_beliefs[0, 0]}")
    print(f"Sample beliefs stats: min={nn_beliefs.min():.4f}, max={nn_beliefs.max():.4f}")

    # Run regression
    print("\nRunning simple regression (mm3.py style) on TRAINED model")
    metrics = run_simple_regression(
        model,
        nn_inputs,
        nn_beliefs,
        nn_probs,
        rcond=args.rcond,
    )

    print("\n=== Trained Model Regression Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Optionally test with random initialization
    if args.test_random:
        print("\n" + "="*50)
        print("Testing with RANDOMLY INITIALIZED model for comparison")
        print("="*50)

        # Create a new random model with same architecture
        random_model = instantiate_model(run_cfg, checkpoint_path, device=args.device)

        # Reinitialize with random weights
        for param in random_model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        random_model.eval()

        print("Running regression on randomly initialized model")
        random_metrics = run_simple_regression(
            random_model,
            nn_inputs,
            nn_beliefs,
            nn_probs,
            rcond=args.rcond,
        )

        print("\n=== Random Model Regression Results ===")
        for key, value in random_metrics.items():
            print(f"{key}: {value}")

        print("\n=== Comparison ===")
        print(f"Trained R²: {metrics['r2']:.6f}")
        print(f"Random R²:  {random_metrics['r2']:.6f}")
        print(f"Difference: {metrics['r2'] - random_metrics['r2']:.6f}")

        if random_metrics['r2'] > 0.5:
            print("\n⚠️  WARNING: Random model has suspiciously high R²!")
            print("This suggests an issue with the regression implementation.")
        else:
            print("\n✓ Random model has low R² as expected.")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
