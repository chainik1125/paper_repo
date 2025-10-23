"""
Simplified Fig2 - Single model visualization from W&B artifact.

Loads a model from a W&B artifact and generates Bloch process belief visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import yaml
from pathlib import Path
from typing import Optional, Tuple
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from matplotlib.patches import Patch

# Import activation analysis utilities
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from epsilon_transformers.training.networks import create_RNN
from transformer_lens import HookedTransformer, HookedTransformerConfig

try:
    import wandb
except ImportError:
    wandb = None


def download_artifact(artifact_path: str, destination: Path) -> Tuple[Path, dict]:
    """Download a W&B artifact and its config.

    Args:
        artifact_path: W&B artifact path (e.g., 'entity/project/artifact:version')
        destination: Local directory to download to

    Returns:
        Tuple of (checkpoint_path, config_dict)
    """
    if wandb is None:
        raise RuntimeError("wandb is not installed")

    api = wandb.Api()

    print(f"Downloading artifact: {artifact_path}")
    artifact = api.artifact(artifact_path)

    # Download artifact
    artifact_dir = Path(artifact.download(root=str(destination)))

    # Find checkpoint file
    pt_files = list(artifact_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in artifact {artifact_path}")

    checkpoint_path = pt_files[0]
    print(f"Checkpoint: {checkpoint_path}")

    # Get run and download config
    run = artifact.logged_by()
    print(f"Source run: {run.path}")

    # Download run config file
    try:
        config_files = [f.name for f in run.files() if 'run_config.yaml' in f.name]
        if config_files:
            config_file = run.file(config_files[0])
            config_file.download(root=str(destination), replace=True)

            config_path = destination / config_files[0]
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config from {config_files[0]}")
            return checkpoint_path, config
    except Exception as e:
        print(f"Warning: Could not download config: {e}")

    # Fallback: use run.config
    config = dict(run.config) if hasattr(run, 'config') else {}

    # If config is empty, infer from checkpoint
    if not config or 'model_config' not in config:
        print("Config not found, inferring from checkpoint...")
        config = infer_config_from_checkpoint(checkpoint_path)

    return checkpoint_path, config


def infer_config_from_checkpoint(checkpoint_path: Path) -> dict:
    """Infer model configuration from checkpoint state dict.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Configuration dictionary
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Get state dict
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    # Infer model parameters
    d_vocab = state['embed.W_E'].shape[0]
    d_model = state['embed.W_E'].shape[1]
    n_ctx_inferred = state['pos_embed.W_pos'].shape[0]
    n_heads = state['blocks.0.attn.W_Q'].shape[0]
    d_head = state['blocks.0.attn.W_Q'].shape[2]
    d_mlp = state['blocks.0.mlp.W_in'].shape[1]

    # Override n_ctx to 7 for faster computation
    n_ctx = 7

    # Count number of layers
    n_layers = len([k for k in state.keys() if k.startswith('blocks.') and '.ln1.w' in k])

    print(f"Inferred config: {n_layers} layers, {d_model} dims, {n_heads} heads, vocab={d_vocab}, n_ctx={n_ctx} (original: {n_ctx_inferred})")

    return {
        'model_config': {
            'n_layers': n_layers,
            'd_model': d_model,
            'n_ctx': n_ctx,
            'n_heads': n_heads,
            'd_head': d_head,
            'd_mlp': d_mlp,
            'd_vocab': d_vocab,
            'act_fn': 'relu',
            'normalization_type': 'LN',
            'attn_only': False,
            'dtype': 'float32',
            'seed': 42
        },
        'process_config': {
            'name': 'tom_quantum',
            'alpha': 1.0,  # Bloch process parameters
            'beta': np.sqrt(51)
        }
    }


def load_transformer_from_checkpoint(checkpoint_path: Path, config: dict, device: str = 'cpu'):
    """Load a transformer model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint
        config: Configuration dictionary
        device: Device to load on

    Returns:
        Loaded HookedTransformer model
    """
    model_config = config.get('model_config', config)

    # Prepare config for HookedTransformer
    if 'dtype' in model_config:
        if isinstance(model_config['dtype'], str):
            model_config['dtype'] = getattr(torch, model_config['dtype'].split('.')[-1])
    else:
        model_config['dtype'] = torch.float32

    model_config['device'] = device

    # Load weights first to infer d_vocab
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)

    # Infer d_vocab from checkpoint if not in config
    if 'd_vocab' not in model_config:
        # Infer from embedding matrix shape
        d_vocab = state_dict['embed.W_E'].shape[0]
        model_config['d_vocab'] = d_vocab
        print(f"Inferred d_vocab={d_vocab} from checkpoint")

    # Create model
    cfg = HookedTransformerConfig(**model_config)
    model = HookedTransformer(cfg)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded transformer: {model_config.get('n_layers', '?')} layers, "
          f"{model_config.get('d_model', '?')} dims, "
          f"{model_config.get('n_heads', '?')} heads")

    return model


def generate_bloch_data(config: dict, device: str = 'cpu'):
    """Generate Bloch process ground truth data.

    Args:
        config: Run configuration dictionary
        device: Device for tensors

    Returns:
        Tuple of (nn_inputs, beliefs, indices, probs)
    """
    print("Generating Bloch process ground truth data...")
    import time
    start = time.time()

    # Transform minimal_impl config format to standard format
    process_config = config.get('process_config', {})

    # Check if config is in minimal_impl format
    if 'mode' in process_config:
        mode = process_config['mode']
        # Get parameters for this mode
        mode_params = process_config.get(mode, {})
        # Create standard format with proper process name
        # Map common bloch aliases to tom_quantum
        if mode in ['bloch', 'bloch_walk', 'tomqa', 'tomqb']:
            std_process_config = {'name': 'tom_quantum'}
        else:
            std_process_config = {'name': mode}
        std_process_config.update(mode_params)

        # Create modified config
        std_config = config.copy()
        std_config['process_config'] = std_process_config
    else:
        # Assume config already in standard format
        std_config = config
        # Make sure it's set to tom_quantum if not already specified
        if 'name' not in std_config['process_config']:
            std_config['process_config']['name'] = 'tom_quantum'

    print(f"  Calling prepare_msp_data (alpha={std_config['process_config'].get('alpha')}, beta={std_config['process_config'].get('beta')}, n_ctx={std_config['model_config']['n_ctx']})...")
    nn_inputs, beliefs, indices, probs, _ = prepare_msp_data(std_config, std_config['model_config'])
    print(f"  prepare_msp_data took {time.time() - start:.1f}s")

    nn_inputs = nn_inputs.to(device)
    beliefs = beliefs.to(device)
    probs = probs.to(device)

    print(f"Generated {len(beliefs)} sequences, belief dims: {beliefs.shape[-1]}")
    print(f"  Probs shape: {probs.shape}")

    return nn_inputs, beliefs, indices, probs


def extract_transformer_activations(model, nn_inputs):
    """Extract residual stream activations from transformer.

    Args:
        model: HookedTransformer model
        nn_inputs: Input sequences

    Returns:
        Combined activations tensor
    """
    activation_keys = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    activation_keys.append('ln_final.hook_normalized')

    _, acts = model.run_with_cache(nn_inputs, names_filter=lambda x: x in activation_keys)

    # Combine all activations
    act_list = [acts[key] for key in sorted(acts.keys())]
    combined = torch.cat(act_list, dim=-1)

    return combined


def run_regression(activations, beliefs, weights, rcond=1e-10, n_splits=10):
    """Run K-fold weighted ridge regression.

    Args:
        activations: Neural activations (batch, seq_len, act_dim)
        beliefs: Ground truth beliefs (batch, seq_len, belief_dim)
        weights: Sample weights (batch,)
        rcond: Regularization parameter
        n_splits: Number of CV splits

    Returns:
        Dictionary with predictions and metrics
    """
    batch_size, seq_len, act_dim = activations.shape
    _, _, belief_dim = beliefs.shape

    # Flatten sequence dimension
    X = activations.reshape(-1, act_dim).cpu().numpy()
    y = beliefs.reshape(-1, belief_dim).cpu().numpy()

    # Flatten weights (already has seq_len dimension)
    sample_weights = weights.cpu().numpy()
    sample_weights = sample_weights.reshape(-1)  # Just flatten, don't repeat
    sample_weights = sample_weights / sample_weights.sum()

    # K-fold cross validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        w_train = sample_weights[train_idx]
        w_train = w_train / w_train.sum()

        # Weighted ridge regression
        alpha = rcond * np.trace(X_train.T @ np.diag(w_train) @ X_train)
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(X_train, y_train, sample_weight=w_train)

        predictions[test_idx] = reg.predict(X_test)

    # Compute metrics on flattened data
    residuals = y - predictions  # (batch*seq_len, belief_dim)

    # Compute per-sample error (summed across belief dimensions)
    per_sample_sq_error = (residuals ** 2).sum(axis=1)  # (batch*seq_len,)
    per_sample_abs_error = np.abs(residuals).sum(axis=1)  # (batch*seq_len,)

    # Weighted metrics
    rmse = np.sqrt(np.sum(sample_weights * per_sample_sq_error))
    mae = np.sum(sample_weights * per_sample_abs_error)

    # R² score
    y_mean = np.average(y, axis=0, weights=sample_weights)
    ss_tot = np.sum(sample_weights * ((y - y_mean) ** 2).sum(axis=1))
    ss_res = np.sum(sample_weights * per_sample_sq_error)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Reshape predictions for output
    predictions = predictions.reshape(batch_size, seq_len, belief_dim)

    return {
        'predicted_beliefs': predictions,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def project_to_simplex(data):
    """Project 3D belief data to 2D simplex coordinates."""
    x_temp = data[:, 0] - data[:, 1] / 2 - data[:, 2] / 2
    y_temp = np.sqrt(3) / 2 * (data[:, 1] - data[:, 2])
    x = -y_temp
    y = x_temp
    return x, y


def transform_for_alpha(weights, min_alpha=0.2, transformation='cbrt'):
    """Transform weights to alpha values for visualization."""
    weights = np.clip(weights, 0, 1)
    if transformation == 'cbrt':
        alpha = np.cbrt(weights)
    elif transformation == 'sqrt':
        alpha = np.sqrt(weights)
    else:
        alpha = weights
    return alpha * (1 - min_alpha) + min_alpha


def visualize_beliefs(gt_beliefs, pred_beliefs, weights, metrics, output_path):
    """Create visualization with ground truth and predictions for Bloch process.

    Args:
        gt_beliefs: Ground truth beliefs (N, belief_dim)
        pred_beliefs: Predicted beliefs (N, belief_dim)
        weights: Sample weights (N,)
        metrics: Dictionary with RMSE, MAE, R2
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3),
                            gridspec_kw={'width_ratios': [1, 1, 0.6]})

    # For Bloch process, use dimensions [1, 2] directly (no simplex projection)
    if gt_beliefs.shape[1] >= 3:
        x_gt, y_gt = gt_beliefs[:, 1], gt_beliefs[:, 2]
        x_pred, y_pred = pred_beliefs[:, 1], pred_beliefs[:, 2]
    else:
        # Fallback if fewer dimensions
        x_gt = gt_beliefs[:, 0] if gt_beliefs.shape[1] >= 1 else np.zeros(len(gt_beliefs))
        y_gt = gt_beliefs[:, 1] if gt_beliefs.shape[1] >= 2 else np.zeros(len(gt_beliefs))
        x_pred = pred_beliefs[:, 0] if pred_beliefs.shape[1] >= 1 else np.zeros(len(pred_beliefs))
        y_pred = pred_beliefs[:, 1] if pred_beliefs.shape[1] >= 2 else np.zeros(len(pred_beliefs))

    # RGB coloring based on coordinates
    def normalize_dim(data_dim):
        min_val, max_val = np.nanmin(data_dim), np.nanmax(data_dim)
        if max_val > min_val:
            norm = (data_dim - min_val) / (max_val - min_val)
        else:
            norm = np.ones_like(data_dim) * 0.5
        return np.nan_to_num(norm, nan=0.5)

    # Use x, y coordinates and distance for RGB
    R = normalize_dim(x_gt)
    G = normalize_dim(y_gt)
    B = normalize_dim(np.sqrt(x_gt**2 + y_gt**2))
    alpha = transform_for_alpha(weights, min_alpha=0.15)  # Bloch uses 0.15 min_alpha
    colors = np.stack([R, G, B, alpha], axis=-1)

    # Plot ground truth (smaller points for Bloch)
    axes[0].scatter(x_gt, y_gt, color=colors, s=0.15, rasterized=True, marker='.')
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_axis_off()

    # Plot predictions
    axes[1].scatter(x_pred, y_pred, color=colors, s=0.05, rasterized=True, marker='.')
    axes[1].set_title('Model Prediction', fontsize=14)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_axis_off()

    # Plot metrics
    ax = axes[2]
    metric_names = ['RMSE', 'MAE', 'R²']
    metric_values = [metrics['rmse'], metrics['mae'], metrics['r2']]

    ax.axis('off')

    # Display metrics as text
    y_pos = 0.8
    ax.text(0.1, y_pos, 'Metrics:', fontsize=12, weight='bold', transform=ax.transAxes)
    y_pos -= 0.15

    for name, value in zip(metric_names, metric_values):
        ax.text(0.1, y_pos, f'{name}:', fontsize=10, transform=ax.transAxes)
        ax.text(0.5, y_pos, f'{value:.4f}', fontsize=10, family='monospace',
               transform=ax.transAxes)
        y_pos -= 0.12

    # Add row label
    axes[0].text(-0.15, 0.5, 'Bloch', rotation=90, fontsize=14, ha='right',
                va='center', transform=axes[0].transAxes)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")


def create_random_model(config: dict, device: str = 'cpu', seed: Optional[int] = None):
    """Create a randomly initialized transformer model.

    Args:
        config: Configuration dictionary with model_config
        device: Device to create model on
        seed: Random seed for initialization (None for random)

    Returns:
        Randomly initialized HookedTransformer
    """
    model_config = config.get('model_config', config).copy()

    # Prepare config
    if 'dtype' in model_config:
        if isinstance(model_config['dtype'], str):
            model_config['dtype'] = getattr(torch, model_config['dtype'].split('.')[-1])
    else:
        model_config['dtype'] = torch.float32

    model_config['device'] = device

    if 'd_vocab' not in model_config:
        model_config['d_vocab'] = 3

    # Set seed if provided
    if seed is not None:
        model_config['seed'] = seed
        print(f"Using seed: {seed}")

    # Create model with random initialization
    cfg = HookedTransformerConfig(**model_config)
    model = HookedTransformer(cfg)
    model.eval()

    seed_info = f" (seed: {seed})" if seed is not None else " (random seed)"
    print(f"Created random transformer: {model_config.get('n_layers', '?')} layers, "
          f"{model_config.get('d_model', '?')} dims, "
          f"{model_config.get('n_heads', '?')} heads{seed_info}")

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Bloch process visualization from W&B artifact or random init")
    parser.add_argument("--artifact", type=str,
                       help="W&B artifact path (e.g., 'entity/project/artifact:version')")
    parser.add_argument("--random-init", action="store_true",
                       help="Use randomly initialized model instead of loading from W&B")
    parser.add_argument("--config", type=str,
                       help="Path to config YAML file (required for --random-init)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for model initialization (only with --random-init)")
    parser.add_argument("--output", type=str, default="Figs/Fig2_artifact_bloch.png",
                       help="Output path for figure")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for computation (cpu/cuda)")
    parser.add_argument("--rcond", type=float, default=1e-10,
                       help="Regularization parameter for regression")
    parser.add_argument("--cache-dir", type=str, default="./wandb_artifacts",
                       help="Directory to cache downloaded artifacts")

    args = parser.parse_args()

    # Validate arguments
    if not args.random_init and not args.artifact:
        parser.error("Either --artifact or --random-init must be specified")

    if args.random_init and not args.config and not args.artifact:
        parser.error("--config is required when using --random-init without --artifact")

    cache_dir = Path(args.cache_dir)
    device = args.device

    print("=" * 80)
    if args.random_init:
        print("Bloch Process Belief Visualization with Random Initialization")
    else:
        print("Bloch Process Belief Visualization from W&B Artifact")
    print("=" * 80)

    # Step 1 & 2: Get config and model
    if args.random_init:
        # Load config from file or artifact
        if args.config:
            print("\n[1/5] Loading configuration from file...")
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        elif args.artifact:
            print("\n[1/5] Downloading config from artifact...")
            _, config = download_artifact(args.artifact, cache_dir)

        print("\n[2/5] Creating randomly initialized model...")
        model = create_random_model(config, device, seed=args.seed)
    else:
        # Original path: download and load from artifact
        print("\n[1/5] Downloading artifact and configuration...")
        checkpoint_path, config = download_artifact(args.artifact, cache_dir)

        print("\n[2/5] Loading model...")
        model = load_transformer_from_checkpoint(checkpoint_path, config, device)

    # Step 3: Generate ground truth data
    print("\n[3/5] Generating ground truth data...")
    nn_inputs, gt_beliefs, indices, probs = generate_bloch_data(config, device)

    # Use ALL timesteps for regression
    print("\n[4/5] Running regression analysis...")
    import time

    print("  Extracting activations...")
    start = time.time()
    activations = extract_transformer_activations(model, nn_inputs)
    print(f"  Activation extraction took {time.time() - start:.1f}s")
    print(f"  Full activations shape: {activations.shape}")
    print(f"  Ground truth beliefs shape: {gt_beliefs.shape}")
    print(f"  Weights shape: {probs.shape}")

    print("  Performing weighted ridge regression (10-fold CV)...")
    start = time.time()
    results = run_regression(
        activations,
        gt_beliefs,
        probs,
        rcond=args.rcond
    )
    print(f"  Regression took {time.time() - start:.1f}s")

    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE: {results['mae']:.6f}")
    print(f"  R²: {results['r2']:.6f}")

    # Step 5: Generate visualization (use last timestep for plotting)
    print("\n[5/5] Generating visualization...")
    start = time.time()
    gt_beliefs_last = gt_beliefs[:, -1, :].cpu().numpy()
    pred_beliefs_last = results['predicted_beliefs'][:, -1, :]
    weights_last = probs[:, -1].cpu().numpy()

    visualize_beliefs(
        gt_beliefs_last,
        pred_beliefs_last,
        weights_last,
        results,
        args.output
    )
    print(f"  Visualization took {time.time() - start:.1f}s")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
