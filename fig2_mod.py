"""
Modified Fig2.py - Loads models from wandb run IDs and generates mess3 visualizations.

This script:
1. Downloads model checkpoints from wandb runs (latest checkpoint by default)
2. Runs regression analysis to map activations to belief states
3. Generates visualization for mess3 process only

Usage:
    # Basic usage with wandb run paths
    uv run python fig2_mod.py \\
        --transformer-run "entity/project/run_id_1" \\
        --lstm-run "entity/project/run_id_2"

    # With custom output path and device
    uv run python fig2_mod.py \\
        --transformer-run "your-entity/quantum-reps/abc123" \\
        --lstm-run "your-entity/quantum-reps/def456" \\
        --output "results/my_mess3_viz.png" \\
        --device "cuda" \\
        --rcond 1e-8

    # The run path format is: "entity/project/run_id"
    # You can find this from your W&B dashboard URL:
    # https://wandb.ai/entity/project/runs/run_id

Note:
    - Both runs should be trained on the mess3 process
    - The script downloads the latest checkpoint from each run
    - Requires WANDB_API_KEY environment variable to be set
    - First run generates ground truth data (ensure process configs match)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import torch
import fnmatch
from pathlib import Path
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from matplotlib.patches import Patch
from tqdm.auto import tqdm

# Import activation analysis utilities
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from epsilon_transformers.training.networks import create_RNN
from transformer_lens import HookedTransformer

try:
    import wandb
except ImportError:
    wandb = None


def download_checkpoint_from_wandb(
    run_path: str,
    destination: Path,
    file_pattern: str = "*.pt",
    index: int = -1,
) -> Tuple[Path, str]:
    """Download a checkpoint from a W&B run.

    Args:
        run_path: W&B run path in format "entity/project/run_id" or "project/run_id"
        destination: Local directory to download to
        file_pattern: Pattern to match checkpoint files (default: "*.pt")
        index: Index of checkpoint to download (-1 for latest, 0 for first)

    Returns:
        Tuple of (local_path, selected_name)
    """
    if wandb is None:
        raise RuntimeError("wandb is not installed; cannot download checkpoints.")

    api = wandb.Api()
    run = api.run(run_path)

    # Get all checkpoint files matching the pattern
    names = sorted({file_obj.name for file_obj in run.files()})
    matches = [name for name in names if fnmatch.fnmatch(name, file_pattern)]

    if not matches:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in run {run_path}")

    # Select checkpoint by index
    if index < 0:
        index = len(matches) + index
    selected_name = matches[index]

    print(f"Downloading checkpoint {selected_name} from {run_path}...")

    # Download the file
    file_ref = run.file(selected_name)
    destination.mkdir(parents=True, exist_ok=True)
    local_path = Path(file_ref.download(root=str(destination), replace=True))

    if local_path.is_dir():
        local_path = local_path / selected_name

    if not local_path.exists():
        raise FileNotFoundError(f"Downloaded file not found at {local_path}")

    print(f"Downloaded to {local_path}")
    return local_path, selected_name


def get_wandb_run_config(run_path: str) -> dict:
    """Get the run configuration from wandb.

    Args:
        run_path: W&B run path

    Returns:
        Run configuration dictionary
    """
    if wandb is None:
        raise RuntimeError("wandb is not installed")

    api = wandb.Api()
    run = api.run(run_path)
    return dict(run.config)


def load_model_from_checkpoint(checkpoint_path: Path, run_config: dict, device: str = 'cpu'):
    """Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        run_config: Run configuration dictionary
        device: Device to load model on

    Returns:
        Loaded model and model_type ('transformer' or 'rnn')
    """
    # Determine model type from config
    model_type = run_config.get('model_type', 'transformer').lower()

    if 'transformer' in model_type:
        # Load transformer model
        from transformer_lens import HookedTransformer, HookedTransformerConfig

        # Get model config and prepare it
        model_config = run_config.get('model_config', run_config)

        # Convert dtype string to torch dtype if needed
        if 'dtype' in model_config and isinstance(model_config['dtype'], str):
            model_config['dtype'] = getattr(torch, model_config['dtype'].split('.')[-1])

        # Set device
        model_config['device'] = device

        # Create model config object
        config = HookedTransformerConfig(**model_config)

        # Create model
        model = HookedTransformer(config)

        # Load state dict
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions
            state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        return model, 'transformer'
    else:
        # Load RNN model (LSTM, GRU, or RNN)
        # Infer vocab size from checkpoint
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
        vocab_size = state_dict['output_layer.weight'].size(0)

        model = create_RNN(run_config, vocab_size, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, 'rnn'


def extract_activations(model, nn_inputs, model_type: str, device: str = 'cpu'):
    """Extract activations from a model.

    Args:
        model: The neural network model
        nn_inputs: Input sequences (torch tensor)
        model_type: 'transformer' or 'rnn'
        device: Device for computation

    Returns:
        Activations dictionary
    """
    nn_inputs = nn_inputs.to(device)

    if model_type == 'transformer':
        # Extract residual stream activations
        activation_keys = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
        activation_keys.append('ln_final.hook_normalized')

        _, acts = model.run_with_cache(nn_inputs, names_filter=lambda x: x in activation_keys)
        return acts
    else:
        # Extract RNN activations
        _, state_dict = model.forward_with_all_states(nn_inputs)
        layer_states = state_dict['layer_states']

        # Create activation dictionary
        acts_dict = {f"layer{i}": layer_states[i] for i in range(layer_states.shape[0])}

        # Add one-hot encoded inputs
        vocab_size = model.output_layer.out_features
        one_hot_inputs = torch.zeros(nn_inputs.shape[0], nn_inputs.shape[1], vocab_size, device=device)
        indices = nn_inputs.long().unsqueeze(-1)
        one_hot_inputs.scatter_(2, indices, 1)
        acts_dict["input"] = one_hot_inputs

        return acts_dict


def combine_activations(acts, model_type: str):
    """Combine activations from all layers into a single tensor.

    Args:
        acts: Activations dictionary
        model_type: 'transformer' or 'rnn'

    Returns:
        Combined activations tensor of shape (batch_size, seq_len, total_dim)
    """
    if model_type == 'transformer':
        # Stack all residual stream activations
        act_list = [acts[key] for key in sorted(acts.keys())]
        combined = torch.cat(act_list, dim=-1)
    else:
        # Stack RNN layer activations
        act_list = [acts[key] for key in sorted(acts.keys())]
        combined = torch.cat(act_list, dim=-1)

    return combined


def run_regression_analysis(
    activations: torch.Tensor,
    beliefs: torch.Tensor,
    weights: torch.Tensor,
    rcond: float = 1e-10,
    n_splits: int = 10,
):
    """Run weighted least squares regression to map activations to beliefs.

    Args:
        activations: Neural network activations (batch, seq_len, act_dim)
        beliefs: Ground truth beliefs (batch, seq_len, belief_dim)
        weights: Sample weights based on sequence probabilities (batch,)
        rcond: Regularization parameter for ridge regression
        n_splits: Number of cross-validation splits

    Returns:
        Dictionary with predictions and metrics
    """
    # Flatten sequence dimension
    batch_size, seq_len, act_dim = activations.shape
    _, _, belief_dim = beliefs.shape

    X = activations.reshape(-1, act_dim).cpu().numpy()
    y = beliefs.reshape(-1, belief_dim).cpu().numpy()

    # Expand weights to match flattened shape
    sample_weights = weights.cpu().numpy()
    sample_weights = np.repeat(sample_weights, seq_len)

    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum()

    # K-fold cross validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    predictions = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = sample_weights[train_idx]

        # Normalize training weights
        w_train = w_train / w_train.sum()

        # Fit weighted ridge regression
        alpha = rcond * np.trace(X_train.T @ np.diag(w_train) @ X_train)
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(X_train, y_train, sample_weight=w_train)

        # Predict on test set
        predictions[test_idx] = reg.predict(X_test)

    # Compute metrics
    residuals = y - predictions
    rmse = np.sqrt(np.mean((residuals ** 2).sum(axis=1)))
    mae = np.mean(np.abs(residuals).sum(axis=1))

    # R² score (weighted)
    ss_res = np.sum(sample_weights * (residuals ** 2).sum(axis=1))
    y_mean = np.average(y, axis=0, weights=sample_weights)
    ss_tot = np.sum(sample_weights * ((y - y_mean) ** 2).sum(axis=1))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Reshape predictions back
    predictions = predictions.reshape(batch_size, seq_len, belief_dim)

    return {
        'predicted_beliefs': predictions,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def generate_mess3_data(run_config: dict, device: str = 'cpu'):
    """Generate mess3 ground truth data (beliefs, inputs, weights).

    Args:
        run_config: Run configuration dictionary
        device: Device for tensors

    Returns:
        Tuple of (nn_inputs, beliefs, indices, probs)
    """
    # Generate MSP data for mess3
    nn_inputs, beliefs, indices, probs, _ = prepare_msp_data(
        run_config, run_config['model_config']
    )

    # Move to device
    nn_inputs = nn_inputs.to(device)
    beliefs = beliefs.to(device)
    probs = probs.to(device)

    return nn_inputs, beliefs, indices, probs


def transform_for_alpha(weights, min_alpha=0.1, transformation='cbrt'):
    """Transform weights to alpha values for visualization."""
    weights = np.asarray(weights)
    weights = np.clip(weights, 0, 1)

    if transformation == 'log':
        alpha = np.log1p(weights * 100) / np.log1p(100)
    elif transformation == 'sqrt':
        alpha = np.sqrt(weights)
    elif transformation == 'cbrt':
        alpha = np.cbrt(weights)
    else:  # linear
        alpha = weights

    alpha = alpha * (1 - min_alpha) + min_alpha
    return alpha


def project_to_simplex(data):
    """Project 3D data to 2D simplex coordinates."""
    x_temp = data[:, 0] - data[:, 1] / 2 - data[:, 2] / 2
    y_temp = np.sqrt(3) / 2 * (data[:, 1] - data[:, 2])
    # Rotate 90 degrees counterclockwise
    x = -y_temp
    y = x_temp
    return x, y


def visualize_mess3_beliefs(
    gt_beliefs: np.ndarray,
    pred_beliefs_transformer: np.ndarray,
    pred_beliefs_lstm: np.ndarray,
    weights: np.ndarray,
    metrics_transformer: dict,
    metrics_lstm: dict,
    output_path: str = "Figs/Fig2_mod.png",
):
    """Generate visualization for mess3 beliefs.

    Args:
        gt_beliefs: Ground truth beliefs
        pred_beliefs_transformer: Transformer predictions
        pred_beliefs_lstm: LSTM predictions
        weights: Sequence weights
        metrics_transformer: Transformer metrics dict
        metrics_lstm: LSTM metrics dict
        output_path: Output file path
    """
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5),
                            gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})

    # Plotting parameters for mess3
    point_size_truth = 1.0
    point_size_pred = 1.0
    min_alpha = 0.2

    # Calculate colors using RGB scheme
    def normalize_dim_color(data_dim):
        min_val, max_val = np.nanmin(data_dim), np.nanmax(data_dim)
        if max_val > min_val:
            norm = (data_dim - min_val) / (max_val - min_val)
        else:
            norm = np.ones_like(data_dim) * 0.5
        return np.nan_to_num(norm, nan=0.5)

    # Project to simplex
    x_gt, y_gt = project_to_simplex(gt_beliefs[:, :3])
    x_trans, y_trans = project_to_simplex(pred_beliefs_transformer[:, :3])
    x_lstm, y_lstm = project_to_simplex(pred_beliefs_lstm[:, :3])

    # RGB coloring based on belief coordinates
    R = normalize_dim_color(gt_beliefs[:, 0])
    G = normalize_dim_color(gt_beliefs[:, 1])
    B = normalize_dim_color(gt_beliefs[:, 2])

    alpha_values = transform_for_alpha(weights, min_alpha=min_alpha, transformation='cbrt')
    colors_rgba = np.stack([R, G, B, alpha_values], axis=-1)

    # Plot ground truth
    axes[0].scatter(x_gt, y_gt, color=colors_rgba, s=point_size_truth, rasterized=True, marker='.')
    axes[0].set_title('Ground Truth', fontsize=15)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_axis_off()

    # Plot transformer predictions
    axes[1].scatter(x_trans, y_trans, color=colors_rgba, s=point_size_pred, rasterized=True, marker='.')
    axes[1].set_title('Transformer', fontsize=15)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_axis_off()

    # Plot LSTM predictions
    axes[2].scatter(x_lstm, y_lstm, color=colors_rgba, s=point_size_pred, rasterized=True, marker='.')
    axes[2].set_title('LSTM', fontsize=15)
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].set_axis_off()

    # Plot RMSE bar chart
    ax_bar = axes[3]
    model_types = ['Transformer', 'LSTM']

    colors = {
        'classical': '#2563eb',
        'random': '#6b7280'
    }

    classical_rmses = [metrics_transformer.get('rmse', 0), metrics_lstm.get('rmse', 0)]
    random_rmses = [metrics_transformer.get('random_rmse', 0), metrics_lstm.get('random_rmse', 0)]

    x = np.arange(len(model_types))
    width = 0.4

    ax_bar.bar(x - width/2, classical_rmses, width, color=colors['classical'], alpha=0.85, label='Classical')
    ax_bar.bar(x + width/2, random_rmses, width, color=colors['random'], alpha=0.85, label='Random')

    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.spines['bottom'].set_linewidth(0.8)
    ax_bar.spines['bottom'].set_color('#666666')

    ax_bar.set_ylabel('RMSE', fontsize=9, color='#333333')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(model_types, fontsize=8, color='#333333')
    ax_bar.tick_params(axis='x', length=0, width=0, pad=8)
    ax_bar.tick_params(axis='y', labelsize=7, length=3, width=0.5, color='#666666')
    ax_bar.legend(loc='upper right', frameon=False, fontsize=7, handlelength=1.0, handletextpad=0.4)
    ax_bar.grid(True, axis='y', alpha=0.15, linestyle='-', linewidth=0.5, color='#cccccc')
    ax_bar.set_axisbelow(True)
    ax_bar.set_title('Model Performance', fontsize=15)

    # Add row label
    axes[0].text(-0.1, 0.5, 'Mess3', rotation=90, fontsize=15, ha='right', va='center',
                transform=axes[0].transAxes)

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate mess3 belief visualization from wandb runs")
    parser.add_argument("--transformer-run", type=str, required=True,
                       help="W&B run path for transformer (e.g., 'entity/project/run_id')")
    parser.add_argument("--lstm-run", type=str, required=True,
                       help="W&B run path for LSTM")
    parser.add_argument("--output", type=str, default="Figs/Fig2_mod.png",
                       help="Output path for figure")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for computation (cpu/cuda)")
    parser.add_argument("--rcond", type=float, default=1e-10,
                       help="Regularization parameter for regression")
    parser.add_argument("--cache-dir", type=str, default="./wandb_cache",
                       help="Directory to cache downloaded checkpoints")

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    device = args.device

    print("=" * 80)
    print("Modified Fig2 - Mess3 Belief Visualization from W&B runs")
    print("=" * 80)

    # Step 1: Download and load transformer model
    print("\n[1/6] Loading Transformer model...")
    trans_ckpt_path, _ = download_checkpoint_from_wandb(
        args.transformer_run, cache_dir / "transformer", index=-1
    )
    trans_config = get_wandb_run_config(args.transformer_run)
    trans_model, trans_type = load_model_from_checkpoint(trans_ckpt_path, trans_config, device)

    # Step 2: Download and load LSTM model
    print("\n[2/6] Loading LSTM model...")
    lstm_ckpt_path, _ = download_checkpoint_from_wandb(
        args.lstm_run, cache_dir / "lstm", index=-1
    )
    lstm_config = get_wandb_run_config(args.lstm_run)
    lstm_model, lstm_type = load_model_from_checkpoint(lstm_ckpt_path, lstm_config, device)

    # Step 3: Generate ground truth data
    print("\n[3/6] Generating mess3 ground truth data...")
    nn_inputs, gt_beliefs, indices, probs = generate_mess3_data(trans_config, device)

    # Flatten for regression (take last timestep beliefs)
    gt_beliefs_flat = gt_beliefs[:, -1, :].cpu().numpy()  # (batch, belief_dim)
    weights_flat = probs.cpu().numpy()  # (batch,)

    print(f"  Generated {len(gt_beliefs_flat)} sequences")
    print(f"  Belief dimensions: {gt_beliefs_flat.shape[1]}")

    # Step 4: Extract activations and run regression for transformer
    print("\n[4/6] Running regression analysis for Transformer...")
    trans_acts = extract_activations(trans_model, nn_inputs, trans_type, device)
    trans_combined = combine_activations(trans_acts, trans_type)

    # Take last timestep activations
    trans_acts_flat = trans_combined[:, -1, :]  # (batch, act_dim)

    # Run regression
    trans_results = run_regression_analysis(
        trans_acts_flat.unsqueeze(1),  # Add seq_len dim
        torch.from_numpy(gt_beliefs_flat).unsqueeze(1).float(),
        torch.from_numpy(weights_flat).float(),
        rcond=args.rcond
    )

    print(f"  Transformer RMSE: {trans_results['rmse']:.6f}")
    print(f"  Transformer R²: {trans_results['r2']:.6f}")

    # Step 5: Extract activations and run regression for LSTM
    print("\n[5/6] Running regression analysis for LSTM...")
    lstm_acts = extract_activations(lstm_model, nn_inputs, lstm_type, device)
    lstm_combined = combine_activations(lstm_acts, lstm_type)
    lstm_acts_flat = lstm_combined[:, -1, :]

    lstm_results = run_regression_analysis(
        lstm_acts_flat.unsqueeze(1),
        torch.from_numpy(gt_beliefs_flat).unsqueeze(1).float(),
        torch.from_numpy(weights_flat).float(),
        rcond=args.rcond
    )

    print(f"  LSTM RMSE: {lstm_results['rmse']:.6f}")
    print(f"  LSTM R²: {lstm_results['r2']:.6f}")

    # Step 6: Generate visualization
    print("\n[6/6] Generating visualization...")

    # For now, use placeholders for random baseline
    metrics_trans = {
        'rmse': trans_results['rmse'],
        'r2': trans_results['r2'],
        'random_rmse': 0.05  # Placeholder
    }

    metrics_lstm = {
        'rmse': lstm_results['rmse'],
        'r2': lstm_results['r2'],
        'random_rmse': 0.05  # Placeholder
    }

    visualize_mess3_beliefs(
        gt_beliefs_flat,
        trans_results['predicted_beliefs'][:, 0, :],  # Remove seq_len dim
        lstm_results['predicted_beliefs'][:, 0, :],
        weights_flat,
        metrics_trans,
        metrics_lstm,
        output_path=args.output
    )

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
