import argparse
import collections
import glob
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from minimal_impl.regression import (
    _combine_layer_activations,
    compute_kfold_split,
    deduplicate_data,
    deduplicate_tensor,
)
from scripts.activation_analysis.config import (
    RCOND_SWEEP_LIST,
    TRANSFORMER_ACTIVATION_KEYS,
)
from scripts.activation_analysis.data_loading import ActivationExtractor
from scripts.activation_analysis.regression import (
    RegressionAnalyzer,
    run_activation_to_beliefs_regression_kf,
)
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Define nested_dict_factory for joblib loading (important!)
nested_dict_factory = collections.defaultdict


def _instantiate_model(run_cfg: dict, checkpoint_path: Path, device: torch.device) -> HookedTransformer:
    model_cfg = dict(run_cfg["model_config"])
    model_cfg.setdefault("device", str(device))

    if "d_vocab" not in model_cfg:
        process_cfg = run_cfg.get("process_config", {})
        name = process_cfg.get("name") or process_cfg.get("mode")
        if name in {"tom_quantum", "bloch", "bloch_manual"}:
            vocab_size = 4
        elif name in {"mess3", "mm3"}:
            vocab_size = 3
        else:
            vocab_size = int(process_cfg.get("d_vocab", 5))
        model_cfg["d_vocab"] = vocab_size

    dtype = model_cfg.get("dtype", "float32")
    if isinstance(dtype, str):
        model_cfg["dtype"] = getattr(torch, dtype)

    hook_cfg = HookedTransformerConfig(**model_cfg)
    model = HookedTransformer(hook_cfg)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _download_wandb_checkpoint(run_path: str, alias: str = "latest") -> Path:
    api = wandb.Api()
    run = api.run(run_path)
    target_artifact = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model":
            continue
        if alias in artifact.aliases or artifact.name.endswith(alias):
            target_artifact = artifact
            break
    if target_artifact is None:
        model_artifacts = [art for art in run.logged_artifacts() if art.type == "model"]
        if not model_artifacts:
            raise RuntimeError(f"No model artifacts found for run {run_path}")
        target_artifact = model_artifacts[-1]

    download_dir = Path(target_artifact.download(root=tempfile.mkdtemp(prefix="fig2_ckpt_")))
    ckpt_files = list(download_dir.glob("*.pt"))
    if not ckpt_files:
        raise RuntimeError(f"No checkpoint files found in artifact {target_artifact.name}")
    return ckpt_files[0]


def load_ground_truth(run_dir: str, filename: str = 'ground_truth_data.joblib') -> dict:
    """Loads the ground truth data for a specific run, allowing custom filename."""
    ground_truth_filepath = os.path.join(run_dir, filename)
    if not os.path.exists(ground_truth_filepath): 
        print(f"Error: Ground truth file not found at {ground_truth_filepath}")
        return None
    try:
        ground_truth_data = joblib.load(ground_truth_filepath)
        print(f"Loaded ground truth data from {ground_truth_filepath}")
        if not all(k in ground_truth_data for k in ['beliefs', 'probs']): 
            print(f"Warning: GT data missing keys.")
        if 'beliefs' in ground_truth_data: 
            ground_truth_data['beliefs'] = np.asarray(ground_truth_data['beliefs'])
        if 'probs' in ground_truth_data: 
            ground_truth_data['probs'] = np.asarray(ground_truth_data['probs'])
        return ground_truth_data
    except AttributeError as e:
         if 'nested_dict_factory' in str(e): 
             print(f"Error loading {ground_truth_filepath}: Still missing 'nested_dict_factory'. Ensure definition is correct.")
         else: 
             print(f"AttributeError loading {ground_truth_filepath}: {e}")
         return None
    except Exception as e: 
        print(f"Error loading ground truth file {ground_truth_filepath}: {e}")
        return None


def load_predictions(run_dir: str, is_markov3_run: bool, target_ckpt_str: str) -> dict:
    """Loads the predictions data for a specific checkpoint file."""
    if target_ckpt_str is None: 
        print("Error: target_ckpt_str cannot be None for loading predictions.")
        return None
    filename = f"markov3_checkpoint_{target_ckpt_str}.joblib" if is_markov3_run else f"checkpoint_{target_ckpt_str}.joblib"
    predictions_filepath = os.path.join(run_dir, filename)
    if not os.path.exists(predictions_filepath): 
        print(f"Error: Checkpoint predictions file not found at {predictions_filepath}")
        return None
    try:
        single_ckpt_data = joblib.load(predictions_filepath)
        if isinstance(single_ckpt_data, dict):
            for layer, data in single_ckpt_data.items():
                if isinstance(data, dict) and 'predicted_beliefs' in data: 
                    data['predicted_beliefs'] = np.asarray(data['predicted_beliefs'])
        return {target_ckpt_str: single_ckpt_data}  # Wrap in dict keyed by checkpoint string
    except AttributeError as e:
         if 'nested_dict_factory' in str(e): 
             print(f"Error loading {predictions_filepath}: Missing 'nested_dict_factory'.")
         else: 
             print(f"AttributeError loading {predictions_filepath}: {e}")
         return None
    except Exception as e: 
        print(f"Error loading checkpoint predictions file {predictions_filepath}: {e}")
        return None


def load_random_baseline_rmse(run_dir: str, target_layer: str, is_markov3_geometry: bool = False) -> float:
    """Load RMSE from checkpoint_0.joblib (randomly initialized network)."""
    try:
        checkpoint_file = 'markov3_checkpoint_0.joblib' if is_markov3_geometry else 'checkpoint_0.joblib'
        checkpoint_path = os.path.join(run_dir, checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            return np.nan
            
        checkpoint_data = joblib.load(checkpoint_path)
        
        if target_layer not in checkpoint_data:
            return np.nan
            
        layer_data = checkpoint_data[target_layer]
        
        if 'rmse' not in layer_data:
            return np.nan
            
        rmse_value = layer_data['rmse']
        
        # Extract scalar value if it's an array
        if hasattr(rmse_value, '__len__') and len(rmse_value) > 0:
            if hasattr(rmse_value, 'mean'):
                return float(rmse_value.mean())
            else:
                return float(rmse_value[0])
        else:
            return float(rmse_value)
        
    except Exception as e:
        return np.nan


def _get_plotting_params(experiment_name: str) -> dict:
    """Returns a dictionary of plotting parameters based on experiment name."""
    name_lower = experiment_name.lower()
    print(name_lower)
    params = { 
        'point_size': {'truth': 1.0, 'pred': 0.5}, 
        'min_alpha': 0.02, 
        'transformation': 'cbrt', 
        'use_pca': False, 
        'project_to_simplex': False, 
        'inds_to_plot': [0, 1], 
        'com': False 
    }
    if 'markov3' in name_lower:
        params.update({ 
            'point_size': {'truth': 10.5, 'pred': 5.0}, 
            'min_alpha': 0.1, 
            'use_pca': True, 
            'project_to_simplex': False, 
            'inds_to_plot': [1,2] 
        })
    elif 'tomqa' in name_lower or 'tomqb' in name_lower or 'tom_quantum' in name_lower or 'bloch' in name_lower:
        params.update({ 
            'point_size': {'truth': 0.15, 'pred': 0.05}, 
            'min_alpha': 0.15, 
            'inds_to_plot': [1, 2] 
        })
    elif 'post_quantum' in name_lower or 'moon' in name_lower: 
        params.update({ 
            'point_size': {'truth': 20., 'pred': 10.}, 
            'min_alpha': 0.01, 
            'com': False, 
            'use_pca':False, 
            'inds_to_plot': [1, 2]
        })
    elif 'rrxor' in name_lower: 
        params.update({ 
            'point_size': {'truth': 30., 'pred': 1}, 
            'min_alpha': 0.1, 
            'com': True, 
            'use_pca': True, 
            'project_to_simplex': False, 
            'inds_to_plot': [1, 2], 
        })
    elif 'fanizza' in name_lower: 
        params.update({ 
            'point_size': {'truth': 40., 'pred': 1}, 
            'min_alpha': 0.1, 
            'com': True, 
            'use_pca': False, 
            'project_to_simplex': False, 
            'inds_to_plot': [2, 3], 
        })
    elif 'mess3' in name_lower:
        params.update({ 
            'point_size': {'truth': 1., 'pred': 1}, 
            'min_alpha': 0.2, 
            'com': False, 
            'use_pca': False, 
            'project_to_simplex': True, 
            'inds_to_plot': [0, 1], 
        })
    return params


def transform_for_alpha(weights, min_alpha=0.1, transformation='log'):
    """Transform weights to alpha values for visualization."""
    weights = np.asarray(weights)
    weights = np.clip(weights, 0, 1)
    
    if transformation == 'log':
        alpha = np.log1p(weights * 100) / np.log1p(100)
    elif transformation == 'sqrt':
        alpha = np.sqrt(weights)
    elif transformation == 'cbrt':
        alpha = np.cbrt(weights)
    elif transformation == 'linear':
        alpha = weights
    else:  # Default to linear
        alpha = weights
    
    alpha = alpha * (1 - min_alpha) + min_alpha
    return alpha


def _project_to_simplex(data):
    """Project 3D data to 2D simplex coordinates."""
    # Project to 2D equilateral triangle, rotated 90 degrees
    x_temp = data[:, 0] - data[:, 1] / 2 - data[:, 2] / 2
    y_temp = np.sqrt(3) / 2 * (data[:, 1] - data[:, 2])
    # Rotate 90 degrees counterclockwise: (x, y) -> (-y, x)
    x = -y_temp
    y = x_temp
    return x, y


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return weights
    w_min = weights.min()
    w_max = weights.max()
    if w_max - w_min < 1e-12:
        return np.zeros_like(weights)
    return (weights - w_min) / (w_max - w_min)


def plot_single_bloch(
    run_config_path: Path,
    wandb_run_path: str,
    checkpoint_alias: str,
    output_path: Path,
    layer: str = "combined",
    n_splits: int = 10,
    device: str = "cpu",
):
    run_cfg = yaml.safe_load(run_config_path.read_text())
    run_cfg = dict(run_cfg)
    run_cfg.setdefault("global_config", {})["device"] = device
    run_cfg.setdefault("train_config", {})
    run_cfg.setdefault("model_config", {})

    checkpoint_path = _download_wandb_checkpoint(wandb_run_path, alias=checkpoint_alias)
    true_beliefs, predicted_beliefs, weights, metrics = _run_single_regression(
        run_cfg,
        checkpoint_path,
        torch.device(device),
        layer=layer,
        n_splits=n_splits,
    )

    params = _get_plotting_params("tom_quantum_single")
    gt_weights_norm = _normalize_weights(weights)
    colors = plt.cm.viridis(gt_weights_norm if gt_weights_norm.size else np.zeros_like(weights))
    colors[:, -1] = transform_for_alpha(weights, params["min_alpha"], params["transformation"])

    x_true, y_true, pca = _calculate_plot_coords(
        true_beliefs,
        true_beliefs,
        params["use_pca"],
        params["project_to_simplex"],
        params["inds_to_plot"],
    )
    x_pred, y_pred, _ = _calculate_plot_coords(
        predicted_beliefs,
        true_beliefs,
        params["use_pca"],
        params["project_to_simplex"],
        params["inds_to_plot"],
        pca_instance=pca,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(x_true, y_true, color=colors, s=params["point_size"]["truth"], marker='.')
    axes[0].set_title("Ground Truth Beliefs")
    axes[0].set_axis_off()

    axes[1].scatter(x_pred, y_pred, color=colors, s=params["point_size"]["pred"], marker='.')
    axes[1].set_title(f"Predicted Beliefs ({layer})")
    axes[1].set_axis_off()

    rmse_value = metrics.get("rmse")
    if isinstance(rmse_value, np.ndarray):
        rmse_value = float(np.mean(rmse_value))
    elif isinstance(rmse_value, (list, tuple)):
        rmse_value = float(np.mean(rmse_value))
    axes[1].text(
        0.02,
        0.95,
        f"RMSE: {rmse_value:.4f}" if rmse_value is not None else "RMSE: N/A",
        transform=axes[1].transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
    )

    fig.suptitle("Bloch Belief Geometry vs Combined-Layer Regression", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def _prepare_ground_truth(run_cfg: dict):
    from epsilon_transformers.analysis.activation_analysis import prepare_msp_data

    n_ctx = run_cfg.get("model_config", {}).get("n_ctx") or run_cfg.get("n_ctx")
    if n_ctx is None:
        raise ValueError("n_ctx must be specified in model_config or top-level config.")

    run_cfg = dict(run_cfg)
    run_cfg.setdefault("global_config", {}).setdefault("device", "cpu")
    run_cfg.setdefault("train_config", {})
    run_cfg["train_config"].setdefault("bos", False)
    run_cfg["n_ctx"] = n_ctx

    nn_inputs, nn_beliefs, _, nn_probs, _ = prepare_msp_data(run_cfg, run_cfg["model_config"])
    dedup_probs, dedup_beliefs, dedup_indices, prefix_to_indices = deduplicate_data(
        nn_inputs, nn_probs, nn_beliefs
    )
    return {
        "nn_inputs": nn_inputs,
        "dedup_probs": dedup_probs,
        "dedup_beliefs": dedup_beliefs,
        "prefix_to_indices": prefix_to_indices,
    }


def _run_single_regression(
    run_cfg: dict,
    checkpoint_path: Path,
    device: torch.device,
    layer: str = "combined",
    n_splits: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    gt = _prepare_ground_truth(run_cfg)
    dedup_probs = gt["dedup_probs"].to(device)
    dedup_beliefs = gt["dedup_beliefs"].to(device)
    prefix_to_indices = gt["prefix_to_indices"]

    kfold_indices = compute_kfold_split(dedup_probs, n_splits=n_splits)

    model = _instantiate_model(run_cfg, checkpoint_path, device)
    extractor = ActivationExtractor(device=str(device))
    cache = extractor.extract_activations(
        model,
        gt["nn_inputs"],
        "transformer",
        relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
    )
    nn_acts = {layer_name: acts for layer_name, acts in cache.items()}
    nn_acts["combined"] = _combine_layer_activations(nn_acts)
    if layer not in nn_acts:
        raise ValueError(f"Layer '{layer}' not found. Available: {list(nn_acts.keys())}")

    dedup_acts, _ = deduplicate_tensor(prefix_to_indices, nn_acts[layer])
    analyzer = RegressionAnalyzer(device=str(device), use_efficient_pinv=True)
    results = run_activation_to_beliefs_regression_kf(
        analyzer,
        dedup_acts.to(device),
        dedup_beliefs,
        dedup_probs,
        kfold_indices,
        rcond_values=RCOND_SWEEP_LIST,
    )
    final_metrics = results.get("final_metrics", {})
    preds = final_metrics.get("predictions")
    if preds is None:
        raise RuntimeError("Regression did not return predictions.")
    return (
        dedup_beliefs.cpu().numpy(),
        preds,
        dedup_probs.cpu().numpy(),
        {
            "rmse": final_metrics.get("rmse"),
            "r2": final_metrics.get("r2"),
            "best_rcond": results.get("best_overall_rcond"),
        },
    )


def _calculate_plot_coords(beliefs_to_plot, gt_beliefs_for_pca, use_pca, project_to_simplex, inds_to_plot, pca_instance=None):
    """Calculate 2D coordinates for plotting beliefs."""
    if beliefs_to_plot is None:
        return None, None, None
        
    if project_to_simplex and beliefs_to_plot.shape[1] >= 3:
        x, y = _project_to_simplex(beliefs_to_plot[:, :3])
        return x, y, None
    elif use_pca and beliefs_to_plot.shape[1] >= 2:
        if pca_instance is None:
            pca_instance = PCA(n_components=2)
            pca_instance.fit(gt_beliefs_for_pca)
        coords_2d = pca_instance.transform(beliefs_to_plot)
        return coords_2d[:, 0], coords_2d[:, 1], pca_instance
    else:
        # Direct indexing
        if beliefs_to_plot.shape[1] >= 2:
            return beliefs_to_plot[:, inds_to_plot[0]], beliefs_to_plot[:, inds_to_plot[1]], None
        else:
            return beliefs_to_plot[:, 0], np.zeros_like(beliefs_to_plot[:, 0]), None


def _plot_beliefs_on_ax(ax: plt.Axes, x_plot: np.ndarray, y_plot: np.ndarray, colors_rgba: np.ndarray, point_size: float, com_point=None):
    """Plots pre-calculated belief points onto a given matplotlib Axes object."""
    plotted_something = False
    if x_plot is not None and y_plot is not None and colors_rgba is not None and x_plot.size > 0 and y_plot.size > 0 and colors_rgba.size > 0:
        if colors_rgba.shape[0] == x_plot.shape[0]:
            ax.scatter(x_plot, y_plot, color=colors_rgba, s=point_size, rasterized=True, marker='.')
            plotted_something = True
        else: 
            print(f"Warning: Mismatch points ({x_plot.shape[0]}) vs colors ({colors_rgba.shape[0]}). Skipping scatter.")
    
    # Plot center of mass if provided
    if com_point is not None and len(com_point) >= 2:
        ax.scatter(com_point[0], com_point[1], color='red', s=point_size*5, marker='*', edgecolor='black', linewidth=0.5, zorder=10)
    
    ax.set_axis_off()


def plot_clean_rmse_bars(ax, metric_data, is_markov_gt_row, is_mess3_row=False):
    """Clean, professional RMSE bar chart with no duplicate legends."""
    model_types = ['Transformer', 'LSTM']
    
    # Clean color palette
    colors = {
        'correct': '#2563eb',      # Professional blue
        'incorrect': '#dc2626',    # Muted red  
        'random': '#6b7280'        # Neutral gray
    }
    
    # Helper function to extract scalar values
    def extract_scalar(value):
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, 'shape') and value.shape == ():
            return float(value.item())
        if hasattr(value, '__len__') and len(value) > 0:
            if hasattr(value, 'mean'):
                return float(value.mean())
            else:
                return float(value[0])
        return 0.0

    # Extract RMSE values
    correct_rmses = []
    incorrect_rmses = []
    random_rmses = []
    
    for m in model_types:
        model_metrics = metric_data.get(m, {})
        correct_rmses.append(extract_scalar(model_metrics.get('correct', 0)))
        incorrect_rmses.append(extract_scalar(model_metrics.get('incorrect', 0)))
        random_rmses.append(extract_scalar(model_metrics.get('random', 0)))

    # Determine bars and labels based on row type
    x = np.arange(len(model_types))
    
    if is_mess3_row:
        # Mess3: only Classical and Random
        bar_data = [correct_rmses, random_rmses]
        bar_colors = [colors['correct'], colors['random']]
        bar_labels = ['Classical', 'Random']
        width = 0.4
        positions = [x - width/2, x + width/2]
    else:
        # Quantum: all three bars
        bar_data = [correct_rmses, incorrect_rmses, random_rmses]
        bar_colors = [colors['correct'], colors['incorrect'], colors['random']]
        if is_markov_gt_row:
            bar_labels = ['Classical', 'Quantum', 'Random']
        else:
            bar_labels = ['Quantum', 'Classical', 'Random']
        width = 0.26
        positions = [x - width, x, x + width]

    # Clear the axis completely first
    ax.clear()
    
    # Plot bars WITHOUT labels first
    bars = []
    for i, (data, color) in enumerate(zip(bar_data, bar_colors)):
        bar = ax.bar(positions[i], data, width, color=color, alpha=0.85, edgecolor='none')
        bars.append(bar)

    # Clean styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#666666')
    
    # Axis formatting
    ax.set_ylabel('RMSE', fontsize=9, color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(model_types, fontsize=8, color='#333333')
    ax.tick_params(axis='x', length=0, width=0, pad=8)
    ax.tick_params(axis='y', labelsize=7, length=3, width=0.5, color='#666666')
    
    # Create legend manually with specific handles and labels
    legend_elements = []
    for i, label in enumerate(bar_labels):
        legend_elements.append(Patch(facecolor=bar_colors[i], alpha=0.85, label=label))
    
    # Add legend with NO title - be very explicit
    legend = ax.legend(handles=legend_elements, loc='upper right', frameon=False, 
                      fontsize=7, handlelength=1.0, handletextpad=0.4, title='')
    # Force remove any title that might exist
    legend.set_title('')
    if legend.get_title():
        legend.get_title().set_visible(False)
    
    # Subtle grid
    ax.grid(True, axis='y', alpha=0.15, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)


def visualize_belief_grid_with_metrics(
    plot_config: list,
    output_base_dir: str,
    plot_output_dir: str,
    target_checkpoint: str,
    target_layer: str,
    output_filename: str = "belief_grid_with_metrics.png",
):
    """Generates a grid plot comparing beliefs with an additional column for RMSE bar charts."""
    n_rows = len(plot_config)
    n_cols = 4  # Ground Truth, Transformer, LSTM, RMSE Bar Chart
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5), 
                           gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})
    if n_rows == 1: 
        axes = axes.reshape(1, n_cols)
    fig.subplots_adjust(hspace=0.05, wspace=0.3)

    all_plot_data = {}
    calculated_coords = {}

    # --- Data Loading and Parameter Setup Loop ---
    for row_idx, config in enumerate(plot_config):
        experiment_name = config['name']
        gt_sweep, gt_run_id_int = config['gt_run']
        gt_run_dir = os.path.join(output_base_dir, f"{gt_sweep}_{gt_run_id_int}")
        print(f"\n--- Loading Row {row_idx+1}: {experiment_name} (Run Dir: {gt_run_dir}) ---")
        
        # Determine if the GROUND TRUTH for this row is markov3
        is_markov3_ground_truth = 'markov3' in experiment_name.lower()
        gt_filename = 'markov3_ground_truth_data.joblib' if is_markov3_ground_truth else 'ground_truth_data.joblib'
        
        gt_data = load_ground_truth(gt_run_dir, filename=gt_filename)
        valid_gt = (gt_data is not None and 'beliefs' in gt_data and 'probs' in gt_data and 
                   gt_data['beliefs'] is not None and gt_data['probs'] is not None and 
                   gt_data['beliefs'].ndim == 2 and gt_data['probs'].ndim == 1 and 
                   gt_data['beliefs'].shape[0] == gt_data['probs'].shape[0])
        
        if not valid_gt:
            print(f"Skipping {experiment_name} row: Failed GT load or invalid shapes.")
            all_plot_data[row_idx] = {'valid': False}
            calculated_coords[row_idx] = {'valid': False}
            for col_idx_off in range(n_cols): 
                axes[row_idx, col_idx_off].set_axis_off()
            continue

        true_beliefs = gt_data['beliefs']
        weights = gt_data['probs']
        belief_dims = true_beliefs.shape[1]
        row_params = _get_plotting_params(experiment_name)
        
        # Validate parameters
        if not row_params['use_pca'] and not row_params['project_to_simplex']:
             if max(row_params['inds_to_plot']) >= belief_dims: 
                 row_params['inds_to_plot'] = [0, 1] if belief_dims >= 2 else [0, 0]
        elif row_params['project_to_simplex'] and belief_dims < 3: 
            row_params['project_to_simplex'] = False
            row_params['use_pca'] = (belief_dims >= 2)
        elif row_params['use_pca'] and belief_dims < 2: 
            row_params['use_pca'] = False
        if max(row_params['inds_to_plot']) >= belief_dims and not row_params['use_pca']: 
            row_params['inds_to_plot'] = [0, 1] if belief_dims >= 2 else [0, 0]
        
        all_plot_data[row_idx] = { 
            'valid': True, 'gt_beliefs': true_beliefs, 'weights': weights, 
            'params': row_params, 'model_data': {}, 'metric_data': {}
        }
        calculated_coords[row_idx] = {
            'valid': True, 'params': row_params, 'coords': {}, 
            'pca': None, 'name': experiment_name
        }
        
        # Load Predictions for both correct and incorrect geometries
        for col_idx, (model_type, (sweep, run_id_int_model)) in enumerate(config['models']):
             run_directory = os.path.join(output_base_dir, f"{sweep}_{run_id_int_model}")
             
             # Find checkpoint
             resolved_ckpt_str = None
             ckpt_pattern = f"checkpoint_*.joblib"
             ckpt_files = glob.glob(os.path.join(run_directory, ckpt_pattern))
             print(f"  Looking for checkpoints in {run_directory} for {model_type}")
             if ckpt_files:
                 available_indices = sorted([int(re.search(r'_(\d+)\.joblib$', f).group(1)) 
                                           for f in ckpt_files if re.search(r'_(\d+)\.joblib$', f)])
                 if available_indices:
                     if target_checkpoint == 'last': 
                         resolved_ckpt_str = str(available_indices[-1])
                     elif target_checkpoint.isdigit() and int(target_checkpoint) in available_indices: 
                         resolved_ckpt_str = target_checkpoint
                     print(f"  Resolved checkpoint: {resolved_ckpt_str}")
             
             if not resolved_ckpt_str:
                 print(f"  Warning: Could not resolve checkpoint for {model_type} in {run_directory}")
                 print(f"  Available files: {ckpt_files[:5] if ckpt_files else 'None'}")
                 all_plot_data[row_idx]['model_data'][model_type] = None
                 all_plot_data[row_idx]['metric_data'][model_type] = {'correct': np.nan, 'incorrect': np.nan}
                 continue

             # Load CORRECT geometry data
             pred_data_correct = load_predictions(run_dir=run_directory, 
                                                is_markov3_run=is_markov3_ground_truth, 
                                                target_ckpt_str=resolved_ckpt_str)
             pred_beliefs, rmse_correct = None, np.nan
             if pred_data_correct and resolved_ckpt_str in pred_data_correct:
                 layer_data = pred_data_correct[resolved_ckpt_str].get(target_layer, {})
                 pred_beliefs = layer_data.get('predicted_beliefs')
                 rmse_correct = layer_data.get('rmse', np.nan)
             all_plot_data[row_idx]['model_data'][model_type] = pred_beliefs
             
             # Load INCORRECT geometry data
             pred_data_incorrect = load_predictions(run_dir=run_directory, 
                                                  is_markov3_run=(not is_markov3_ground_truth), 
                                                  target_ckpt_str=resolved_ckpt_str)
             rmse_incorrect = np.nan
             if pred_data_incorrect and resolved_ckpt_str in pred_data_incorrect:
                 layer_data_incorrect = pred_data_incorrect[resolved_ckpt_str].get(target_layer, {})
                 rmse_incorrect = layer_data_incorrect.get('rmse', np.nan)
             
             # Load random baseline RMSE
             random_rmse = load_random_baseline_rmse(
                 run_dir=run_directory,
                 target_layer=target_layer,
                 is_markov3_geometry=is_markov3_ground_truth
             )
             
             all_plot_data[row_idx]['metric_data'][model_type] = {
                 'correct': rmse_correct, 'incorrect': rmse_incorrect, 'random': random_rmse
             }

    # --- Coordinate Calculation and Color Generation Loop ---
    print("\n--- Calculating Coordinates and Colors (RGB Scheme) ---")
    for row_idx, data in all_plot_data.items():
        if not data['valid']: 
            continue

        params = data['params']
        gt_beliefs = data['gt_beliefs']
        weights = data['weights']
        model_data = data['model_data']
        belief_dims = gt_beliefs.shape[1]
        experiment_name = calculated_coords[row_idx]['name']

        # Calculate coords for GT (and fit PCA if needed)
        x_gt, y_gt, pca_instance = _calculate_plot_coords(
            gt_beliefs, gt_beliefs, params['use_pca'], params['project_to_simplex'], 
            params['inds_to_plot'], None
        )
        calculated_coords[row_idx]['coords']['gt'] = (x_gt, y_gt)
        calculated_coords[row_idx]['pca'] = pca_instance

        # Calculate coords for models
        for model_type, pred_beliefs in model_data.items():
            x_pred, y_pred, _ = _calculate_plot_coords(
                pred_beliefs, gt_beliefs, params['use_pca'], params['project_to_simplex'], 
                params['inds_to_plot'], pca_instance
            )
            calculated_coords[row_idx]['coords'][model_type] = (x_pred, y_pred)

        # Calculate RGB Colors
        def normalize_dim_color(data_dim):
            min_val, max_val = np.nanmin(data_dim), np.nanmax(data_dim)
            if max_val > min_val: 
                norm = (data_dim - min_val) / (max_val - min_val)
            else: 
                norm = np.ones_like(data_dim) * 0.5
            return np.nan_to_num(norm, nan=0.5)

        alpha_values = transform_for_alpha(
            weights, min_alpha=params['min_alpha'], transformation=params['transformation']
        )

        if 'mess3' in experiment_name.lower() and belief_dims >= 3:
            R = normalize_dim_color(gt_beliefs[:, 0])
            G = normalize_dim_color(gt_beliefs[:, 1])
            B = normalize_dim_color(gt_beliefs[:, 2])
        elif x_gt is not None and y_gt is not None:
            R = normalize_dim_color(x_gt)
            G = normalize_dim_color(y_gt)
            if not params['use_pca'] and not params['project_to_simplex'] and belief_dims > 2:
                 plotted_inds = params['inds_to_plot']
                 third_dim_index = next((i for i in range(belief_dims) if i not in plotted_inds), plotted_inds[0])
                 B_source = gt_beliefs[:, third_dim_index]
            else:
                 B_source = np.sqrt(x_gt**2 + y_gt**2)
            B = normalize_dim_color(B_source)
        else:
            R, G, B = (np.zeros_like(alpha_values) for _ in range(3))

        if all(c is not None and hasattr(c, 'shape') and c.shape == alpha_values.shape for c in [R, G, B]):
             colors_rgba = np.stack([R, G, B, alpha_values], axis=-1)
        else:
             colors_rgba = np.zeros((weights.shape[0], 4))

        calculated_coords[row_idx]['colors'] = colors_rgba

    # --- Plotting Loop ---
    print("\n--- Plotting Grid ---")
    for row_idx, coord_data in calculated_coords.items():
        if not coord_data['valid']:
             for col_idx_off in range(n_cols): 
                 axes[row_idx, col_idx_off].set_axis_off()
             continue
        
        # Plot belief scatters (first 3 columns)
        params = coord_data['params']
        colors_rgba = coord_data['colors']
        point_size = params['point_size']
        
        # Ground truth
        ax_gt = axes[row_idx, 0]
        x_gt, y_gt = coord_data['coords'].get('gt', (None, None))
        _plot_beliefs_on_ax(ax_gt, x_gt, y_gt, colors_rgba, point_size['truth'])
        
        # Model predictions
        model_order = ["Transformer", "LSTM"]
        for col_idx, model_type in enumerate(model_order):
            ax_model = axes[row_idx, col_idx + 1]
            x_pred, y_pred = coord_data['coords'].get(model_type, (None, None))
            _plot_beliefs_on_ax(ax_model, x_pred, y_pred, colors_rgba, point_size['pred'])
        
        # Plot bar chart in the 4th column
        ax_bar = axes[row_idx, 3]
        metric_data = all_plot_data[row_idx].get('metric_data', {})
        is_markov_gt_row = 'markov3' in coord_data['name'].lower()
        is_mess3_row = 'mess3' in coord_data['name'].lower()
        plot_clean_rmse_bars(ax_bar, metric_data, is_markov_gt_row, is_mess3_row)

        # Set aspect ratio for scatter plots only
        for col_idx in range(n_cols - 1):
            axes[row_idx, col_idx].set_aspect('equal', adjustable='box')

    # --- Titles and Labels ---
    cols = ['Ground Truth', 'Transformer', 'LSTM', 'Model Performance']
    rows = [config['name'] for config in plot_config]
    for ax, col in zip(axes[0], cols): 
        ax.set_title(col, fontsize=15)
    for ax, row in zip(axes[:,0], rows): 
        ax.text(-0.1, 0.5, row, rotation=90, fontsize=15, ha='right', va='center', transform=ax.transAxes)

    # Final cleanup: ensure no duplicate legends on bar charts
    for row_idx in range(n_rows):
        ax = axes[row_idx, 3]
        legends = [c for c in ax.get_children() if c.__class__.__name__ == 'Legend']
        for legend in legends:
            if legend.get_title():
                legend.get_title().set_text('')
                legend.get_title().set_visible(False)
        
        # Keep only the first legend
        if len(legends) > 1:
            for legend in legends[1:]:
                legend.remove()

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])

    full_output_path = os.path.join(plot_output_dir, output_filename)
    os.makedirs(plot_output_dir, exist_ok=True)
    
    # Last-chance cleanup - remove ALL legend titles right before saving
    for row_idx in range(n_rows):
        ax = axes[row_idx, 3]
        if ax.get_legend():
            ax.get_legend()._set_loc(ax.get_legend()._loc)
            ax.get_legend().set_title(None)
            if ax.get_legend().get_title():
                ax.get_legend().get_title().set_visible(False)
    
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    print(f"\nGrid figure with metrics saved to {full_output_path}")
    
    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Bloch belief plot for a single checkpoint")
    parser.add_argument("--run-config", type=Path, required=True, help="Path to train2 run_config.yaml")
    parser.add_argument("--wandb-run", type=str, required=True, help="WandB run path (entity/project/run)")
    parser.add_argument("--checkpoint-alias", type=str, default="latest", help="Artifact alias or checkpoint name")
    parser.add_argument("--output", type=Path, default=Path("fig2_bloch_single.png"), help="Output image path")
    parser.add_argument("--layer", type=str, default="combined", help="Activation layer to analyze")
    parser.add_argument("--device", type=str, default="cpu", help="Device for regression (e.g., cpu or cuda:0)")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of KFold splits for regression")

    args = parser.parse_args()
    plot_single_bloch(
        run_config_path=args.run_config,
        wandb_run_path=args.wandb_run,
        checkpoint_alias=args.checkpoint_alias,
        output_path=args.output,
        layer=args.layer,
        n_splits=args.n_splits,
        device=args.device,
    )
