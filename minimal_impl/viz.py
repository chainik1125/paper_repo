#!/usr/bin/env python3
"""
Visualise the MM3 belief fractal and a linear readout from a transformer's
residual stream. The script can be configured entirely via a YAML file or the
CLI.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    _self_path = _Path(__file__).resolve()
    _parent = _self_path.parent
    repo_root = _parent.parent if _parent.name == "minimal_impl" else _parent
    sys.path.append(str(repo_root))

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from epsilon_transformers.analysis.activation_analysis import (
    get_beliefs_for_nn_inputs,
)
from minimal_impl.main import load_config, prepare_transformer_parameters
from minimal_impl.mm3 import build_mm3_process, generate_mm3_transformer_data
from minimal_impl.model import create_hooked_transformer

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent


def _download_from_run(
    run_path: str,
    file_name: Optional[str],
    destination: Path,
    *,
    pattern: Optional[str],
    index: Optional[int],
) -> Tuple[Path, str]:
    """Download a checkpoint from the specified W&B run into ``destination``."""
    if wandb is None:
        raise RuntimeError("wandb is not installed; cannot download run files.")

    api = wandb.Api()
    run = api.run(run_path)
    use_pattern = pattern
    if not use_pattern and _looks_like_glob(file_name):
        use_pattern = file_name

    selected_name: Optional[str] = None
    if not use_pattern and file_name:
        # Try direct download when the name is explicit and without wildcards.
        try:
            file_ref = run.file(file_name)
        except Exception:
            use_pattern = file_name
        else:
            local_path = Path(file_ref.download(root=str(destination), replace=True))
            if local_path.is_dir():
                local_path = local_path / file_name
            if not local_path.exists():
                raise FileNotFoundError(
                    f"Downloaded file not found at {local_path}. "
                    "Check that the file name matches the object logged to W&B."
                )
            selected_name = file_name
            return local_path, selected_name

    # Pattern-based selection.
    names = sorted({file_obj.name for file_obj in run.files()})
    if not use_pattern:
        use_pattern = file_name or "*.pt"
    matches = [name for name in names if fnmatch.fnmatch(name, use_pattern)]
    idx = _select_index(index, len(matches))
    selected_name = matches[idx]
    file_ref = run.file(selected_name)
    local_path = Path(file_ref.download(root=str(destination), replace=True))
    if local_path.is_dir():
        local_path = local_path / selected_name
    if not local_path.exists():
        raise FileNotFoundError(
            f"Downloaded file not found at {local_path}. "
            f"Selected checkpoint name: {selected_name}"
        )
    return local_path, selected_name


def _download_from_artifact(
    artifact_path: str,
    file_name: Optional[str],
    destination: Path,
    *,
    pattern: Optional[str],
    index: Optional[int],
) -> Tuple[Path, str]:
    """Download a checkpoint from a W&B artifact into ``destination``."""
    if wandb is None:
        raise RuntimeError("wandb is not installed; cannot download artifacts.")

    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = Path(artifact.download(root=str(destination), recursive=True))

    use_pattern = pattern
    if not use_pattern and _looks_like_glob(file_name):
        use_pattern = file_name

    if not use_pattern and file_name:
        candidate = artifact_dir / file_name
        if candidate.exists():
            return candidate, file_name
        use_pattern = file_name

    if not use_pattern:
        use_pattern = file_name or "*.pt"

    relative_paths = sorted(
        str(path.relative_to(artifact_dir))
        for path in artifact_dir.rglob("*")
        if path.is_file()
    )
    matches = [name for name in relative_paths if fnmatch.fnmatch(name, use_pattern)]
    idx = _select_index(index, len(matches))
    selected_name = matches[idx]
    candidate = artifact_dir / selected_name
    if not candidate.exists():
        raise FileNotFoundError(
            f"Selected file {selected_name!r} not found after downloading artifact."
        )
    return candidate, selected_name


def _resolve_checkpoint(
    *,
    checkpoint_path: Optional[Path],
    run_path: Optional[str],
    artifact_path: Optional[str],
    file_name: Optional[str],
    pattern: Optional[str],
    index: Optional[int],
    cache_dir: Path,
) -> Tuple[Path, str]:
    """Resolve where to load a checkpoint from (local path, run files, or artifact)."""
    if checkpoint_path:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        return checkpoint_path, checkpoint_path.name

    cache_dir.mkdir(parents=True, exist_ok=True)

    if artifact_path:
        return _download_from_artifact(
            artifact_path,
            file_name,
            cache_dir,
            pattern=pattern,
            index=index,
        )

    if run_path:
        return _download_from_run(
            run_path,
            file_name,
            cache_dir,
            pattern=pattern,
            index=index,
        )

    raise ValueError(
        "A checkpoint source is required. Provide --checkpoint-path, "
        "--wandb-run, or --artifact; or set one in the viz YAML."
    )


def _project_to_simplex(beliefs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D belief vectors onto a 2D simplex (equilateral triangle).
    """
    if beliefs.shape[1] < 3:
        padded = np.zeros((beliefs.shape[0], 3), dtype=np.float32)
        padded[:, : beliefs.shape[1]] = beliefs
        beliefs = padded

    x_temp = beliefs[:, 0] - beliefs[:, 1] / 2.0 - beliefs[:, 2] / 2.0
    y_temp = np.sqrt(3.0) / 2.0 * (beliefs[:, 1] - beliefs[:, 2])
    x = -y_temp
    y = x_temp
    return x, y


def _belief_colors(beliefs: np.ndarray) -> np.ndarray:
    """
    Map belief vectors to RGB colours by using the first three coordinates.
    """
    if beliefs.shape[1] < 3:
        padded = np.zeros((beliefs.shape[0], 3), dtype=np.float32)
        padded[:, : beliefs.shape[1]] = beliefs
        beliefs = padded
    beliefs = np.clip(beliefs, 0.0, 1.0)
    return beliefs


def _alpha_from_probabilities(probs: np.ndarray, min_alpha: float = 0.05) -> np.ndarray:
    """Convert sequence probabilities into transparency values."""
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    if probs.size == 0:
        return np.array([], dtype=np.float32)
    normalized = probs / (probs.max() + 1e-8)
    alpha = np.cbrt(normalized)
    alpha = min_alpha + (1.0 - min_alpha) * alpha
    return np.clip(alpha, min_alpha, 1.0)


def _select_index(index: Optional[int], total: int) -> int:
    """Normalise an index (supports negative values)."""
    if total <= 0:
        raise ValueError("No checkpoint files matched the requested pattern.")
    if index is None:
        index = -1
    try:
        idx = int(index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint index: {index}") from exc
    if idx < 0:
        idx += total
    idx = max(0, min(idx, total - 1))
    return idx


def _looks_like_glob(value: Optional[str]) -> bool:
    """Return True if ``value`` appears to include glob wildcards."""
    if not value:
        return False
    return any(ch in value for ch in "*?[]")


def _prepare_dataset(mm3_cfg: dict, device: torch.device):
    dataset = generate_mm3_transformer_data(
        n_ctx=mm3_cfg.get("n_ctx", 7),
        bos=mm3_cfg.get("bos", False),
        x=mm3_cfg.get("x", 0.15),
        a=mm3_cfg.get("a", 0.6),
        device=device,
        as_numpy=False,
    )
    return dataset


def _build_process(mm3_cfg: dict):
    return build_mm3_process(
        x=mm3_cfg.get("x", 0.15),
        a=mm3_cfg.get("a", 0.6),
    )


@torch.no_grad()
def compute_linear_readout(
    model: torch.nn.Module,
    dataset,
    *,
    mm3_cfg: dict,
    activation_name: str,
) -> dict:
    """
    Fit a linear map from the residual stream activations to MM3 belief states
    and return the ground-truth and predicted beliefs.
    """
    device = model.cfg.device

    process = _build_process(mm3_cfg)
    bos = mm3_cfg.get("bos", False)
    n_ctx = mm3_cfg.get("n_ctx", dataset.transformer_inputs.shape[1] - 1)
    seq_len = n_ctx + 1
    msp_depth = seq_len + (1 if bos else 2)
    msp = process.derive_mixed_state_tree(depth=msp_depth)
    tree_paths = msp.paths
    tree_beliefs = msp.belief_states
    tree_unnormalized = msp.unnorm_belief_states
    path_probs = msp.path_probs

    msp_beliefs = [tuple(round(float(b), 5) for b in belief.squeeze()) for belief in tree_beliefs]
    msp_belief_index = {tuple_belief: idx for idx, tuple_belief in enumerate(set(msp_beliefs))}
    probs_dict = {tuple(path): prob for path, prob in zip(tree_paths, path_probs)}

    contexts = dataset.transformer_inputs[:, :-1].to(torch.int64)
    contexts = contexts.to(device)

    beliefs_out = get_beliefs_for_nn_inputs(
        contexts.cpu(),
        msp_belief_index,
        tree_paths,
        tree_beliefs,
        tree_unnormalized,
        probs_dict,
    )
    belief_states = beliefs_out[0].to(device).float()

    _, cache = model.run_with_cache(
        contexts,
        names_filter=lambda name: name == activation_name,
    )
    activations = cache[activation_name].float()

    acts_flat = activations.reshape(-1, activations.shape[-1])
    beliefs_flat = belief_states.reshape(-1, belief_states.shape[-1])

    lstsq = torch.linalg.lstsq(acts_flat, beliefs_flat)
    linear_map = lstsq.solution
    preds_flat = acts_flat @ linear_map
    preds = preds_flat.reshape_as(belief_states)

    residuals = preds_flat - beliefs_flat
    mse = torch.mean(residuals.pow(2)).item()
    rmse = float(np.sqrt(mse))

    total_variance = torch.mean(
        (beliefs_flat - beliefs_flat.mean(dim=0, keepdim=True)).pow(2)
    )
    r_squared = 1.0 - (torch.mean(residuals.pow(2)) / total_variance) if total_variance > 0 else float("nan")
    if torch.is_tensor(r_squared):
        r_squared = float(r_squared.item())

    return {
        "belief_states": belief_states.detach().cpu(),
        "predicted_beliefs": preds.detach().cpu(),
        "linear_map": linear_map.detach().cpu(),
        "rmse": rmse,
        "mse": mse,
        "r_squared": r_squared,
    }


def _plot_fractal(
    *,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    title_suffix: str,
    sample: Optional[int] = None,
):
    """
    Render ground-truth and predicted belief fractals side-by-side.
    """
    gt_final = ground_truth[:, -1, :]
    pred_final = predicted[:, -1, :]

    num_points = gt_final.shape[0]
    indices = np.arange(num_points)
    if sample is not None and sample < num_points:
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(indices, size=sample, replace=False)
        gt_final = gt_final[indices]
        pred_final = pred_final[indices]
        probs = probs[indices]

    alpha = _alpha_from_probabilities(probs)

    gt_colors = _belief_colors(gt_final)
    pred_colors = _belief_colors(pred_final)
    gt_rgba = np.concatenate([gt_colors, alpha[:, None]], axis=1)
    pred_rgba = np.concatenate([pred_colors, alpha[:, None]], axis=1)

    gt_x, gt_y = _project_to_simplex(gt_final)
    pred_x, pred_y = _project_to_simplex(pred_final)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(gt_x, gt_y, color=gt_rgba, s=6, linewidths=0, rasterized=True)
    axes[0].set_title("Ground-Truth Beliefs")
    axes[0].set_axis_off()

    axes[1].scatter(pred_x, pred_y, color=pred_rgba, s=6, linewidths=0, rasterized=True)
    axes[1].set_title("Linear Readout Predictions")
    axes[1].set_axis_off()

    fig.suptitle(f"MM3 Belief Fractal — {title_suffix}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_state_dict(checkpoint: Path, device: torch.device) -> dict:
    """Load a Torch state dict from ``checkpoint``."""
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {checkpoint} does not contain a state dict.")
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise the MM3 belief fractal and its linear readout.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("minimal_impl/mm3_config.yaml"),
        help="Path to the MM3 training configuration used for the run.",
    )
    parser.add_argument(
        "--viz-config",
        type=Path,
        help="YAML file describing the visualization configuration. CLI flags override it.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Local path to a checkpoint containing the trained model weights.",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        help="Weights & Biases run path (entity/project/run_id) that contains the checkpoint file.",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        help="Short W&B run id (e.g. '25pq51gr'). Entity/project are inferred or can be set via config.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="W&B entity to use when constructing run or artifact paths.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project to use when constructing run or artifact paths.",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        help="Weights & Biases artifact path (entity/project/artifact:alias) to download the checkpoint from.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="model.pt",
        help="File name of the checkpoint inside the run or artifact.",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        help="Glob pattern used to match checkpoint files (e.g. 'checkpoints/model-step*').",
    )
    parser.add_argument(
        "--checkpoint-index",
        type=int,
        default=-1,
        help="Index of the matched checkpoint to use (supports negative indices).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Figs/mm3_fractal.png"),
        help="Where to save the generated figure.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="ln_final.hook_normalized",
        help="Activation name to read from the residual stream cache.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="If provided, randomly sample this many contexts for visualization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run the analysis on (e.g. 'cpu', 'cuda', or 'auto').",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".wandb-cache"),
        help="Directory used to cache downloaded checkpoints.",
    )
    parser.add_argument(
        "--artifact-alias",
        type=str,
        default="latest",
        help="Artifact alias/version when inferring the run history artifact from a run id.",
    )
    return parser


def _load_viz_yaml(path: Path) -> Dict[str, Any]:
    """Load and normalise the visualization YAML configuration."""
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Visualization config must be a mapping.")
    cfg = data.get("viz", data)
    cfg = dict(cfg)
    checkpoint_cfg = cfg.pop("checkpoint", {})
    if isinstance(checkpoint_cfg, dict):
        if "path" in checkpoint_cfg:
            cfg["checkpoint_path"] = checkpoint_cfg["path"]
        if "run_id" in checkpoint_cfg:
            cfg["wandb_run_id"] = checkpoint_cfg["run_id"]
        if "entity" in checkpoint_cfg:
            cfg["wandb_entity"] = checkpoint_cfg["entity"]
        if "project" in checkpoint_cfg:
            cfg["wandb_project"] = checkpoint_cfg["project"]
        if "wandb_run" in checkpoint_cfg:
            cfg["wandb_run"] = checkpoint_cfg["wandb_run"]
        if "artifact" in checkpoint_cfg:
            cfg["artifact"] = checkpoint_cfg["artifact"]
        if "file" in checkpoint_cfg:
            cfg["checkpoint_file"] = checkpoint_cfg["file"]
        if "pattern" in checkpoint_cfg:
            cfg["checkpoint_pattern"] = checkpoint_cfg["pattern"]
        if "index" in checkpoint_cfg:
            cfg["checkpoint_index"] = checkpoint_cfg["index"]
        if "artifact_alias" in checkpoint_cfg:
            cfg["artifact_alias"] = checkpoint_cfg["artifact_alias"]
    return cfg


def _resolve_option(
    name: str,
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: Dict[str, Any],
) -> Any:
    """Resolve a configuration value preferring CLI overrides, then YAML, then defaults."""
    cli_value = getattr(args, name)
    default_value = parser.get_default(name)
    if cli_value is not None and cli_value != default_value:
        return cli_value
    if config and name in config and config[name] is not None:
        return config[name]
    return default_value


def _resolve_path(value: Any, *, fallback_dir: Path) -> Optional[Path]:
    """Coerce ``value`` into a Path, checking both CWD and ``fallback_dir``."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    fallback_candidate = fallback_dir / path
    if fallback_candidate.exists():
        return fallback_candidate
    return cwd_candidate


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    viz_cfg: Dict[str, Any] = {}
    if args.viz_config:
        viz_config_path = _resolve_path(args.viz_config, fallback_dir=SCRIPT_DIR)
        if viz_config_path is None or not viz_config_path.exists():
            raise FileNotFoundError(f"Visualization config not found: {args.viz_config}")
        viz_cfg = _load_viz_yaml(viz_config_path)

    config_value = _resolve_option("config", args=args, parser=parser, config=viz_cfg)
    mm3_config_path = _resolve_path(config_value, fallback_dir=SCRIPT_DIR)
    if mm3_config_path is None or not mm3_config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_value}")

    cfg = load_config(mm3_config_path)
    mm3_cfg = cfg.get("mm3", {}) if isinstance(cfg, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    mm3_wandb_cfg = cfg.get("wandb", {}) if isinstance(cfg, dict) else {}

    checkpoint_path_value = _resolve_option("checkpoint_path", args=args, parser=parser, config=viz_cfg)
    checkpoint_path = (
        _resolve_path(checkpoint_path_value, fallback_dir=SCRIPT_DIR) if checkpoint_path_value else None
    )

    wandb_run = _resolve_option("wandb_run", args=args, parser=parser, config=viz_cfg)
    artifact = _resolve_option("artifact", args=args, parser=parser, config=viz_cfg)
    wandb_run_id = _resolve_option("wandb_run_id", args=args, parser=parser, config=viz_cfg)
    wandb_entity = _resolve_option("wandb_entity", args=args, parser=parser, config=viz_cfg)
    wandb_project = _resolve_option("wandb_project", args=args, parser=parser, config=viz_cfg)
    checkpoint_file = _resolve_option("checkpoint_file", args=args, parser=parser, config=viz_cfg)
    checkpoint_pattern = _resolve_option("checkpoint_pattern", args=args, parser=parser, config=viz_cfg)
    checkpoint_index_value = _resolve_option("checkpoint_index", args=args, parser=parser, config=viz_cfg)
    artifact_alias = _resolve_option("artifact_alias", args=args, parser=parser, config=viz_cfg) or "latest"

    checkpoint_pattern = checkpoint_pattern if checkpoint_pattern not in ("", None) else None
    checkpoint_index_for_selection = (
        checkpoint_index_value if checkpoint_index_value not in ("", None) else None
    )

    default_entity = mm3_wandb_cfg.get("entity")
    default_project = mm3_wandb_cfg.get("project")
    if not wandb_entity:
        wandb_entity = default_entity or os.environ.get("WANDB_ENTITY")
    if not wandb_project:
        wandb_project = default_project or os.environ.get("WANDB_PROJECT")

    if wandb_run_id:
        if not wandb_entity or not wandb_project:
            raise ValueError(
                "W&B entity and project are required when specifying a run_id. "
                "Set them in the viz config, CLI, MM3 config, or environment."
            )
        if not wandb_run:
            wandb_run = f"{wandb_entity}/{wandb_project}/{wandb_run_id}"
        if not artifact:
            artifact = f"{wandb_entity}/{wandb_project}/run-{wandb_run_id}-history:{artifact_alias}"

    output_value = _resolve_option("output", args=args, parser=parser, config=viz_cfg)
    output_path = _resolve_path(output_value, fallback_dir=Path.cwd())
    if output_path is None:
        raise ValueError("Output path could not be resolved.")

    activation = _resolve_option("activation", args=args, parser=parser, config=viz_cfg)
    sample_value = _resolve_option("sample", args=args, parser=parser, config=viz_cfg)
    sample = int(sample_value) if sample_value not in (None, "") else None
    if sample is not None and sample <= 0:
        sample = None

    device_value = _resolve_option("device", args=args, parser=parser, config=viz_cfg)
    cache_dir_value = _resolve_option("cache_dir", args=args, parser=parser, config=viz_cfg)
    cache_dir = _resolve_path(cache_dir_value, fallback_dir=Path.cwd())
    if cache_dir is None:
        cache_dir = Path(".wandb-cache")

    checkpoint_file_normalized = (
        checkpoint_file if checkpoint_file not in (None, "") else None
    )

    checkpoint_path_resolved, checkpoint_name = _resolve_checkpoint(
        checkpoint_path=checkpoint_path,
        run_path=wandb_run,
        artifact_path=artifact,
        file_name=checkpoint_file_normalized,
        pattern=checkpoint_pattern,
        index=checkpoint_index_for_selection,
        cache_dir=cache_dir,
    )

    if device_value == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device_value)

    dataset = _prepare_dataset(mm3_cfg, device=resolved_device)
    transformer_params = prepare_transformer_parameters(model_cfg, str(resolved_device))
    model = create_hooked_transformer(dataset, transformer_params)
    state_dict = load_state_dict(checkpoint_path_resolved, resolved_device)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    readout = compute_linear_readout(
        model,
        dataset,
        mm3_cfg=mm3_cfg,
        activation_name=str(activation),
    )

    probs = dataset.probabilities.detach().cpu().numpy().astype(np.float32)
    ground_truth = readout["belief_states"].numpy()
    predicted = readout["predicted_beliefs"].numpy()

    if checkpoint_index_for_selection is None:
        checkpoint_index_display = None
    else:
        try:
            checkpoint_index_display = int(checkpoint_index_for_selection)
        except (TypeError, ValueError):
            checkpoint_index_display = checkpoint_index_for_selection

    summary = {
        "config": str(mm3_config_path),
        "checkpoint": str(checkpoint_path_resolved),
        "checkpoint_name": checkpoint_name,
        "rmse": readout["rmse"],
        "mse": readout["mse"],
        "r_squared": readout["r_squared"],
        "activation": activation,
        "num_sequences": int(ground_truth.shape[0]),
        "context_length": int(ground_truth.shape[1]),
        "belief_dim": int(ground_truth.shape[2]),
        "device": str(resolved_device),
        "sampled_points": sample,
        "output": str(output_path),
        "wandb_run": wandb_run,
        "wandb_run_id": wandb_run_id,
        "artifact": artifact,
        "checkpoint_pattern": checkpoint_pattern,
        "checkpoint_index": checkpoint_index_display,
    }
    print(json.dumps(summary, indent=2))

    _plot_fractal(
        ground_truth=ground_truth,
        predicted=predicted,
        probs=probs,
        output_path=output_path,
        title_suffix=os.path.basename(str(output_path)),
        sample=sample,
    )
    print(f"Saved fractal comparison to {output_path}")


if __name__ == "__main__":
    main()
