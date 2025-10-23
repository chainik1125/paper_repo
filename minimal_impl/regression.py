"""
Lightweight regression driver that mirrors the paper's activation-analysis pipeline
but is controlled by a simple YAML config instead of the full sweep infrastructure.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None

from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS, RCOND_SWEEP_LIST
from scripts.activation_analysis.data_loading import ActivationExtractor
from scripts.activation_analysis.regression import (
    RegressionAnalyzer,
    run_activation_to_beliefs_regression_kf,
    _train_final_model,
)


# ---------------------------------------------------------------------------
# Utility helpers (copied/adapted from run_regression_analysis.py)
# ---------------------------------------------------------------------------

def _combine_layer_activations(nn_acts: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate activations across layers along the last dimension."""
    collected = [act for act in nn_acts.values()]
    return torch.cat(collected, dim=2)


def _extract_epoch_from_name(name: str) -> Optional[int]:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def find_duplicate_prefixes(nn_inputs: torch.Tensor) -> Dict[Tuple[int, ...], List[Tuple[int, int]]]:
    prefix_map: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    batch_size, seq_len = nn_inputs.shape
    for seq_idx in range(batch_size):
        seq = nn_inputs[seq_idx]
        for pos in range(seq_len):
            prefix = tuple(seq[: pos + 1].cpu().numpy().tolist())
            prefix_map.setdefault(prefix, []).append((seq_idx, pos))
    return prefix_map


def deduplicate_tensor(
    prefix_to_indices: Dict[Tuple[int, ...], List[Tuple[int, int]]],
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    unique_values = []
    unique_indices: List[Tuple[int, int]] = []
    for prefix, indices in prefix_to_indices.items():
        seq_idx, pos = indices[0]
        unique_values.append(tensor[seq_idx, pos])
        unique_indices.append((seq_idx, pos))
    stacked = torch.stack(unique_values)
    return stacked, unique_indices


def deduplicate_data(
    nn_inputs: torch.Tensor,
    probs: torch.Tensor,
    beliefs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], Dict[Tuple[int, ...], List[Tuple[int, int]]]]:
    prefix_to_indices = find_duplicate_prefixes(nn_inputs)
    dedup_probs = []
    dedup_beliefs = []
    dedup_indices: List[Tuple[int, int]] = []
    for prefix, indices in prefix_to_indices.items():
        seq_idx, pos = indices[0]
        total_prob = probs[seq_idx, pos]
        for seq_idx2, pos2 in indices[1:]:
            total_prob += probs[seq_idx2, pos2]
        dedup_probs.append(total_prob)
        dedup_beliefs.append(beliefs[seq_idx, pos])
        dedup_indices.append((seq_idx, pos))
    return (
        torch.stack(dedup_probs),
        torch.stack(dedup_beliefs),
        dedup_indices,
        prefix_to_indices,
    )


def compute_kfold_split(flat_probs: torch.Tensor, n_splits: int = 10, random_state: int = 42):
    from sklearn.model_selection import KFold

    if isinstance(flat_probs, torch.Tensor):
        flat_probs = flat_probs.cpu().detach().numpy()
    all_positions = np.arange(len(flat_probs))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(all_positions))


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WandBConfig:
    entity: str
    project: str
    checkpoints_artifact: Optional[str] = None
    analysis_artifact: Optional[str] = None
    api_key: Optional[str] = None  # optional override
    run_path: Optional[str] = None


@dataclass
class RunConfig:
    name: str
    run_config_path: Path
    checkpoints_dir: Optional[Path] = None
    wandb: Optional[WandBConfig] = None
    n_ctx: Optional[int] = None
    bos: bool = False
    process: Optional[Dict[str, float]] = None
    only_initial_and_final: bool = True
    rcond_values: Optional[List[float]] = None
    rcond: Optional[float] = None
    single_layer: Optional[str] = None


@dataclass
class RegressionConfig:
    output_dir: Path
    device: str = "cpu"
    regression_device: str = "cuda"
    n_splits: int = 10
    random_state: int = 42
    runs: List[RunConfig] = None  # type: ignore


@dataclass
class CheckpointEntry:
    path: Path
    name: str
    epoch: Optional[int] = None


# ---------------------------------------------------------------------------
# Core regression driver
# ---------------------------------------------------------------------------

class RegressionRunner:
    def __init__(self, cfg: RegressionConfig):
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.activation_extractor = ActivationExtractor(device=cfg.device)
        self.reg_analyzer = RegressionAnalyzer(
            device=cfg.regression_device,
            use_efficient_pinv=True,
        )

    def _get_wandb_api(self, wandb_cfg: WandBConfig):
        if wandb is None:
            raise RuntimeError("wandb is not installed; cannot download artifacts.")
        if wandb_cfg.api_key:
            os.environ["WANDB_API_KEY"] = wandb_cfg.api_key
        return wandb.Api()

    def _download_artifact(self, wandb_cfg: WandBConfig, artifact_name: str) -> Path:
        if wandb is None:
            raise RuntimeError("wandb is not installed; cannot download artifacts.")
        if wandb_cfg.api_key:
            os.environ["WANDB_API_KEY"] = wandb_cfg.api_key
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        download_dir = Path(tempfile.mkdtemp(prefix="regression_artifact_"))
        artifact.download(root=str(download_dir))
        return download_dir

    def _load_run_config(self, path: Path) -> Dict[str, Any]:
        with path.open("r") as handle:
            cfg = yaml.safe_load(handle)
        if "process_config" not in cfg:
            raise ValueError(f"Run config at {path} missing 'process_config'.")
        return cfg

    def _prepare_ground_truth(
        self, run_cfg: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[Tuple[int, ...], List[Tuple[int, int]]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        n_ctx = run_cfg.get("model_config", {}).get("n_ctx") or run_cfg.get("n_ctx")
        if n_ctx is None:
            raise ValueError("n_ctx must be specified in the run config.")

        nn_inputs, nn_beliefs, _, nn_probs, _ = prepare_msp_data(
            run_cfg,
            run_cfg["process_config"],
        )

        dedup_probs, dedup_beliefs, dedup_indices, prefix_to_indices = deduplicate_data(
            nn_inputs,
            nn_probs,
            nn_beliefs,
        )
        return (
            dedup_probs,
            dedup_beliefs,
            dedup_indices,
            prefix_to_indices,
            nn_inputs,
            nn_beliefs,
            nn_probs,
        )

    def _instantiate_model(self, run_cfg: Dict[str, Any], checkpoint_path: Path) -> HookedTransformer:
        model_cfg = dict(run_cfg["model_config"])
        model_cfg.setdefault("device", run_cfg["global_config"]["device"])

        # Ensure d_vocab is set from context
        if "d_vocab" not in model_cfg:
            # minimal guess: vocabulary size equals process cardinality
            process_cfg = run_cfg["process_config"]
            name = process_cfg["name"]
            if name == "tom_quantum":
                vocab_size = 4
            elif name == "mess3":
                vocab_size = 3
            else:
                vocab_size = 5
            model_cfg["d_vocab"] = vocab_size

        model_cfg["dtype"] = getattr(torch, model_cfg.get("dtype", "float32"))
        hook_cfg = HookedTransformerConfig(**model_cfg)
        model = HookedTransformer(hook_cfg)
        state = torch.load(checkpoint_path, map_location=run_cfg["global_config"]["device"])
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def _download_artifact(self, wandb_cfg: WandBConfig, artifact_name: str) -> Path:
        api = self._get_wandb_api(wandb_cfg)
        artifact = api.artifact(artifact_name)
        download_dir = Path(tempfile.mkdtemp(prefix="regression_artifact_"))
        artifact.download(root=str(download_dir))
        return download_dir

    def _download_wandb_checkpoints(self, wandb_cfg: WandBConfig) -> List[CheckpointEntry]:
        api = self._get_wandb_api(wandb_cfg)
        entries: List[CheckpointEntry] = []

        if wandb_cfg.run_path:
            run = api.run(wandb_cfg.run_path)
            for artifact in run.logged_artifacts():
                if artifact.type != "model":
                    continue
                download_dir = Path(tempfile.mkdtemp(prefix="regression_artifact_"))
                artifact.download(root=str(download_dir))
                ckpt_files = list(download_dir.glob("*.pt"))
                if not ckpt_files:
                    continue
                artifact_name = artifact.name.split("/")[-1]
                entries.append(
                    CheckpointEntry(
                        path=ckpt_files[0],
                        name=artifact_name,
                        epoch=_extract_epoch_from_name(artifact_name),
                    )
                )
        elif wandb_cfg.checkpoints_artifact:
            download_dir = self._download_artifact(wandb_cfg, wandb_cfg.checkpoints_artifact)
            ckpt_files = list(download_dir.glob("*.pt"))
            for ckpt in ckpt_files:
                entries.append(
                    CheckpointEntry(
                        path=ckpt,
                        name=ckpt.stem,
                        epoch=_extract_epoch_from_name(ckpt.stem),
                    )
                )
        else:
            raise ValueError("WandB configuration must provide either 'run_path' or 'checkpoints_artifact'.")

        return entries

    def _resolve_checkpoints(self, run: RunConfig) -> List[CheckpointEntry]:
        entries: List[CheckpointEntry] = []

        if run.checkpoints_dir is not None:
            for ckpt in sorted(run.checkpoints_dir.glob("*.pt")):
                entries.append(
                    CheckpointEntry(
                        path=ckpt,
                        name=ckpt.stem,
                        epoch=_extract_epoch_from_name(ckpt.stem),
                    )
                )

        if run.wandb is not None:
            entries.extend(self._download_wandb_checkpoints(run.wandb))

        if not entries:
            raise ValueError(f"No checkpoints found for run '{run.name}'.")

        entries.sort(key=lambda e: (e.epoch if e.epoch is not None else float("inf"), e.name))
        return entries

    def _select_checkpoints(self, entries: List[CheckpointEntry], only_first_last: bool) -> List[CheckpointEntry]:
        if not entries:
            return []
        if only_first_last and len(entries) >= 2:
            return [entries[0], entries[-1]]
        return entries

    def run(self):
        for run in self.cfg.runs:
            print(f"\n=== Running regression for: {run.name} ===")

            checkpoints = self._resolve_checkpoints(run)
            checkpoints = self._select_checkpoints(checkpoints, run.only_initial_and_final)
            if not checkpoints:
                print(f"  No checkpoints found for {run.name}; skipping.")
                continue

            # Load run config
            run_cfg = self._load_run_config(run.run_config_path)
            run_cfg.setdefault("global_config", {}).setdefault("device", self.cfg.device)

            if run.process:
                run_cfg["process_config"] = run.process
            if run.n_ctx is not None:
                run_cfg.setdefault("model_config", {})["n_ctx"] = run.n_ctx
                run_cfg["n_ctx"] = run.n_ctx
            run_cfg.setdefault("train_config", {})
            run_cfg["train_config"].setdefault("bos", run.bos)

            # Prepare ground truth beliefs
            (
                dedup_probs,
                dedup_beliefs,
                dedup_indices,
                prefix_to_indices,
                nn_inputs,
                nn_beliefs,
                nn_probs,
            ) = self._prepare_ground_truth(run_cfg)
            kfold_indices = compute_kfold_split(
                dedup_probs,
                n_splits=self.cfg.n_splits,
                random_state=self.cfg.random_state,
            )

            run_output_dir = self.output_dir / run.name
            run_output_dir.mkdir(parents=True, exist_ok=True)

            # Save ground truth snapshots
            ground_truth_data = {
                "probs": dedup_probs.cpu().numpy(),
                "beliefs": dedup_beliefs.cpu().numpy(),
                "indices": np.array(dedup_indices, dtype=object),
            }
            joblib.dump(ground_truth_data, run_output_dir / "ground_truth_data.joblib")

            rcond_values = run.rcond_values or RCOND_SWEEP_LIST
            summary_rows: List[Dict[str, Any]] = []

            for entry in tqdm(checkpoints, desc=f"{run.name} checkpoints"):
                ckpt_name = entry.name
                print(f"  Processing checkpoint: {ckpt_name}")

                model = self._instantiate_model(run_cfg, entry.path)

                # Extract activations
                cache = self.activation_extractor.extract_activations(
                    model,
                    nn_inputs,
                    "transformer",
                    relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
                )
                nn_acts = {layer: acts for layer, acts in cache.items()}
                nn_acts["combined"] = _combine_layer_activations(nn_acts)

                if run.single_layer is not None:
                    if run.single_layer not in nn_acts:
                        available_layers = ", ".join(sorted(nn_acts.keys()))
                        raise ValueError(
                            f"Requested single_layer '{run.single_layer}' not found. "
                            f"Available layers: {available_layers}"
                        )
                    layer_iterable = [(run.single_layer, nn_acts[run.single_layer])]
                else:
                    layer_iterable = nn_acts.items()

                save_data = collections.defaultdict(dict)
                for layer, activations in layer_iterable:
                    dedup_acts, _ = deduplicate_tensor(prefix_to_indices, activations)

                    # Print shapes for debugging
                    print(f"    Layer: {layer}")
                    print(f"      Input shape (dedup_acts): {dedup_acts.shape}")
                    print(f"      Target shape (dedup_beliefs): {dedup_beliefs.shape}")
                    print(f"      Weights shape (dedup_probs): {dedup_probs.shape}")

                    if run.rcond is not None:
                        best_rcond = float(run.rcond)
                        final_metrics = _train_final_model(
                            dedup_acts.to(self.cfg.device),
                            dedup_beliefs.to(self.cfg.device),
                            dedup_probs.to(self.cfg.device),
                            best_rcond,
                        )
                        results = {
                            "best_overall_rcond": best_rcond,
                            "final_metrics": final_metrics,
                        }
                    else:
                        results = run_activation_to_beliefs_regression_kf(
                            self.reg_analyzer,
                            dedup_acts.to(self.cfg.device),
                            dedup_beliefs.to(self.cfg.device),
                            dedup_probs.to(self.cfg.device),
                            kfold_indices,
                            rcond_values=rcond_values,
                        )
                        final_metrics = results.get("final_metrics", {})

                    save_data[layer]["best_rcond"] = results.get("best_overall_rcond")
                    save_data[layer]["predicted_beliefs"] = final_metrics.get("predictions")
                    save_data[layer]["rmse"] = results["final_metrics"]["rmse"]
                    save_data[layer]["mae"] = results["final_metrics"]["mae"]
                    save_data[layer]["r2"] = results["final_metrics"]["r2"]
                    save_data[layer]["dist"] = results["final_metrics"]["dist"]
                    save_data[layer]["mse"] = results["final_metrics"]["mse"]

                joblib.dump(save_data, run_output_dir / f"checkpoint_{ckpt_name}.joblib")

                summary_layer = run.single_layer or "combined"
                summary_metrics = save_data.get(summary_layer)
                if summary_metrics:
                    rmse_arr = summary_metrics.get("rmse")
                    rmse_mean = float(np.mean(rmse_arr)) if rmse_arr is not None else float("nan")
                    r2_value = summary_metrics.get("r2")
                    summary_rows.append(
                        {
                            "checkpoint": ckpt_name,
                            "epoch": entry.epoch,
                            "best_rcond": summary_metrics.get("best_rcond"),
                            "rmse_mean": rmse_mean,
                            "r2": float(r2_value) if r2_value is not None else float("nan"),
                            "layer": summary_layer,
                        }
                    )

            if summary_rows:
                df = pd.DataFrame(summary_rows)
                df = df.sort_values(by=["epoch", "checkpoint"])
                df.to_csv(run_output_dir / "summary.csv", index=False)
                print(f"Summary for {run.name}:\n{df}")


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

def _parse_regression_config(path: Path) -> RegressionConfig:
    with path.open("r") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ValueError(f"Config at {path} is empty.")

    output_dir = Path(raw.get("output_dir", "regression_results"))
    device = raw.get("device", "cpu")
    regression_device = raw.get("regression_device", "cuda")
    n_splits = int(raw.get("n_splits", 10))
    random_state = int(raw.get("random_state", 42))

    runs_raw = raw.get("runs", [])
    runs: List[RunConfig] = []
    for entry in runs_raw:
        wandb_cfg = entry.get("wandb")
        wandb_config = None
        if wandb_cfg:
            wandb_config = WandBConfig(
                entity=wandb_cfg["entity"],
                project=wandb_cfg["project"],
                checkpoints_artifact=wandb_cfg.get("checkpoints_artifact"),
                analysis_artifact=wandb_cfg.get("analysis_artifact"),
                api_key=wandb_cfg.get("api_key"),
                run_path=wandb_cfg.get("run_path"),
            )

        runs.append(
            RunConfig(
                name=entry["name"],
                run_config_path=Path(entry["run_config_path"]),
                checkpoints_dir=Path(entry["checkpoints_dir"]) if entry.get("checkpoints_dir") else None,
                wandb=wandb_config,
                n_ctx=entry.get("n_ctx"),
                bos=bool(entry.get("bos", False)),
                process=entry.get("process"),
                only_initial_and_final=bool(entry.get("only_initial_and_final", True)),
                rcond_values=entry.get("rcond_values"),
                rcond=entry.get("rcond"),
                single_layer=entry.get("single_layer"),
            )
        )

    cfg = RegressionConfig(
        output_dir=output_dir,
        device=device,
        regression_device=regression_device,
        n_splits=n_splits,
        random_state=random_state,
        runs=runs,
    )
    return cfg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run belief regression for arbitrary runs.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to regression_config.yaml",
    )
    args = parser.parse_args()

    cfg = _parse_regression_config(args.config)
    runner = RegressionRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
