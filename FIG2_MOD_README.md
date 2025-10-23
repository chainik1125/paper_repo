# Fig2_mod.py - Modified Figure 2 Generator

This script generates mess3 belief visualizations by loading models directly from W&B run IDs.

## Overview

Unlike the original `Fig2.py` which loads pre-computed regression analysis from HuggingFace, `fig2_mod.py`:

1. **Downloads checkpoints** from W&B runs (latest checkpoint by default)
2. **Runs regression analysis** on-the-fly to map neural activations to belief states
3. **Generates visualization** for the mess3 process only

## Requirements

- WANDB_API_KEY environment variable set
- Both models should be trained on the mess3 process
- Models can be Transformer or RNN (LSTM/GRU/RNN)

## Usage

### Basic Usage

```bash
uv run python fig2_mod.py \
    --transformer-run "entity/project/transformer_run_id" \
    --lstm-run "entity/project/lstm_run_id"
```

### Advanced Usage

```bash
uv run python fig2_mod.py \
    --transformer-run "your-entity/quantum-reps/abc123" \
    --lstm-run "your-entity/quantum-reps/def456" \
    --output "results/my_mess3_viz.png" \
    --device "cuda" \
    --rcond 1e-8 \
    --cache-dir "./my_wandb_cache"
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--transformer-run` | Yes | - | W&B run path for transformer model |
| `--lstm-run` | Yes | - | W&B run path for LSTM/RNN model |
| `--output` | No | `Figs/Fig2_mod.png` | Output path for generated figure |
| `--device` | No | `cpu` | Device for computation (`cpu` or `cuda`) |
| `--rcond` | No | `1e-10` | Regularization parameter for regression |
| `--cache-dir` | No | `./wandb_cache` | Directory to cache downloaded checkpoints |

## Finding Your W&B Run Path

The run path has the format: `entity/project/run_id`

You can find it from your W&B dashboard URL:
```
https://wandb.ai/entity/project/runs/run_id
                    ^^^^^^  ^^^^^^^      ^^^^^^
                    entity  project      run_id
```

Example:
- URL: `https://wandb.ai/simplex-ai/quantum-transformers/runs/abc123xyz`
- Run path: `simplex-ai/quantum-transformers/abc123xyz`

## Output

The script generates a single-row visualization with 4 panels:

1. **Ground Truth**: Theoretical mess3 belief states (simplex projection)
2. **Transformer**: Transformer's learned representation mapped to belief space
3. **LSTM**: LSTM's learned representation mapped to belief space
4. **Model Performance**: RMSE bar chart comparing models

## How It Works

### Pipeline Steps

1. **Download Checkpoints**: Downloads latest checkpoint from each W&B run
2. **Load Models**: Initializes model architectures and loads weights
3. **Generate Ground Truth**: Computes theoretical belief states for mess3 process
4. **Extract Activations**: Runs models on input sequences to get activations
5. **Run Regression**: Performs weighted least squares regression (10-fold CV)
6. **Visualize**: Creates belief space scatter plots and performance charts

### Regression Analysis

The script uses weighted ridge regression with K-fold cross-validation:
- **K-folds**: 10 splits
- **Regularization**: Controlled by `--rcond` parameter
- **Weighting**: Samples weighted by sequence probability
- **Metrics**: RMSE, MAE, R² computed on test folds

### Activation Extraction

**Transformers**:
- Extracts residual stream from all layers: `blocks.{i}.hook_resid_post`
- Includes final layer normalization: `ln_final.hook_normalized`
- Concatenates all layer activations

**RNNs** (LSTM/GRU/RNN):
- Extracts hidden states from all layers
- Adds one-hot encoded input tokens
- Concatenates layer states + inputs

## Comparison with Original Fig2.py

| Feature | Fig2.py | fig2_mod.py |
|---------|---------|-------------|
| Data Source | HuggingFace (pre-computed) | W&B runs (live) |
| Processes | 3 (mess3, bloch, moon) | 1 (mess3 only) |
| Regression | Pre-computed | Computed on-the-fly |
| Model Loading | Download from HF dataset | Download from W&B |
| Flexibility | Fixed model IDs | Any W&B run |
| Speed | Fast (uses cached results) | Slower (runs regression) |

## Example Workflow

```bash
# 1. Train models on mess3
uv run python scripts/launcher_cuda_parallel.py \
    --config scripts/experiment_config_transformer_mess3_bloch.yaml

# 2. Note the W&B run IDs from training output
# Transformer run: simplex-ai/quantum-transformers/abc123
# LSTM run: simplex-ai/quantum-transformers/def456

# 3. Generate visualization
uv run python fig2_mod.py \
    --transformer-run "simplex-ai/quantum-transformers/abc123" \
    --lstm-run "simplex-ai/quantum-transformers/def456" \
    --device "cuda"

# 4. View output
open Figs/Fig2_mod.png
```

## Troubleshooting

### "wandb is not installed"
```bash
uv add wandb
```

### "No files matching '*.pt' found"
- Check that the run completed training and saved checkpoints
- Verify the run path is correct
- Ensure you have access to the W&B project

### "WANDB_API_KEY not set"
```bash
export WANDB_API_KEY="your_api_key_here"
```

### Memory issues
- Use `--device cpu` if CUDA runs out of memory
- Reduce batch size in the model configuration

## Notes

- The script uses the **first model's config** to generate ground truth data
- Ensure both models were trained on the same mess3 process configuration
- Downloaded checkpoints are cached in `--cache-dir` to avoid re-downloading
- The visualization uses simplex projection for mess3 (3-state classical process)
