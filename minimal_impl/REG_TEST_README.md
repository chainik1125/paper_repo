# reg_test.py - Regression Test Script

This script tests the regression implementation by loading a trained model from WandB and performing the simple regression approach from `mm3.py`.

## Purpose

The script helps diagnose issues with the full regression pipeline by:
1. Loading a model from WandB (like the full pipeline does)
2. Performing the simple, straightforward regression from `mm3.py`
3. Reporting metrics to compare against the full pipeline results

## Usage

### Basic Usage

```bash
uv run python minimal_impl/reg_test.py \
  --config minimal_impl/reg_test_config.yaml \
  --entity YOUR_WANDB_ENTITY \
  --project YOUR_WANDB_PROJECT \
  --run-path YOUR_ENTITY/YOUR_PROJECT/RUN_ID
```

### Full Example

```bash
# Test with a specific checkpoint
uv run python minimal_impl/reg_test.py \
  --config minimal_impl/reg_test_config.yaml \
  --entity dmitry2-uiuc \
  --project epsilon-mm3 \
  --run-path dmitry2-uiuc/epsilon-mm3/abc123xyz \
  --checkpoint final_model.pt \
  --device cuda \
  --rcond 1e-10
```

### Command Line Arguments

- `--config`: Path to config YAML (required)
- `--entity`: WandB entity name (required)
- `--project`: WandB project name (required)
- `--run-path`: Full WandB run path in format `entity/project/run_id` (required)
- `--checkpoint`: Checkpoint name to load (default: `final_model.pt`)
- `--device`: Device to use (default: auto-detect CUDA)
- `--rcond`: Regularization parameter for lstsq (default: 1e-10)
- `--api-key`: WandB API key (optional, can use `WANDB_API_KEY` env var)
- `--test-random`: Also test with a randomly initialized model for comparison

## Configuration File

The config file must have this structure:

```yaml
model_config:
  n_ctx: 7           # Context length
  n_layers: 2        # Number of transformer layers
  d_model: 128       # Model dimension
  n_heads: 2         # Number of attention heads
  d_mlp: 512         # MLP hidden dimension
  d_vocab: 3         # Vocabulary size (3 for mess3)
  dtype: float32

process_config:
  name: mess3        # Process name
  x: 0.05           # Process parameter
  a: 0.85           # Process parameter

train_config:
  bos: false         # Whether sequences have BOS token
```

## What It Does

1. **Downloads checkpoint** from WandB artifacts
2. **Loads model** with the specified configuration
3. **Generates ground truth** belief states from the GHMM process
4. **Extracts activations** from all transformer layers
5. **Performs regression** using simple lstsq (no standardization, no weighted sqrt)
6. **Reports metrics**: RMSE, MSE, R², and matrix rank

## Testing Random Models

To verify the regression is working correctly, use the `--test-random` flag to test a randomly initialized model:

```bash
uv run python minimal_impl/reg_test.py \
  --config minimal_impl/reg_test_config.yaml \
  --entity YOUR_ENTITY \
  --project YOUR_PROJECT \
  --run-path YOUR_ENTITY/YOUR_PROJECT/RUN_ID \
  --test-random
```

This will test both the trained model and a randomly initialized model. The random model should have R² near 0, while the trained model should have high R². If the random model has high R², there's an issue with the regression implementation.

## Expected Output

### Without --test-random

```
Loading config from minimal_impl/reg_test_config.yaml
Downloading checkpoint from dmitry2-uiuc/epsilon-mm3/abc123xyz
Downloaded checkpoint: /tmp/reg_test_xyz/final_model.pt
Loading model from /tmp/reg_test_xyz/final_model.pt
Model loaded on cuda
Preparing ground truth belief states
Number of sequences: 2187
Sequence length: 8
Belief dimension: 3

Running simple regression (mm3.py style)
Activation shape: torch.Size([15309, 1280])
Belief shape: torch.Size([15309, 3])
Rcond: 1e-10

=== Regression Results ===
rmse: 0.0234
mse: 0.000548
rank: 1280
r2: 0.9876

Test complete!
```

### With --test-random

```
=== Trained Model Regression Results ===
rmse: 0.0234
mse: 0.000548
rank: 1280
r2: 0.9876

==================================================
Testing with RANDOMLY INITIALIZED model for comparison
==================================================
Running regression on randomly initialized model
Activation shape: torch.Size([15309, 1280])
Belief shape: torch.Size([15309, 3])
Rcond: 1e-10

=== Random Model Regression Results ===
rmse: 0.5821
mse: 0.3388
rank: 1280
r2: 0.0123

=== Comparison ===
Trained R²: 0.987600
Random R²:  0.012300
Difference: 0.975300

✓ Random model has low R² as expected.

Test complete!
```


## Troubleshooting

### "wandb is not installed"
```bash
uv add wandb
```

### "No checkpoint found"
- Verify the run path is correct: `entity/project/run_id`
- Check that the run has model artifacts uploaded
- Try `--checkpoint initial_model.pt` if final checkpoint doesn't exist

### Config validation errors
- Ensure `d_vocab` matches the process (3 for mess3, 4 for tom_quantum)
- Verify `n_ctx` matches the training config
- Check that process parameters match the training run

## Comparing with Full Pipeline

To compare this with the full regression pipeline:

1. Run this script and note the R² value
2. Run the full pipeline on the same model
3. Compare R² values - they should be similar for a trained model
4. For a **randomly initialized** model, R² should be very low (near 0)

If the full pipeline shows high R² for random models but this script shows low R², there's likely an issue with the full pipeline's standardization or weighting.
