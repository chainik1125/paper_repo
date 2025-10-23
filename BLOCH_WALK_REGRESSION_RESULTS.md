# Bloch Walk Regression Test Results

## Summary

Tested the Bloch walk model from WandB run `b9y0lvw8` using the simple regression approach from `mm3.py`.

## Model Details

- **Run**: https://wandb.ai/dmitry2-uiuc/epsilon-bloch-walk_manual/runs/b9y0lvw8
- **Process**: tom_quantum (Bloch Walk)
- **Parameters**: alpha=1.0, beta=7.14142842854285
- **Architecture**:
  - 2 layers
  - d_model=64
  - n_heads=4
  - d_head=16
  - d_mlp=256

## Regression Results

### Trained Model (Checkpoint 179200 - FINAL)
- **RMSE**: 0.0633
- **MSE**: 0.00401
- **R²**: 0.4745
- **Matrix Rank**: 256
- **Checkpoint**: 179200 (fully trained)

### Trained Model (Checkpoint 0 - INITIAL)
- **RMSE**: 0.0628
- **MSE**: 0.00395
- **R²**: 0.4822
- **Matrix Rank**: 256
- **Checkpoint**: 0 (before training)

### Randomly Initialized Model
- **RMSE**: 0.0729
- **MSE**: 0.00531
- **R²**: 0.3030
- **Matrix Rank**: 40

### Comparison
- **Final trained vs Random**: R² difference = 0.171 (trained is better)
- **Initial vs Final trained**: R² difference = 0.008 (negligible!)
- **Random model has R² = 0.30**, which is surprisingly high

## Analysis

### Key Findings

1. **Training barely improves regression performance!**
   - Checkpoint 0 (initial): R² = 0.4822
   - Checkpoint 179200 (final): R² = 0.4745
   - **The model got worse, not better!** This is extremely surprising.

2. **Random model has unexpectedly high R²** (0.30)
   - For a truly random model, we'd expect R² ≈ 0
   - This suggests there may be some structure in the activations even without training
   - Or there's an issue with how beliefs/activations are computed

3. **Low overall R²** (~0.47 for trained model)
   - The paper reports R² ≈ 1.0 for Bloch walk
   - This 0.47 is much lower than expected
   - Possible explanations:
     - Wrong parameters (but we matched the run config)
     - Need different regularization (rcond)
     - Missing preprocessing/standardization
     - The simple lstsq approach doesn't match the full pipeline
     - Need to use weighted regression or standardization

### Important Technical Note

**CUDA lstsq Issue**: Initially, `torch.linalg.lstsq` on CUDA produced:
- Empty rank tensor
- Predictions completely out of range (-13455 to 6785 instead of -0.3 to 0.3)
- Negative R² values

Moving the computation to CPU fixed this issue. This suggests there may be a bug or numerical instability in PyTorch's CUDA implementation of lstsq.

## Comparison with Full Pipeline

If your full regression pipeline (`scripts/activation_analysis/regression.py`) shows:
- **High R² for random models** (>0.5): There's a bug in the R² calculation
- **Different R² for trained models**: May be due to:
  - Standardization differences
  - Weighted regression
  - Different checkpoints
  - The R² formula bug I identified earlier

## Recommendations

1. **Fix the R² formula bug** in `scripts/activation_analysis/regression.py`:
   ```python
   # WRONG (appears in many places):
   explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
   r_squared = (explained_var / total_var).item()

   # CORRECT (should be):
   residual_var = torch.sum((Y_pred * sqrt_weights - Y_weighted)**2)
   r_squared = (1.0 - (residual_var / total_var)).item()
   ```

2. **Test with different rcond values** to see if R² improves

3. **Check which checkpoint epoch** the full pipeline uses (may not be epoch 0)

4. **Verify standardization** - the full pipeline standardizes activations, which may improve results

5. **Use CPU for lstsq** if using CUDA causes issues

## Next Steps

To get R² closer to the paper's reported ≈1.0:
1. Try different rcond values (sweep from 1e-15 to 1e-3)
2. Apply standardization to activations before regression
3. Test with later checkpoints (trained longer)
4. Verify the process parameters match exactly
5. Check if weighted regression helps (though this test doesn't use weights)
