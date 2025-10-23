"""
Regression analysis functions for mapping activations to belief states.
"""
import torch
import numpy as np
import pandas as pd
import logging
import traceback
from tqdm.auto import tqdm
from collections import defaultdict

from scripts.activation_analysis.utils import standardize, unstandardize_coefficients, report_variance_explained, setup_logging
from scripts.activation_analysis.config import REPORT_VARIANCE, DO_BASELINE

# Module logger
logger = logging.getLogger("regression")

def compute_efficient_pinv_from_svd(matrix, rcond_values):
    """
    Efficiently compute pseudoinverses for multiple rcond values using a single SVD.
    
    Args:
        matrix: The input matrix for which to compute pseudoinverses
        rcond_values: List of rcond values to use for thresholding
        
    Returns:
        Dictionary mapping each rcond value to its corresponding pseudoinverse matrix,
        and the singular values from the SVD
    """
    # Compute SVD once
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    
    # Get the maximum singular value for relative thresholding
    max_singular_value = S[0]
    
    # Dictionary to store results
    pinvs = {}
    
    # For each rcond value, create a custom pseudoinverse
    for rcond in rcond_values:
        # Compute threshold for this rcond value
        threshold = rcond * max_singular_value
        
        # Create reciprocal of singular values with thresholding
        S_pinv = torch.zeros_like(S)
        above_threshold = S > threshold
        S_pinv[above_threshold] = 1.0 / S[above_threshold]
        
        # Compute pseudoinverse as V * S_pinv * U.T
        # Use broadcasting to multiply each singular value with corresponding column of V
        S_pinv_diag = torch.diag(S_pinv)
        pinv_matrix = Vh.T @ S_pinv_diag @ U.T
        
        # Store result for this rcond value
        pinvs[rcond] = pinv_matrix
    
    return pinvs, S

class RegressionAnalyzer:
    """Class for performing regression analysis on activations."""
    
    def __init__(self, device='cpu', use_efficient_pinv=False):
        self.device = device
        self.use_efficient_pinv = use_efficient_pinv
    
    def process_single_layer(self, act_tensor, belief_states, nn_word_probs, rcond, layer_name=None):
        """Process regression for a single layer."""
        try:
            # Check shapes and log information
            logger.debug(f"Processing layer with activation shape: {act_tensor.shape}")
            logger.debug(f"Belief states shape: {belief_states.shape}")
            logger.debug(f"Word probs shape: {nn_word_probs.shape}")
            
            # Reshape tensors for regression
            X = act_tensor.view(-1, act_tensor.shape[-1]).to(self.device)
            Y = belief_states.view(-1, belief_states.shape[-1]).to(self.device)
            weights = nn_word_probs.view(-1).to(self.device)
            
            # Check for shape compatibility
            if X.shape[0] != Y.shape[0] or X.shape[0] != weights.shape[0]:
                raise ValueError(f"Shape mismatch: X: {X.shape}, Y: {Y.shape}, weights: {weights.shape}")
                
            # Normalize weights
            weights = weights / weights.sum()
            
            # --- STEP 1: Standardize features ---
            X_std, mean, std = standardize(X)
            
            # --- STEP 2: SVD for variance analysis (separate from regression) ---
            if REPORT_VARIANCE:
                var_expl_str, singular_values = report_variance_explained(X_std, REPORT_VARIANCE)
            else:
                var_expl_str, singular_values = "", None
            
            # --- STEP 3: Weighted regression ---
            # Add bias term
            ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
            X_std_bias = torch.cat([ones, X_std], dim=1)
            
            # Apply weights
            sqrt_weights = torch.sqrt(weights).unsqueeze(1)
            X_weighted = X_std_bias * sqrt_weights
            Y_weighted = Y * sqrt_weights

            # Calculate beta using pseudoinverse with specified regularization
            A = X_weighted.T @ X_weighted
            beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)

            # Convert coefficients back to original scale
            beta = unstandardize_coefficients(beta_std, mean, std)
            
            # --- STEP 4: Evaluate regression ---
            # Make predictions
            ones_orig = torch.ones(X.shape[0], 1, device=X.device)
            X_orig_bias = torch.cat([ones_orig, X], dim=1)
            Y_pred = X_orig_bias @ beta
            
            # Calculate weighted error
            distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
            weighted_distances = distances * weights
            mean_dist = weighted_distances.sum()
            
            # Calculate R-squared
            total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
            residual_var = torch.sum((Y_pred * sqrt_weights - Y_weighted)**2)
            r_squared = (1.0 - (residual_var / total_var)).item() if total_var > 0 else 0
            
            dims = X.shape[1]
            return mean_dist.item(), dims, var_expl_str, singular_values, beta.cpu().numpy(), r_squared
            
        except Exception as e:
            logger.error(f"Error in process_single_layer: {e}")
            raise
    
    def process_activation_layers(self, acts, belief_states, nn_word_probs, rcond, cached_svds=None):
        """
        Process regression for all activation layers and their combinations.
        
        Args:
            acts: Dictionary of activations by layer
            belief_states: Target belief states
            nn_word_probs: Probability weights
            rcond: Regularization parameter
            cached_svds: Optional dict of pre-computed SVDs to avoid redundant computation
            
        Returns:
            DataFrame of results, singular value dict, weights dict, best layer name, best layers dict
        """
        records = []
        best_layers = {}
        weights = {}
        singular_values_dict = {}
        best_idx = 0
        
        # Process each layer individually
        for layer_idx, (layer_name, act_tensor) in enumerate(acts.items()):
            # Use cached SVD data if available
            if cached_svds and layer_name in cached_svds:
                X = cached_svds[layer_name]['X']
                X_std = cached_svds[layer_name]['X_std']
                Y = cached_svds[layer_name]['Y'] 
                weights_tensor = cached_svds[layer_name]['weights']
                singular_vals = cached_svds[layer_name]['singular_values']
                var_expl = cached_svds[layer_name]['var_expl']
                mean = cached_svds[layer_name]['mean']
                std = cached_svds[layer_name]['std']
                dims = X.shape[1]
                
                # Check if we have precomputed efficient pinvs
                if self.use_efficient_pinv and 'efficient_pinvs' in cached_svds[layer_name] and cached_svds[layer_name]['efficient_pinvs'] is not None:
                    # Use the precomputed efficient pseudoinverse
                    if rcond in cached_svds[layer_name]['efficient_pinvs']:
                        # Get the optimized pinv for this rcond value
                        X_weighted = cached_svds[layer_name]['X_weighted']
                        Y_weighted = cached_svds[layer_name]['Y_weighted']
                        pinv_A = cached_svds[layer_name]['efficient_pinvs'][rcond]
                        
                        # Calculate beta using precomputed pseudoinverse
                        beta_std = pinv_A @ (X_weighted.T @ Y_weighted)
                        
                        # Convert coefficients back to original scale
                        beta = unstandardize_coefficients(beta_std, mean, std)
                        
                        # Make predictions and calculate metrics
                        ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                        X_orig_bias = torch.cat([ones_orig, X], dim=1)
                        Y_pred = X_orig_bias @ beta
                        
                        # Calculate weighted error
                        distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                        weighted_distances = distances * weights_tensor
                        norm_dist = weighted_distances.sum().item()
                        
                        # Calculate R-squared
                        sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                        total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                        explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                        r_squared = (explained_var / total_var).item()
                    else:
                        # If this rcond isn't in the precomputed pinvs, fall back to standard method
                        logger.warning(f"rcond {rcond} not found in precomputed efficient_pinvs, falling back to standard pinv")
                        # Use standard method (below)
                        use_efficient = False
                else:
                    # Use standard pinv method
                    ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                    X_std_bias = torch.cat([ones, X_std], dim=1)
                    sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                    X_weighted = X_std_bias * sqrt_weights
                    Y_weighted = Y * sqrt_weights
                    
                    # Calculate beta using pseudoinverse with specified regularization
                    A = X_weighted.T @ X_weighted
                    beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)
                    
                    # Convert coefficients back to original scale
                    beta = unstandardize_coefficients(beta_std, mean, std)
                    
                    # Make predictions and calculate metrics
                    ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                    X_orig_bias = torch.cat([ones_orig, X], dim=1)
                    Y_pred = X_orig_bias @ beta
                    
                    # Calculate weighted error
                    distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                    weighted_distances = distances * weights_tensor
                    norm_dist = weighted_distances.sum().item()
                    
                    # Calculate R-squared
                    total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                    explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                    r_squared = (explained_var / total_var).item()
                
            else:
                # Compute everything from scratch
                norm_dist, dims, var_expl, singular_vals, beta, r_squared = self.process_single_layer(
                    act_tensor, belief_states, nn_word_probs, rcond
                )
            
            # Store the weights for EVERY layer, not just the best one
            weights[layer_name] = beta
            
            records.append({
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "norm_dist": norm_dist,
                "dims": dims,
                "variance_explained": var_expl,
                "r_squared": r_squared
            })
            
            best_layers[layer_name] = norm_dist
            singular_values_dict[layer_name] = singular_vals.tolist() if hasattr(singular_vals, 'tolist') else None
            
            # Track best layer index
            if layer_idx == 0 or norm_dist < records[best_idx]["norm_dist"]:
                best_idx = len(records) - 1
        
        # Process concatenated activations - without input/embeddings
        non_input_layers = {}
        for k, v in acts.items():
            if not any(x in k.lower() for x in ['pre', 'embedding', 'input']):
                non_input_layers[k] = v
        logger.debug(f"Non-input layers: {list(non_input_layers.keys())}")
        
        if non_input_layers:
            layer_name = "concat_no_input"
            concat_acts = torch.cat([non_input_layers[k] for k in non_input_layers], dim=-1)
            
            # Use cached SVD if available - same logic as above
            if cached_svds and layer_name in cached_svds:
                X = cached_svds[layer_name]['X']
                X_std = cached_svds[layer_name]['X_std']
                Y = cached_svds[layer_name]['Y'] 
                weights_tensor = cached_svds[layer_name]['weights']
                singular_vals = cached_svds[layer_name]['singular_values']
                var_expl = cached_svds[layer_name]['var_expl']
                mean = cached_svds[layer_name]['mean']
                std = cached_svds[layer_name]['std']
                dims = X.shape[1]
                
                # Check if we have precomputed efficient pinvs
                if self.use_efficient_pinv and 'efficient_pinvs' in cached_svds[layer_name] and cached_svds[layer_name]['efficient_pinvs'] is not None:
                    # Use the precomputed efficient pseudoinverse
                    if rcond in cached_svds[layer_name]['efficient_pinvs']:
                        # Get the optimized pinv for this rcond value
                        X_weighted = cached_svds[layer_name]['X_weighted']
                        Y_weighted = cached_svds[layer_name]['Y_weighted']
                        pinv_A = cached_svds[layer_name]['efficient_pinvs'][rcond]
                        
                        # Calculate beta using precomputed pseudoinverse
                        beta_std = pinv_A @ (X_weighted.T @ Y_weighted)
                        
                        # Convert coefficients back to original scale
                        beta = unstandardize_coefficients(beta_std, mean, std)
                        
                        # Make predictions and calculate metrics
                        ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                        X_orig_bias = torch.cat([ones_orig, X], dim=1)
                        Y_pred = X_orig_bias @ beta
                        
                        # Calculate weighted error
                        distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                        weighted_distances = distances * weights_tensor
                        norm_dist = weighted_distances.sum().item()
                        
                        # Calculate R-squared
                        sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                        total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                        explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                        r_squared = (explained_var / total_var).item()
                    else:
                        # If this rcond isn't in the precomputed pinvs, fall back to standard method
                        logger.warning(f"rcond {rcond} not found in precomputed efficient_pinvs, falling back to standard pinv")
                        # Original method below
                        ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                        X_std_bias = torch.cat([ones, X_std], dim=1)
                        sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                        X_weighted = X_std_bias * sqrt_weights
                        Y_weighted = Y * sqrt_weights
                        
                        A = X_weighted.T @ X_weighted
                        beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)
                        
                        beta = unstandardize_coefficients(beta_std, mean, std)
                        
                        ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                        X_orig_bias = torch.cat([ones_orig, X], dim=1)
                        Y_pred = X_orig_bias @ beta
                        
                        distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                        weighted_distances = distances * weights_tensor
                        norm_dist = weighted_distances.sum().item()
                        
                        total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                        explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                        r_squared = (explained_var / total_var).item()
                else:
                    # Use standard method
                    ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                    X_std_bias = torch.cat([ones, X_std], dim=1)
                    sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                    X_weighted = X_std_bias * sqrt_weights
                    Y_weighted = Y * sqrt_weights
                    
                    A = X_weighted.T @ X_weighted
                    beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)
                    
                    beta = unstandardize_coefficients(beta_std, mean, std)
                    
                    ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                    X_orig_bias = torch.cat([ones_orig, X], dim=1)
                    Y_pred = X_orig_bias @ beta
                    
                    distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                    weighted_distances = distances * weights_tensor
                    norm_dist = weighted_distances.sum().item()
                    
                    total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                    explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                    r_squared = (explained_var / total_var).item()
            else:
                norm_dist, dims, var_expl, singular_vals, beta, r_squared = self.process_single_layer(
                    concat_acts, belief_states, nn_word_probs, rcond
                )
            
            # Store weights for this layer too
            weights[layer_name] = beta
            
            records.append({
                "layer_name": layer_name,
                "layer_idx": layer_idx + 1,
                "norm_dist": norm_dist,
                "dims": dims,
                "variance_explained": var_expl,
                "r_squared": r_squared
            })
            
            best_layers[layer_name] = norm_dist
            singular_values_dict[layer_name] = singular_vals.tolist() if hasattr(singular_vals, 'tolist') else None
            
            if norm_dist < records[best_idx]["norm_dist"]:
                best_idx = len(records) - 1
        
        # Process all concatenated activations - with input/embeddings
        layer_name = "concat_all"
        concat_acts = torch.cat([acts[k] for k in acts], dim=-1)
        
        # Use cached SVD if available
        if cached_svds and layer_name in cached_svds:
            X = cached_svds[layer_name]['X']
            X_std = cached_svds[layer_name]['X_std']
            Y = cached_svds[layer_name]['Y'] 
            weights_tensor = cached_svds[layer_name]['weights']
            singular_vals = cached_svds[layer_name]['singular_values']
            var_expl = cached_svds[layer_name]['var_expl']
            mean = cached_svds[layer_name]['mean']
            std = cached_svds[layer_name]['std']
            dims = X.shape[1]
            
            # Check if we have precomputed efficient pinvs
            if self.use_efficient_pinv and 'efficient_pinvs' in cached_svds[layer_name] and cached_svds[layer_name]['efficient_pinvs'] is not None:
                # Use the precomputed efficient pseudoinverse
                if rcond in cached_svds[layer_name]['efficient_pinvs']:
                    # Get the optimized pinv for this rcond value
                    X_weighted = cached_svds[layer_name]['X_weighted']
                    Y_weighted = cached_svds[layer_name]['Y_weighted']
                    pinv_A = cached_svds[layer_name]['efficient_pinvs'][rcond]
                    
                    # Calculate beta using precomputed pseudoinverse
                    beta_std = pinv_A @ (X_weighted.T @ Y_weighted)
                    
                    # Convert coefficients back to original scale
                    beta = unstandardize_coefficients(beta_std, mean, std)
                    
                    # Make predictions and calculate metrics
                    ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                    X_orig_bias = torch.cat([ones_orig, X], dim=1)
                    Y_pred = X_orig_bias @ beta
                    
                    # Calculate weighted error
                    distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                    weighted_distances = distances * weights_tensor
                    norm_dist = weighted_distances.sum().item()
                    
                    # Calculate R-squared
                    sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                    total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                    explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                    r_squared = (explained_var / total_var).item()
                else:
                    # If this rcond isn't in the precomputed pinvs, fall back to standard method
                    logger.warning(f"rcond {rcond} not found in precomputed efficient_pinvs, falling back to standard pinv")
                    # Original method
                    ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                    X_std_bias = torch.cat([ones, X_std], dim=1)
                    sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                    X_weighted = X_std_bias * sqrt_weights
                    Y_weighted = Y * sqrt_weights
                    
                    A = X_weighted.T @ X_weighted
                    beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)
                    
                    beta = unstandardize_coefficients(beta_std, mean, std)
                    
                    ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                    X_orig_bias = torch.cat([ones_orig, X], dim=1)
                    Y_pred = X_orig_bias @ beta
                    
                    distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                    weighted_distances = distances * weights_tensor
                    norm_dist = weighted_distances.sum().item()
                    
                    total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                    explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                    r_squared = (explained_var / total_var).item()
            else:
                # Standard method
                ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                X_std_bias = torch.cat([ones, X_std], dim=1)
                sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                X_weighted = X_std_bias * sqrt_weights
                Y_weighted = Y * sqrt_weights
                
                A = X_weighted.T @ X_weighted
                beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)
                
                beta = unstandardize_coefficients(beta_std, mean, std)
                
                ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                X_orig_bias = torch.cat([ones_orig, X], dim=1)
                Y_pred = X_orig_bias @ beta
                
                distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                weighted_distances = distances * weights_tensor
                norm_dist = weighted_distances.sum().item()
                
                total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                r_squared = (explained_var / total_var).item()
        else:
            norm_dist, dims, var_expl, singular_vals, beta, r_squared = self.process_single_layer(
                concat_acts, belief_states, nn_word_probs, rcond
            )
        
        # Store weights for this layer too
        weights[layer_name] = beta
        
        records.append({
            "layer_name": layer_name,
            "layer_idx": layer_idx + 2 if 'layer_idx' in locals() else 1,
            "norm_dist": norm_dist,
            "dims": dims,
            "variance_explained": var_expl,
            "r_squared": r_squared
        })
        
        best_layers[layer_name] = norm_dist
        singular_values_dict[layer_name] = singular_vals.tolist() if hasattr(singular_vals, 'tolist') else None
        
        if norm_dist < records[best_idx]["norm_dist"]:
            best_idx = len(records) - 1
        
        best_layer = records[best_idx]["layer_name"]
        return pd.DataFrame(records), singular_values_dict, weights, best_layer, best_layers

    def analyze_checkpoint_with_activations(self, target_activations, belief_targets, rcond_sweep_list, compare_implementations=False):
        """
        Analyze activations across all targets and regularization parameters.
        
        Args:
            target_activations: Dictionary mapping target names to their corresponding activation dictionaries
            belief_targets: Dictionary of belief targets to analyze
            rcond_sweep_list: List of regularization parameters to sweep
            compare_implementations: If True, run both original and optimized implementations and compare
            
        Returns:
            tuple: (list of DataFrames, best weights dictionary, best singular values dictionary)
        """
        all_results = []
        best_weights_dict = {}
        best_singular_dict = {}
        
        for target_name, target_data in belief_targets.items():
            if target_name not in target_activations:
                logger.warning(f"No activations found for target {target_name}, skipping")
                continue
                
            target = target_data['beliefs']
            probs = target_data['probs']
            acts = target_activations[target_name]
            
            target_best_weights = {}
            target_best_singular = {}
            
            # Log shapes for debugging
            logger.debug(f"Target {target_name} - Beliefs shape: {target.shape}, Probs shape: {probs.shape}")
            for layer_name, layer_acts in acts.items():
                logger.debug(f"  {layer_name} activation shape: {layer_acts.shape}")
            
            # Precompute SVD and other expensive operations for each layer
            cached_svds = {}
            logger.info("Precomputing SVD for each layer...")
            
            for layer_name, act_tensor in acts.items():
                try:
                    # Reshape tensors for regression
                    X = act_tensor.view(-1, act_tensor.shape[-1]).to(self.device)
                    Y = target.view(-1, target.shape[-1]).to(self.device)
                    weights_tensor = probs.view(-1).to(self.device)
                    weights_tensor = weights_tensor / weights_tensor.sum()
                    
                    # Standardize features
                    X_std, mean, std = standardize(X)
                    
                    # SVD for variance analysis
                    var_expl_str, singular_values = report_variance_explained(X_std, REPORT_VARIANCE)
                    
                    # Add bias term 
                    ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                    X_std_bias = torch.cat([ones, X_std], dim=1)
                    
                    # Apply weights
                    sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                    X_weighted = X_std_bias * sqrt_weights
                    Y_weighted = Y * sqrt_weights
                    
                    # Compute weighted design matrix
                    A = X_weighted.T @ X_weighted
                    
                    # If using efficient implementation or comparing, precompute all pinvs
                    if self.use_efficient_pinv or compare_implementations:
                        # Pre-compute all pseudoinverses in one go using optimized method
                        pinv_matrices, _ = compute_efficient_pinv_from_svd(A, rcond_sweep_list)
                        
                        # Store for use in regression
                        efficient_pinvs = pinv_matrices
                    else:
                        efficient_pinvs = None
                    
                    # Cache computations
                    cached_svds[layer_name] = {
                        'X': X,
                        'X_std': X_std,
                        'Y': Y,
                        'weights': weights_tensor,
                        'mean': mean,
                        'std': std,
                        'singular_values': singular_values,
                        'var_expl': var_expl_str,
                        'X_weighted': X_weighted,
                        'Y_weighted': Y_weighted,
                        'efficient_pinvs': efficient_pinvs
                    }
                    
                    # Store singular values for later use - they don't depend on rcond
                    if layer_name not in target_best_singular:
                        target_best_singular[layer_name] = []
                    
                    if singular_values is not None:
                        target_best_singular[layer_name].append({
                            "singular_values": singular_values.tolist() if hasattr(singular_values, 'tolist') else singular_values
                        })
                except Exception as e:
                    logger.warning(f"Error precomputing SVD for layer {layer_name}: {e}")
                    # We'll fall back to non-cached computation for this layer
            
            # Now do the same for concatenated layers
            # Process concatenated activations - without input/embeddings
            non_input_layers = {}
            for k, v in acts.items():
                if not any(x in k.lower() for x in ['pre', 'embedding', 'input']):
                    non_input_layers[k] = v
                    
            if non_input_layers:
                try:
                    concat_acts = torch.cat([non_input_layers[k] for k in non_input_layers], dim=-1)
                    # Precompute SVD for concat_no_input
                    X = concat_acts.view(-1, concat_acts.shape[-1]).to(self.device)
                    Y = target.view(-1, target.shape[-1]).to(self.device)
                    weights_tensor = probs.view(-1).to(self.device)
                    weights_tensor = weights_tensor / weights_tensor.sum()
                    
                    X_std, mean, std = standardize(X)
                    var_expl_str, singular_values = report_variance_explained(X_std, REPORT_VARIANCE)
                    
                    # Add bias term 
                    ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                    X_std_bias = torch.cat([ones, X_std], dim=1)
                    
                    # Apply weights
                    sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                    X_weighted = X_std_bias * sqrt_weights
                    Y_weighted = Y * sqrt_weights
                    
                    # Compute weighted design matrix
                    A = X_weighted.T @ X_weighted
                    
                    # If using efficient implementation or comparing, precompute all pinvs
                    if self.use_efficient_pinv or compare_implementations:
                        # Pre-compute all pseudoinverses in one go
                        pinv_matrices, _ = compute_efficient_pinv_from_svd(A, rcond_sweep_list)
                        efficient_pinvs = pinv_matrices
                    else:
                        efficient_pinvs = None
                    
                    cached_svds["concat_no_input"] = {
                        'X': X,
                        'X_std': X_std,
                        'Y': Y,
                        'weights': weights_tensor,
                        'mean': mean,
                        'std': std,
                        'singular_values': singular_values,
                        'var_expl': var_expl_str,
                        'X_weighted': X_weighted,
                        'Y_weighted': Y_weighted,
                        'efficient_pinvs': efficient_pinvs
                    }
                except Exception as e:
                    logger.warning(f"Error precomputing SVD for concat_no_input: {e}")
            
            # Similarly for concat_all
            try:
                concat_acts = torch.cat([acts[k] for k in acts], dim=-1)
                # Precompute SVD for concat_all
                X = concat_acts.view(-1, concat_acts.shape[-1]).to(self.device)
                Y = target.view(-1, target.shape[-1]).to(self.device)
                weights_tensor = probs.view(-1).to(self.device)
                weights_tensor = weights_tensor / weights_tensor.sum()
                
                X_std, mean, std = standardize(X)
                var_expl_str, singular_values = report_variance_explained(X_std, REPORT_VARIANCE)
                
                # Add bias term 
                ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                X_std_bias = torch.cat([ones, X_std], dim=1)
                
                # Apply weights
                sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                X_weighted = X_std_bias * sqrt_weights
                Y_weighted = Y * sqrt_weights
                
                # Compute weighted design matrix
                A = X_weighted.T @ X_weighted
                
                # If using efficient implementation or comparing, precompute all pinvs
                if self.use_efficient_pinv or compare_implementations:
                    # Pre-compute all pseudoinverses in one go
                    pinv_matrices, _ = compute_efficient_pinv_from_svd(A, rcond_sweep_list)
                    efficient_pinvs = pinv_matrices
                else:
                    efficient_pinvs = None
                
                cached_svds["concat_all"] = {
                    'X': X,
                    'X_std': X_std,
                    'Y': Y,
                    'weights': weights_tensor,
                    'mean': mean,
                    'std': std,
                    'singular_values': singular_values,
                    'var_expl': var_expl_str,
                    'X_weighted': X_weighted,
                    'Y_weighted': Y_weighted,
                    'efficient_pinvs': efficient_pinvs
                }
            except Exception as e:
                logger.warning(f"Error precomputing SVD for concat_all: {e}")
            
            # Sweep across regularization parameters
            for rcond_val in tqdm(rcond_sweep_list, desc=f"rcond sweep for target {target_name}", leave=False):
                # Create a copy of the cached_svds for this run
                cached_svds_copy = {k: v.copy() for k, v in cached_svds.items()}
                
                # Use the original implementation
                original_use_efficient = self.use_efficient_pinv
                
                # If we're comparing, first run with the original implementation
                if compare_implementations:
                    self.use_efficient_pinv = False
                    
                    # Run original implementation
                    df_orig, _, weights_orig, best_layer_orig, best_layers_orig = self.process_activation_layers(
                        acts, target, probs, rcond_val, cached_svds=cached_svds_copy
                    )
                    
                    # Now run with efficient implementation
                    self.use_efficient_pinv = True
                    df_eff, _, weights_eff, best_layer_eff, best_layers_eff = self.process_activation_layers(
                        acts, target, probs, rcond_val, cached_svds=cached_svds
                    )
                    
                    # Compare the results
                    for layer in best_layers_orig:
                        if layer in best_layers_eff:
                            diff = abs(best_layers_orig[layer] - best_layers_eff[layer])
                            rel_diff = diff / best_layers_orig[layer] if best_layers_orig[layer] > 0 else 0
                            logger.info(f"Layer {layer} - Original: {best_layers_orig[layer]:.6f}, "
                                      f"Efficient: {best_layers_eff[layer]:.6f}, "
                                      f"Diff: {diff:.6f}, Rel Diff: {rel_diff:.6f}")
                    
                    # Reset to original setting
                    self.use_efficient_pinv = original_use_efficient
                    
                    # Use the efficient results for the rest of the processing
                    df = df_eff
                    weights = weights_eff
                    best_layers = best_layers_eff
                else:
                    # Just use the current setting
                    df, _, weights, _, best_layers = self.process_activation_layers(
                        acts, target, probs, rcond_val, cached_svds=cached_svds
                    )
                
                df['target'] = target_name
                df['rcond'] = rcond_val
                all_results.append(df)
                
                # Track best weights
                for layer, dist in best_layers.items():
                    if layer not in target_best_weights or dist < target_best_weights[layer]["dist"]:
                        target_best_weights[layer] = {
                            "weights": weights.get(layer, None),
                            "rcond": rcond_val,
                            "dist": dist
                        }
            
            best_weights_dict[target_name] = target_best_weights
            
            # Ensure we have singular values for this target
            if not target_best_singular:
                logger.warning(f"No singular values found for target {target_name}")
                # Create empty placeholder to prevent errors
                target_best_singular = {layer: [] for layer in acts.keys()}
            
            # Log how many singular value entries we're storing
            total_entries = sum(len(entries) for entries in target_best_singular.values())
            logger.info(f"Storing {total_entries} singular value entries for target {target_name}")
            
            best_singular_dict[target_name] = target_best_singular
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
            
        return combined_df, best_weights_dict, best_singular_dict

    def run_random_baseline(self, random_acts, belief_targets, rcond_val):
        """Run regression on random baseline activations."""
        if not DO_BASELINE:
            return [], {}, {}
            
        all_results = []
        best_singular_dict = {}
        best_weights_dict = {}
        
        for random_idx, acts in enumerate(tqdm(random_acts, desc="Random baselines")):
            for target_name, target_data in belief_targets.items():
                target = target_data['beliefs']
                probs = target_data['probs']
                
                df, singular_values, weights, _, best_layers = self.process_activation_layers(
                    acts, target, probs, rcond_val
                )
                
                df['checkpoint'] = f'RANDOM_{random_idx}'
                df['target'] = target_name
                df['rcond'] = rcond_val
                all_results.append(df)
                
                # Store singular values for first few random models
                if random_idx < 10:
                    if target_name not in best_singular_dict:
                        best_singular_dict[target_name] = {}
                    
                    for layer, sv in singular_values.items():
                        if layer not in best_singular_dict[target_name]:
                            best_singular_dict[target_name][layer] = []
                        
                        best_singular_dict[target_name][layer].append({
                            "random_idx": random_idx,
                            "singular_values": sv
                        })
                    
                    # Store best weights for this random model
                    if target_name not in best_weights_dict:
                        best_weights_dict[target_name] = {}
                    
                    for layer, dist in best_layers.items():
                        layer_key = f"{layer}_random_{random_idx}"
                        
                        if layer_key not in best_weights_dict[target_name]:
                            best_weights_dict[target_name][layer_key] = {
                                "weights": weights.get(layer, None),
                                "rcond": rcond_val,
                                "dist": dist,
                                "random_idx": random_idx
                            }
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
            
        return combined_df, best_singular_dict, best_weights_dict
    
    def run_random_baseline_streaming(self, random_act_generator, belief_targets, rcond_val, all_results=None, best_singular_dict=None, best_weights_dict=None):
        """Process a single random baseline at a time.
        
        This is a memory-efficient version that processes one random model at a time
        and appends results to the provided collections or creates new ones.
        
        Args:
            random_act_generator: Generator yielding (random_idx, activations) tuples
            belief_targets: Dictionary of belief targets
            rcond_val: Regularization parameter
            all_results: Optional list to accumulate DataFrames (created if None)
            best_singular_dict: Optional dict for singular values (created if None)
            best_weights_dict: Optional dict for weights (created if None)
            
        Returns:
            tuple: (all_results, best_singular_dict, best_weights_dict)
        """
        if not DO_BASELINE:
            return [], {}, {}
            
        # Initialize accumulators if not provided
        all_results = all_results or []
        best_singular_dict = best_singular_dict or {}
        best_weights_dict = best_weights_dict or {}
        
        # Process the current random network
        random_idx, acts = next(random_act_generator, (None, None))
        if random_idx is None:
            return all_results, best_singular_dict, best_weights_dict
            
        for target_name, target_data in belief_targets.items():
            target = target_data['beliefs']
            probs = target_data['probs']
            
            df, singular_values, weights, _, best_layers = self.process_activation_layers(
                acts, target, probs, rcond_val
            )
            
            df['checkpoint'] = f'RANDOM_{random_idx}'
            df['target'] = target_name
            df['rcond'] = rcond_val
            all_results.append(df)
            
            # Store singular values for first few random models
            if random_idx < 10:
                if target_name not in best_singular_dict:
                    best_singular_dict[target_name] = {}
                
                for layer, sv in singular_values.items():
                    if layer not in best_singular_dict[target_name]:
                        best_singular_dict[target_name][layer] = []
                    
                    best_singular_dict[target_name][layer].append({
                        "random_idx": random_idx,
                        "singular_values": sv
                    })
                
                # Store best weights for this random model
                if target_name not in best_weights_dict:
                    best_weights_dict[target_name] = {}
                
                for layer, dist in best_layers.items():
                    layer_key = f"{layer}_random_{random_idx}"
                    
                    if layer_key not in best_weights_dict[target_name]:
                        best_weights_dict[target_name][layer_key] = {
                            "weights": weights.get(layer, None),
                            "rcond": rcond_val,
                            "dist": dist,
                            "random_idx": random_idx
                        }
        
        return all_results, best_singular_dict, best_weights_dict

def run_single_rcond_sweep(regression_analyzer, activation, belief_states, nn_word_probs, rcond_values):
    """Run an rcond sweep on a single activation and target."""
    results = []
    
    # Pre-compute SVD (this is the expensive part)
    from scripts.activation_analysis.regression import compute_efficient_pinv_from_svd, standardize
    
    # Reshape tensors
    X = activation.view(-1, activation.shape[-1])
    Y = belief_states.view(-1, belief_states.shape[-1])
    weights = nn_word_probs.view(-1)
    
    # Standardize features
    X_std, mean, std = standardize(X)
    
    # Add bias terms and prepare weighted regression
    ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
    X_std_bias = torch.cat([ones, X_std], dim=1)
    sqrt_weights = torch.sqrt(weights).unsqueeze(1)
    X_weighted = X_std_bias * sqrt_weights
    Y_weighted = Y * sqrt_weights
    
    # Compute efficient pseudoinverses for all rcond values
    A = X_weighted.T @ X_weighted
    pinvs_dict, singular_values = compute_efficient_pinv_from_svd(A, rcond_values)
    
    # Run regression for each rcond value
    for rcond in rcond_values:
        pinv_A = pinvs_dict[rcond]
        beta_std = pinv_A @ (X_weighted.T @ Y_weighted)
        
        # Get results and store
        # [add code here to calculate and store metrics]
        results.append({"rcond": rcond, "beta": beta_std, "singular_values": singular_values})
    
    return results

def run_single_rcond_sweep_with_predictions_flat(regression_analyzer, activation, belief_states, nn_word_probs, rcond_values):
    """
    Run an rcond sweep on a single activation and target, returning the best result with predictions.
    
    Args:
        regression_analyzer: Instance of RegressionAnalyzer
        activation: Tensor of activations
        belief_states: Target belief states
        nn_word_probs: Word probabilities for weighting
        rcond_values: List of regularization parameters to try
        
    Returns:
        dict: The best result (with lowest norm_dist) containing:
            - rcond: The best regularization parameter
            - norm_dist: The weighted error metric
            - beta: Regression coefficients
            - r_squared: R-squared value
            - singular_values: Singular values from SVD
            - predictions: The predicted belief states using the best model
            - true_values: The actual belief states for comparison
    """
    best_result = None
    best_norm_dist = float('inf')
    
    # Pre-compute SVD (this is the expensive part)
    from scripts.activation_analysis.regression import compute_efficient_pinv_from_svd
    
    # Reshape tensors
    X = activation  
    Y = belief_states   
    weights = nn_word_probs

    print(X.shape, Y.shape, weights.shape)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Add bias terms and prepare weighted regression
    ones = torch.ones(X.shape[0], 1, device=X.device)
    X_bias = torch.cat([ones, X], dim=1)
    sqrt_weights = torch.sqrt(weights).unsqueeze(1)
    X_weighted = X_bias * sqrt_weights
    Y_weighted = Y * sqrt_weights
    
    # Compute efficient pseudoinverses for all rcond values
    A = X_weighted.T @ X_weighted
    pinvs_dict, singular_values = compute_efficient_pinv_from_svd(A, rcond_values)
    
    # Variables to store the best predictions
    best_predictions = None
    
    # Run regression for each rcond value
    for rcond in rcond_values:
        pinv_A = pinvs_dict[rcond]
        beta = pinv_A @ (X_weighted.T @ Y_weighted)
        
        # Make predictions
        Y_pred = X_bias @ beta
        
        # Calculate weighted error (norm_dist)
        distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
        weighted_distances = distances * weights
        norm_dist = weighted_distances.sum().item()
        
        # Calculate R-squared
        total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
        explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
        r_squared = (explained_var / total_var).item()
        
        # Create result dictionary
        result = {
            "rcond": rcond,
            "norm_dist": norm_dist,
            "beta": beta.cpu().detach().numpy(),
            "r_squared": r_squared,
            "singular_values": singular_values.cpu().detach().numpy()
        }
        
        # Check if this is the best result so far
        if norm_dist < best_norm_dist:
            best_norm_dist = norm_dist
            best_result = result
            best_predictions = Y_pred

    # Reshape predictions back to original shape
    if best_predictions is not None:
        reshaped_predictions = best_predictions.view(belief_states.shape)
        
        # Add predictions to the best result
        best_result["predictions"] = reshaped_predictions.cpu().detach().numpy()
        best_result["true_values"] = belief_states.cpu().detach().numpy()
        best_result["weights"] = weights.cpu().detach().numpy()
    
    return best_result

def run_single_rcond_sweep_with_predictions(regression_analyzer, activation, belief_states, nn_word_probs, rcond_values):
    """
    Run an rcond sweep on a single activation and target, returning the best result with predictions.
    
    Args:
        regression_analyzer: Instance of RegressionAnalyzer
        activation: Tensor of activations
        belief_states: Target belief states
        nn_word_probs: Word probabilities for weighting
        rcond_values: List of regularization parameters to try
        
    Returns:
        dict: The best result (with lowest norm_dist) containing:
            - rcond: The best regularization parameter
            - norm_dist: The weighted error metric
            - beta: Regression coefficients
            - r_squared: R-squared value
            - singular_values: Singular values from SVD
            - predictions: The predicted belief states using the best model
            - true_values: The actual belief states for comparison
    """
    best_result = None
    best_norm_dist = float('inf')
    
    # Pre-compute SVD (this is the expensive part)
    from scripts.activation_analysis.regression import compute_efficient_pinv_from_svd
    
    # Reshape tensors
    X = activation.view(-1, activation.shape[-1])
    Y = belief_states.view(-1, belief_states.shape[-1])
    weights = nn_word_probs.view(-1)
    
    # Save original shapes for reshaping predictions later
    original_shape = belief_states.shape
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Add bias terms and prepare weighted regression
    ones = torch.ones(X.shape[0], 1, device=X.device)
    X_bias = torch.cat([ones, X], dim=1)
    sqrt_weights = torch.sqrt(weights).unsqueeze(1)
    X_weighted = X_bias * sqrt_weights
    Y_weighted = Y * sqrt_weights
    
    # Compute efficient pseudoinverses for all rcond values
    A = X_weighted.T @ X_weighted
    pinvs_dict, singular_values = compute_efficient_pinv_from_svd(A, rcond_values)
    
    # Variables to store the best predictions
    best_predictions = None
    
    # Run regression for each rcond value
    for rcond in rcond_values:
        pinv_A = pinvs_dict[rcond]
        beta = pinv_A @ (X_weighted.T @ Y_weighted)
        
        # Make predictions
        Y_pred = X_bias @ beta
        
        # Calculate weighted error (norm_dist)
        distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
        weighted_distances = distances * weights
        norm_dist = weighted_distances.sum().item()
        
        # Calculate R-squared
        total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
        explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
        r_squared = (explained_var / total_var).item()
        
        # Create result dictionary
        result = {
            "rcond": rcond,
            "norm_dist": norm_dist,
            "beta": beta.cpu().detach().numpy(),
            "r_squared": r_squared,
            "singular_values": singular_values.cpu().detach().numpy()
        }
        
        # Check if this is the best result so far
        if norm_dist < best_norm_dist:
            best_norm_dist = norm_dist
            best_result = result
            best_predictions = Y_pred
    
    # Reshape predictions back to original shape
    if best_predictions is not None:
        reshaped_predictions = best_predictions.view(original_shape)
        
        # Add predictions to the best result
        best_result["predictions"] = reshaped_predictions.cpu().detach().numpy()
        best_result["true_values"] = belief_states.cpu().detach().numpy()
    
    return best_result



def run_paul_rcond_sweep_with_sklearn_predictions_flat(regression_analyzer, activation, belief_states, nn_word_probs, rcond_values):
    """
    Run an rcond sweep on a single activation and target, returning the best result with predictions.
    
    Args:
        regression_analyzer: Instance of RegressionAnalyzer
        activation: Tensor of activations
        belief_states: Target belief states
        nn_word_probs: Word probabilities for weighting
        rcond_values: List of regularization parameters to try
        
    Returns:
        dict: A dictionary with best results for three approaches:
            - old_method: Using pseudoinverse of (X^T X) from compute_efficient_pinv_from_svd
            - new_method: Using pseudoinverse of X directly via compute_efficient_pinv_from_svd
            - sklearn_method: Using sklearn's LinearRegression (no regularization)
    """
    best_result_old = None
    best_norm_dist_old = float('inf')
    
    best_result_new = None
    best_norm_dist_new = float('inf')
    
    # Import our helper function (used for both old and new methods)
    from scripts.activation_analysis.regression import compute_efficient_pinv_from_svd
    
    # Reshape tensors (assuming they're already flattened as needed)
    X = activation   # shape: (n, d)
    Y = belief_states   # shape: (n, d_b)
    weights = nn_word_probs  # shape: (n,)
    
    print(X.shape, Y.shape, weights.shape)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Add bias terms and prepare weighted regression
    ones = torch.ones(X.shape[0], 1, device=X.device)
    X_bias = torch.cat([ones, X], dim=1)  # X with bias column
    sqrt_weights = torch.sqrt(weights).unsqueeze(1)
    X_weighted = X_bias * sqrt_weights     # weighted design matrix (for our pseudoinverse methods)
    Y_weighted = Y * sqrt_weights          # weighted targets
    
    # === OLD METHOD: Using (X_weighted^T X_weighted)^+ X_weighted^T Y_weighted ===
    # Compute A = X_weighted.T @ X_weighted and its pseudoinverses for all rcond values
    A = X_weighted.T @ X_weighted
    pinvs_dict_old, singular_values_old = compute_efficient_pinv_from_svd(A, rcond_values)
    
    # === NEW METHOD: Using direct pseudoinverse of X_weighted via compute_efficient_pinv_from_svd ===
    # Here we call our helper with X_weighted directly (note: no bias is added here because it's already in X_bias)
    pinvs_dict_new, singular_values_new = compute_efficient_pinv_from_svd(X_weighted, rcond_values)  # # NEW
    
    # Variables to store the best predictions for old and new methods
    best_predictions_old = None
    best_predictions_new = None
    
    # Run regression for each rcond value (old and new methods)
    for rcond in rcond_values:
        # --- Old Method ---
        pinv_A_old = pinvs_dict_old[rcond]
        beta_old = pinv_A_old @ (X_weighted.T @ Y_weighted)
        
        # Make predictions for old method
        Y_pred_old = X_bias @ beta_old
        
        # Calculate weighted error (norm_dist) for old method
        distances_old = torch.sqrt(torch.sum((Y_pred_old - Y)**2, dim=1))
        weighted_distances_old = distances_old * weights
        norm_dist_old = weighted_distances_old.sum().item()
        
        # Calculate R-squared for old method
        total_var_old = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
        residual_var_old = torch.sum((Y_pred_old * sqrt_weights - Y_weighted)**2)
        r_squared_old = (1.0 - (residual_var_old / total_var_old)).item() if total_var_old > 0 else 0
        
        # --- New Method ---
        pinv_A_new = pinvs_dict_new[rcond]  # pseudoinverse computed on X_weighted directly
        beta_new = pinv_A_new @ Y_weighted   # note: no extra X_weighted.T multiplication here
        
        # Make predictions for new method
        Y_pred_new = X_bias @ beta_new
        
        # Calculate weighted error (norm_dist) for new method
        distances_new = torch.sqrt(torch.sum((Y_pred_new - Y)**2, dim=1))
        weighted_distances_new = distances_new * weights
        norm_dist_new = weighted_distances_new.sum().item()
        
        # Calculate R-squared for new method
        total_var_new = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
        explained_var_new = torch.sum((Y_pred_new * sqrt_weights - Y_weighted.mean(dim=0))**2)
        r_squared_new = (explained_var_new / total_var_new).item()
        
        # Create result dictionaries for both methods for current rcond
        result_old = {
            "rcond": rcond,
            "norm_dist": norm_dist_old,
            "beta": beta_old.cpu().detach().numpy(),
            "r_squared": r_squared_old,
            "singular_values": singular_values_old.cpu().detach().numpy()
        }
        
        result_new = {
            "rcond": rcond,
            "norm_dist": norm_dist_new,
            "beta": beta_new.cpu().detach().numpy(),
            "r_squared": r_squared_new,
            "singular_values": singular_values_new.cpu().detach().numpy()
        }
        
        # Update best result for old method
        if norm_dist_old < best_norm_dist_old:
            best_norm_dist_old = norm_dist_old
            best_result_old = result_old
            best_predictions_old = Y_pred_old
        
        # Update best result for new method
        if norm_dist_new < best_norm_dist_new:
            best_norm_dist_new = norm_dist_new
            best_result_new = result_new
            best_predictions_new = Y_pred_new

    # Reshape predictions back to original shape and add to result dictionaries for old and new methods
    if best_predictions_old is not None:
        reshaped_predictions_old = best_predictions_old.view(belief_states.shape)
        best_result_old["predictions"] = reshaped_predictions_old.cpu().detach().numpy()
        best_result_old["true_values"] = Y.cpu().detach().numpy()
        best_result_old["weights"] = weights.cpu().detach().numpy()
    
    if best_predictions_new is not None:
        reshaped_predictions_new = best_predictions_new.view(belief_states.shape)
        best_result_new["predictions"] = reshaped_predictions_new.cpu().detach().numpy()
        best_result_new["true_values"] = Y.cpu().detach().numpy()
        best_result_new["weights"] = weights.cpu().detach().numpy()
    
    # === SKLEARN METHOD: Using sklearn's LinearRegression (no regularization) ===
    # For sklearn, we use the original X (without bias) and let the model fit the intercept.
    from sklearn.linear_model import LinearRegression  # # NEW
    X_np = X.cpu().detach().numpy()   # use original activation (no bias) because fit_intercept=True
    Y_np = Y.cpu().detach().numpy()
    weights_np = weights.cpu().detach().numpy()
    
    reg = LinearRegression(fit_intercept=True)  # # NEW
    reg.fit(X_np, Y_np, sample_weight=weights_np)  # # NEW
    Y_pred_sklearn_np = reg.predict(X_np)  # # NEW
    Y_pred_sklearn = torch.tensor(Y_pred_sklearn_np, device=X.device)  # # NEW
    
    # Calculate weighted error (norm_dist) for sklearn method
    distances_sklearn = torch.sqrt(torch.sum((Y_pred_sklearn - Y)**2, dim=1))
    weighted_distances_sklearn = distances_sklearn * weights
    norm_dist_sklearn = weighted_distances_sklearn.sum().item()
    
    # Calculate R-squared for sklearn method (using same weighted targets)
    total_var_sklearn = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
    explained_var_sklearn = torch.sum((Y_pred_sklearn * sqrt_weights - Y_weighted.mean(dim=0))**2)
    r_squared_sklearn = (explained_var_sklearn / total_var_sklearn).item()
    
    best_result_sklearn = {  # # NEW
        "norm_dist": norm_dist_sklearn,
        "beta": {"intercept": reg.intercept_, "coef": reg.coef_},  # separate intercept and coefficients
        "r_squared": r_squared_sklearn,
        "predictions": Y_pred_sklearn.cpu().detach().numpy(),
        "true_values": Y.cpu().detach().numpy(),
        "weights": weights.cpu().detach().numpy()
    }
    
    # Return a dictionary containing the best results from all three methods
    return {
        "old_method": best_result_old,
        "new_method": best_result_new,
        "sklearn_method": best_result_sklearn
    }


def run_old_method_train_split_choice(
    regression_analyzer,
    train_activations,   # X_train, shape (N_train, D)
    train_beliefs,       # Y_train, shape (N_train, B)
    train_probs,         # w_train, shape (N_train,)
    test_activations,    # X_test,  shape (N_test, D)
    test_beliefs,        # Y_test,  shape (N_test, B)
    test_probs,          # w_test,  shape (N_test,)
    rcond_values
):
    """
    Sweep over rcond_values using ONLY the 'old method' on a single (train, test) split.
    
    1) Fit on the TRAIN set for each rcond (the "old method"): 
         A = (X_w)^T (X_w),  beta = pinv(A) @ (X_w^T Y_w)
       Where X_w = X_train_bias * sqrt(weights), etc.
    2) Compute the (weighted) TRAIN error for each rcond.
    3) Pick the rcond that yields the lowest TRAIN error.
    4) Return final results including:
         - best_rcond
         - final train error and test error
         - best_weights, final predictions on both train and test
         - etc.

    Args:
        regression_analyzer: your RegressionAnalyzer instance
        train_activations:   torch.Tensor (N_train, D)
        train_beliefs:       torch.Tensor (N_train, B)
        train_probs:         torch.Tensor (N_train,)
        test_activations:    torch.Tensor (N_test, D)
        test_beliefs:        torch.Tensor (N_test, B)
        test_probs:          torch.Tensor (N_test,)
        rcond_values:        list of rcond values to try
    
    Returns:
        dict with keys:
          "best_rcond",
          "train_norm_dist",
          "test_norm_dist",
          "best_weights",
          "train_predictions",
          "test_predictions",
          "train_true_values",
          "test_true_values",
          "train_weights",
          "test_weights"
    """
    import torch
    from scripts.activation_analysis.regression import compute_efficient_pinv_from_svd

    device = train_activations.device

    # -----------------------------
    # 0) Normalize the train & test probabilities
    # -----------------------------
    train_probs = train_probs / train_probs.sum()
    test_probs  = test_probs  / test_probs.sum()

    # Move everything to the same device
    X_train = train_activations.to(device)
    Y_train = train_beliefs.to(device)
    w_train = train_probs.to(device)

    X_test  = test_activations.to(device)
    Y_test  = test_beliefs.to(device)
    w_test  = test_probs.to(device)

    # -----------------------------
    # 1) Construct X with a bias column for the train set
    # -----------------------------
    N_train, D = X_train.shape
    ones_train = torch.ones((N_train, 1), device=device)
    X_train_bias = torch.cat([ones_train, X_train], dim=1)  # (N_train, D+1)

    # Weighted design
    sqrt_w_train = torch.sqrt(w_train).unsqueeze(1)         # (N_train, 1)
    X_train_weighted = X_train_bias * sqrt_w_train          # (N_train, D+1)
    Y_train_weighted = Y_train * sqrt_w_train               # (N_train, B)

    # -----------------------------
    # 2) For the test set (just for final evaluation)
    # -----------------------------
    N_test = X_test.shape[0]
    ones_test = torch.ones((N_test, 1), device=device)
    X_test_bias = torch.cat([ones_test, X_test], dim=1)     # (N_test, D+1)

    # We'll track the best result (lowest TRAIN error)
    best_train_error = float("inf")
    best_rcond = None
    best_beta = None
    best_predictions_train = None

    # Precompute A = X_train_weighted^T @ X_train_weighted once
    A = X_train_weighted.T @ X_train_weighted  # (D+1, D+1)
    # Compute pseudoinverses for all rcond values
    pinvs_dict, singular_vals = compute_efficient_pinv_from_svd(A, rcond_values)

    # -------------------------------------------------------------
    # 2) For each rcond, fit on TRAIN and measure TRAIN error
    # -------------------------------------------------------------
    for rcond in rcond_values:
        pinv_A = pinvs_dict[rcond]  # (D+1, D+1)
        # old method: beta = pinv_A @ (X_train_weighted^T @ Y_train_weighted)
        beta = pinv_A @ (X_train_weighted.T @ Y_train_weighted)  # shape (D+1, B)

        # predictions on TRAIN
        Y_pred_train = X_train_bias @ beta  # (N_train, B)

        # Weighted train error
        dist_train = torch.sqrt(torch.sum((Y_pred_train - Y_train)**2, dim=1))  # (N_train,)
        train_error = (dist_train * w_train).sum().item()

        # If this rcond is better on the TRAIN set, update
        if train_error < best_train_error:
            best_train_error = train_error
            best_rcond = rcond
            best_beta = beta
            best_predictions_train = Y_pred_train

    # -------------------------------------------------------------
    # 3) Evaluate the final chosen model on TRAIN and TEST
    # -------------------------------------------------------------
    train_norm_dist = float("inf")
    test_norm_dist  = float("inf")
    best_predictions_test = None

    if best_beta is not None:
        # Recompute the final train predictions if needed
        train_norm_dist = best_train_error  # we already have it, but let's keep name consistent

        # Evaluate on TEST
        Y_pred_test = X_test_bias @ best_beta  # (N_test, B)
        dist_test = torch.sqrt(torch.sum((Y_pred_test - Y_test)**2, dim=1))
        test_norm_dist = (dist_test * w_test).sum().item()
        best_predictions_test = Y_pred_test

    # -------------------------------------------------------------
    # 4) Build and return final dictionary
    # -------------------------------------------------------------
    result_dict = {
        "best_rcond":         best_rcond,
        "train_norm_dist":    train_norm_dist,  
        "test_norm_dist":     test_norm_dist,   
        "best_weights":       best_beta.detach().cpu().numpy() if best_beta is not None else None,
        "train_predictions":  best_predictions_train.detach().cpu().numpy() if best_predictions_train is not None else None,
        "test_predictions":   best_predictions_test.detach().cpu().numpy()  if best_predictions_test is not None else None,
        "train_true_values":  Y_train.detach().cpu().numpy(),
        "test_true_values":   Y_test.detach().cpu().numpy(),
        "train_weights":      w_train.detach().cpu().numpy(),
        "test_weights":       w_test.detach().cpu().numpy(),
    }

    return result_dict


def run_activation_to_beliefs_regression_cv(
    regression_analyzer,
    activations,
    beliefs,
    probs,
    train_positions,
    test_positions,
    rcond_values
):
    """
    Run activation to beliefs regression using cross-validation.
    """
    train_probs = probs[train_positions]
    test_probs  = probs[test_positions]

    # -----------------------------
    # 0) Normalize the train & test probabilities
    # -----------------------------
    train_probs = train_probs / train_probs.sum()
    test_probs  = test_probs  / test_probs.sum()

    # -----------------------------
    # 1) Construct X with a bias column for the train set
    # -----------------------------
    N_train = len(train_positions)
    N_test  = len(test_positions)

    ones_train = torch.ones((N_train, 1), device=activations.device)
    X_train_bias = torch.cat([ones_train, activations[train_positions]], dim=1)  # (N_train, D+1)   

    # Weighted design
    sqrt_w_train = torch.sqrt(train_probs).unsqueeze(1)         # (N_train, 1)
    X_train_weighted = X_train_bias * sqrt_w_train          # (N_train, D+1)
    Y_train = beliefs[train_positions]
    Y_train_weighted = Y_train * sqrt_w_train               # (N_train, B)

    # -----------------------------
    # Construct X with a bias column for the test set
    # -----------------------------
    ones_test = torch.ones((N_test, 1), device=activations.device)
    X_test_bias = torch.cat([ones_test, activations[test_positions]], dim=1)     # (N_test, D+1)
    Y_test = beliefs[test_positions]
    sqrt_w_test = torch.sqrt(test_probs).unsqueeze(1)         # (N_test, 1)
    X_test_weighted = X_test_bias * sqrt_w_test          # (N_test, D+1)
    Y_test_weighted = Y_test * sqrt_w_test               # (N_test, B)

    pinvs_dict, singular_values = compute_efficient_pinv_from_svd(X_train_weighted, rcond_values)  # # NEW

    # -----------------------------
    # 2) For each rcond, fit on TRAIN and measure TRAIN error
    # -----------------------------
    best_train_error = float("inf")
    best_rcond = None
    best_beta = None
    best_predictions_train = None

    for rcond in rcond_values:
        pinv_A = pinvs_dict[rcond]  # (D+1, D+1)
        # NEW METHOD: beta = pinv_A @ (X_train_weighted^T @ Y_train_weighted)
        beta = pinv_A @  Y_train_weighted  # shape (D+1, B)
        Y_pred_train = X_train_bias @ beta  # (N_train, B)
        dist_train = torch.sqrt(torch.sum((Y_pred_train - Y_train)**2, dim=1))  # (N_train,)
        train_error = (dist_train * train_probs).sum().item()

        # If this rcond is better on the TRAIN set, update
        if train_error < best_train_error:
            best_train_error = train_error
            best_rcond = rcond
            best_beta = beta
            best_predictions_train = Y_pred_train

    # -------------------------------------------------------------
    # 3) Evaluate the final chosen model on TRAIN and TEST
    # ------------------------------------------------------------- 
    train_norm_dist = float("inf")
    test_norm_dist  = float("inf")
    best_predictions_test = None

    if best_beta is not None:
        # Recompute the final train predictions if needed
        train_norm_dist = best_train_error  # we already have it, but let's keep name consistent

        # Evaluate on TEST
        Y_pred_test = X_test_bias @ best_beta  # (N_test, B)
        dist_test = torch.sqrt(torch.sum((Y_pred_test - Y_test)**2, dim=1))
        test_norm_dist = (dist_test * test_probs).sum().item()
        best_predictions_test = Y_pred_test

    # -------------------------------------------------------------
    # 4) Build and return final dictionary
    # -------------------------------------------------------------
    result_dict = {
        "best_rcond":         best_rcond,
        "train_norm_dist":    train_norm_dist,  
        "norm_dist":          test_norm_dist,   
        "predictions":        best_predictions_test.detach().cpu().numpy()  if best_predictions_test is not None else None,
        "true_values":        Y_test.detach().cpu().numpy(),
        "weights":            test_probs.detach().cpu().numpy()
    }

    return result_dict


def run_activation_to_beliefs_regression_kf(
    regression_analyzer,
    activations,
    beliefs,
    probs,
    kf,
    rcond_values
):
    """
    Run activation to beliefs regression using K-Fold cross-validation to tune rcond hyperparameter,
    then train a final model on the entire dataset using the best overall rcond.

    Args:
        regression_analyzer: Instance of RegressionAnalyzer (unused but kept for API consistency)
        activations: Tensor of activations (N, D)
        beliefs: Tensor of beliefs (N, B)
        probs: Tensor of original probabilities/weights (N,)
        kf: List of tuples [(train_indices_1, test_indices_1), ...] for K-fold CV
        rcond_values: List of rcond values to try

    Returns:
        Dictionary containing:
            best_overall_rcond: The rcond value that performed best across all folds
            avg_test_errors: Dictionary mapping each rcond to its average test error
            final_metrics: Results from the final model trained on all data
            per_fold_results: Detailed results from each fold
    """
    # For aggregating test errors per rcond across folds
    rcond_errors = defaultdict(list)
    # Special key for sklearn fallback
    sklearn_fallback_key = "sklearn_fallback"
    
    all_fold_results = []
    num_valid_folds = 0
    
    # Run K-fold CV to find best rcond
    for fold_idx, (train_indices, test_indices) in enumerate(kf):
        #print(f"Processing fold {fold_idx+1}/{len(kf)}")
        
        # Evaluate all rcond values on this fold's test set
        fold_result = _run_single_fold_regression(
            activations=activations,
            beliefs=beliefs,
            probs=probs,
            train_indices=train_indices,
            test_indices=test_indices,
            rcond_values=rcond_values
        )
        
        # Store detailed fold results
        all_fold_results.append(fold_result)
        
        # Aggregate errors by rcond value
        if fold_result.get("rcond_test_errors"):
            has_valid_errors = False
            for rcond, error in fold_result["rcond_test_errors"].items():
                if error is not None and not np.isinf(error) and not np.isnan(error):
                    rcond_errors[rcond].append(error)
                    has_valid_errors = True
            
            if has_valid_errors:
                num_valid_folds += 1
        else:
            print(f"Warning: Fold {fold_idx+1} failed to produce valid results")
    
    # Calculate average error per rcond across folds
    avg_test_errors = {}
    for rcond, errors in rcond_errors.items():
        if errors:  # Only if we have valid errors for this rcond
            avg_test_errors[rcond] = np.mean(errors)
    
    if not avg_test_errors:
        print("Error: No valid cross-validation results obtained")
        return {
            "best_overall_rcond": None,
            "avg_test_errors": {},
            "final_metrics": {},
            "per_fold_results": all_fold_results
        }
    
    # Find the best overall rcond across all folds
    # Fix for linter error - use a different approach to find the minimum
    min_avg_cv_error = float('inf')
    best_overall_rcond = None
    for rcond, error in avg_test_errors.items():
        if error < min_avg_cv_error and not np.isnan(error) and not np.isinf(error):
            min_avg_cv_error = error
            best_overall_rcond = rcond
    
    if best_overall_rcond is None:
        print("Error: Could not find a valid best rcond")
        return {
            "best_overall_rcond": None,
            "avg_test_errors": avg_test_errors,
            "final_metrics": {},
            "per_fold_results": all_fold_results
        }
    
    print(f"Best rcond from CV: {best_overall_rcond}, with avg error: {min_avg_cv_error:.6f}")
    
    # Now train final model on the ENTIRE dataset using the best rcond
    final_metrics = _train_final_model(
        activations=activations,
        beliefs=beliefs, 
        probs=probs,
        best_rcond=best_overall_rcond,
        sklearn_fallback_key=sklearn_fallback_key
    )
    
    # Create backward-compatible return structure
    backward_compatible_results = {
        # Keep original backward compatibility keys
        "avg_fold_test_error": min_avg_cv_error,  # Minimum average error across folds
        "avg_fold_train_error": min_avg_cv_error,  # Just use the same error for backward compatibility
        "norm_dist": final_metrics.get("norm_dist", float("inf")),  # The weighted error from final model
        "per_fold_results": all_fold_results,
        
        # Add the predictions, true values and weights from the final model
        "predictions": final_metrics.get("predictions", None),
        "true_values": final_metrics.get("true_values", None),
        "weights": final_metrics.get("weights", None),
        
        # New structure keys
        "best_overall_rcond": best_overall_rcond,
        "avg_test_errors": avg_test_errors,
        "final_metrics": final_metrics,
    }
    
    return backward_compatible_results


def _run_single_fold_regression(
    activations,
    beliefs,
    probs,
    train_indices,
    test_indices,
    rcond_values
):
    """
    Perform regression on a single train/test fold, evaluating all rcond values on test data.
    
    Args:
        activations: Tensor of activations (N, D)
        beliefs: Tensor of beliefs (N, B)
        probs: Tensor of probabilities/weights (N,)
        train_indices: Indices for training set
        test_indices: Indices for test set
        rcond_values: List of rcond values to evaluate
        
    Returns:
        Dictionary containing test errors for each rcond value
    """
    device = activations.device
    sklearn_fallback_key = "sklearn_fallback"
    
    # Prepare data for this fold
    # Convert indices to tensors if needed
    if not isinstance(train_indices, torch.Tensor):
        train_indices = torch.tensor(train_indices, device=device, dtype=torch.long)
    if not isinstance(test_indices, torch.Tensor):
        test_indices = torch.tensor(test_indices, device=device, dtype=torch.long)
    
    # Extract fold data
    X_train = activations[train_indices]
    Y_train = beliefs[train_indices]
    probs_train = probs[train_indices]
    
    X_test = activations[test_indices]
    Y_test = beliefs[test_indices]
    probs_test = probs[test_indices]
    
    # Normalize probabilities within the fold
    probs_train_norm = probs_train / probs_train.sum()
    probs_test_norm = probs_test / probs_test.sum()
    
    # Add bias terms
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    
    ones_train = torch.ones((N_train, 1), device=device)
    X_train_bias = torch.cat([ones_train, X_train], dim=1)
    
    ones_test = torch.ones((N_test, 1), device=device)
    X_test_bias = torch.cat([ones_test, X_test], dim=1)
    
    # Weighted design matrix for training
    sqrt_weights_train = torch.sqrt(probs_train_norm).unsqueeze(1)
    X_train_weighted = X_train_bias * sqrt_weights_train
    Y_train_weighted = Y_train * sqrt_weights_train
    
    # Dictionary to collect test errors for each rcond
    rcond_test_errors = {}
    
    # Try SVD approach for all rcond values
    try:
        # Compute pseudoinverses for all rcond values
        pinvs_dict, _ = compute_efficient_pinv_from_svd(X_train_weighted, rcond_values)
        
        # For each rcond, train model and evaluate on test set
        for rcond in rcond_values:
            if rcond in pinvs_dict:
                try:
                    # Calculate beta using current rcond
                    pinv_A = pinvs_dict[rcond]
                    beta = pinv_A @ Y_train_weighted
                    
                    # Predict on test set
                    Y_pred_test = X_test_bias @ beta
                    
                    # Calculate weighted test error
                    dists = torch.sqrt(torch.sum((Y_pred_test - Y_test)**2, dim=1))
                    test_error = (dists * probs_test_norm).sum().item()
                    
                    # Store test error for this rcond
                    rcond_test_errors[rcond] = test_error
                except Exception as e:
                    print(f"Error evaluating rcond {rcond}: {e}")
                    rcond_test_errors[rcond] = float('inf')
    
    except Exception as e:
        print(f"SVD approach failed: {e}")
        print("Trying sklearn fallback...")
        
        # If SVD fails completely, try sklearn
        try:
            from sklearn.linear_model import LinearRegression
            
            # Convert to numpy
            X_train_np = X_train.cpu().detach().numpy()
            Y_train_np = Y_train.cpu().detach().numpy()
            train_weights_np = probs_train_norm.cpu().detach().numpy()
            
            X_test_np = X_test.cpu().detach().numpy()
            Y_test_np = Y_test.cpu().detach().numpy()
            test_weights_np = probs_test_norm.cpu().detach().numpy()
            
            # Train model with sklearn
            model = LinearRegression(fit_intercept=True)
            model.fit(X_train_np, Y_train_np, sample_weight=train_weights_np)
            
            # Predict and evaluate
            Y_pred_test_np = model.predict(X_test_np)
            
            # Calculate weighted test error
            dists = np.sqrt(np.sum((Y_pred_test_np - Y_test_np)**2, axis=1))
            test_error = np.sum(dists * test_weights_np)
            
            # Store with special key
            rcond_test_errors[sklearn_fallback_key] = float(test_error)
            
        except Exception as e:
            print(f"Sklearn fallback also failed: {e}")
            # Return empty dict if everything fails
            return {"rcond_test_errors": {}}
    
    return {"rcond_test_errors": rcond_test_errors}


def _train_final_model(activations, beliefs, probs, best_rcond, sklearn_fallback_key="sklearn_fallback"):
    """
    Train a final model on the entire dataset using the best rcond from CV.
    
    Args:
        activations: Tensor of activations (N, D)
        beliefs: Tensor of beliefs (N, B)
        probs: Tensor of probabilities/weights (N,)
        best_rcond: Best rcond value from cross-validation
        sklearn_fallback_key: Key indicating sklearn should be used
        
    Returns:
        Dictionary with final model metrics including norm_dist, r_squared, mse, mae, rmse
    """
    device = activations.device
    
    # Prepare full dataset
    X = activations
    Y = beliefs
    N, D = X.shape
    B = Y.shape[1] # Get belief dimension

    # Normalize probabilities for training stability (e.g., pinv)
    probs_norm = probs / probs.sum()
    
    # Prepare for prediction
    ones = torch.ones((N, 1), device=device)
    X_bias = torch.cat([ones, X], dim=1)
    
    # Check which method to use
    if best_rcond == sklearn_fallback_key:
        print("Training final model with sklearn (selected by CV)")
        try:
            from sklearn.linear_model import LinearRegression
            
            # Convert to numpy
            X_np = X.cpu().detach().numpy()
            Y_np = Y.cpu().detach().numpy()
            weights_np = probs_norm.cpu().detach().numpy() # Use normalized for fitting consistency
            
            # Train final model
            model = LinearRegression(fit_intercept=True)
            model.fit(X_np, Y_np, sample_weight=weights_np)
            
            # Get predictions
            Y_pred_np = model.predict(X_np)
            Y_pred = torch.tensor(Y_pred_np, device=device)
            
            model_type = "sklearn"
        except Exception as e:
            print(f"Error training final sklearn model: {e}")
            return {"error": str(e)}
    
    else:
        print(f"Training final model with SVD using rcond={best_rcond}")
        try:
            # Weighted design matrix using normalized probs
            sqrt_weights = torch.sqrt(probs_norm).unsqueeze(1)
            X_weighted = X_bias * sqrt_weights
            Y_weighted = Y * sqrt_weights
            
            # Compute pseudoinverse for best rcond
            pinvs_dict, _ = compute_efficient_pinv_from_svd(X_weighted, [best_rcond])
            
            if best_rcond in pinvs_dict:
                pinv_A = pinvs_dict[best_rcond]
                beta = pinv_A @ Y_weighted
                
                # Predict on full dataset
                Y_pred = X_bias @ beta
                model_type = "svd"
            else:
                raise ValueError(f"Failed to compute pseudoinverse for rcond={best_rcond}")
                
        except Exception as e:
            print(f"Error training final SVD model: {e}")
            print("Trying sklearn fallback...")
            
            try:
                from sklearn.linear_model import LinearRegression
                
                # Convert to numpy
                X_np = X.cpu().detach().numpy()
                Y_np = Y.cpu().detach().numpy()
                weights_np = probs_norm.cpu().detach().numpy() # Use normalized for fitting
                
                # Train final model
                model = LinearRegression(fit_intercept=True)
                model.fit(X_np, Y_np, sample_weight=weights_np)
                
                # Get predictions
                Y_pred_np = model.predict(X_np)
                Y_pred = torch.tensor(Y_pred_np, device=device)
                
                model_type = "sklearn_fallback"
            except Exception as e_sk:
                print(f"Final sklearn fallback also failed: {e_sk}")
                return {"error": str(e_sk)}
    
    # --- Calculate metrics for the final model ---
    
    # Ensure probs sum is non-zero for weighting
    total_prob_mass = probs.sum()
    if total_prob_mass <= 0:
        print("Warning: Total probability mass is zero or negative. Metrics will be NaN.")
        mse = torch.full((B,), float('nan'), device=device)
        mae = torch.full((B,), float('nan'), device=device)
        rmse = torch.full((B,), float('nan'), device=device)
        norm_dist = float('nan')
        r_squared = float('nan')
    else:
        # 1. Weighted norm distance (sum of weighted Euclidean distances) using original probs
        distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
        norm_dist = (distances * probs).sum().item()
        
        # 2. Calculate R^2 (weighted) using normalized probs (consistent with typical R^2 definition)
        sqrt_weights_norm = torch.sqrt(probs_norm).unsqueeze(1)
        Y_weighted_mean = (Y * sqrt_weights_norm).sum(dim=0) / sqrt_weights_norm.sum()

        total_var = torch.sum(((Y * sqrt_weights_norm) - Y_weighted_mean)**2)
        residual_var = torch.sum(((Y_pred - Y) * sqrt_weights_norm)**2)
        r_squared = (1.0 - (residual_var / total_var)).item() if total_var > 0 else 0

        # 3. Calculate MSE, MAE, RMSE (weighted per belief dimension) using original probs
        squared_errors = (Y_pred - Y)**2
        abs_errors = torch.abs(Y_pred - Y)
        
        # Reshape probs to (N, 1) for broadcasting with errors (N, B)
        probs_reshaped = probs.unsqueeze(1)
        
        mse = (squared_errors * probs_reshaped).sum(dim=0) / total_prob_mass
        mae = (abs_errors * probs_reshaped).sum(dim=0) / total_prob_mass
        rmse = torch.sqrt(mse)
    
    # Return final metrics (converting relevant tensors to numpy)
    return {
        "dist": norm_dist,
        "r2": r_squared,
        "mse": mse.cpu().detach().numpy(), # Added
        "mae": mae.cpu().detach().numpy(), # Added
        "rmse": rmse.cpu().detach().numpy(), # Added
        "predictions": Y_pred.cpu().detach().numpy(),
        "true_values": Y.cpu().detach().numpy(),
        "weights": probs.cpu().detach().numpy(), # Return original probs used for weighting metrics
        "model_type": model_type
    }








import torch
import numpy as np
from collections import defaultdict
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score # For weighted R2 calculation
# Ensure you have scikit-learn installed: pip install scikit-learn

# Removed _run_single_fold_ridge_regression and _train_final_model_ridge
# as RidgeCV handles both CV tuning and final model fitting.

# It's good practice to rename the function if its core mechanism changes
def run_activation_to_beliefs_regression_ridgecv(
    regression_analyzer, # Unused, kept for API consistency
    activations,
    beliefs,
    probs,
    kf, # KFold object is NO LONGER USED by RidgeCV default (GCV), but kept for API compatibility
    alpha_values
):
    """
    Run activation to beliefs regression using RidgeCV to efficiently find the
    best L2 regularization alpha via internal cross-validation (GCV default),
    and evaluate the final model trained on the entire dataset.

    Maintains backward compatibility with the return structure of the previous
    rcond/manual-CV based functions.

    Args:
        regression_analyzer: Instance (unused).
        activations: Tensor of activations (N, D).
        beliefs: Tensor of beliefs (N, B).
        probs: Tensor of original probabilities/weights (N,).
        kf: KFold object or iterable (IGNORED by default RidgeCV, kept for API).
           To use K-Fold with RidgeCV, pass cv=kf directly to RidgeCV constructor.
        alpha_values: List or iterable of alpha (L2 regularization strength) values to try.

    Returns:
        Dictionary containing (structured for backward compatibility):
            best_overall_rcond: The best *alpha* value found via CV (renamed key).
            avg_test_errors: Dictionary mapping each alpha to its avg CV error (MSE).
            final_metrics: Results from the final Ridge model.
            per_fold_results: Set to None (not available from default RidgeCV).
            avg_fold_test_error: Minimum average CV error (MSE) for the best alpha.
            avg_fold_train_error: Set equal to avg_fold_test_error for compatibility.
            norm_dist: Weighted norm distance from the final model.
            predictions: Predictions from the final model on the full dataset.
            true_values: True belief values for the full dataset.
            weights: Original probabilities/weights for the full dataset.
            r_squared: Weighted R-squared score of the final model.
    """

    device = activations.device # Get device from input tensor

    # --- Prepare full dataset and convert to NumPy ---

    X_np = activations.cpu().detach().numpy()
    Y_np = beliefs.cpu().detach().numpy()
    original_probs_np = probs.cpu().detach().numpy()
    
    # Normalize probabilities for sample weighting during RidgeCV fitting
    weights_np = original_probs_np / np.sum(original_probs_np)


    # RidgeCV can handle multidimensional targets natively
    model = RidgeCV(
        alphas=alpha_values,
        fit_intercept=True,
        scoring='neg_mean_squared_error',
        cv=None,  # Use GCV (efficient LOO) by default
        store_cv_values=True,
        gcv_mode='auto'
    )

    # Fit the model - it finds the best alpha internally using CV
    print(f"Fitting RidgeCV model with {len(alpha_values)} alpha values")
    print(f"X_np shape: {X_np.shape}, Y_np shape: {Y_np.shape}, weights_np shape: {weights_np.shape}")
    
    # Fit model directly on multidimensional target
    model.fit(X_np, Y_np, sample_weight=weights_np)
    best_alpha = model.alpha_
    Y_pred_np = model.predict(X_np)
    
    # Extract cross-validation results - SIMPLIFIED
    # Average over all dimensions except the last one (alpha values)
    axes_to_average = tuple(range(model.cv_values_.ndim - 1))
    mean_neg_mse_per_alpha = np.mean(model.cv_values_, axis=axes_to_average)

    # Convert negative MSE to positive MSE
    avg_mse_per_alpha = -mean_neg_mse_per_alpha

    # Create mapping of alpha values to their errors and find the best one's error
    avg_test_errors_dict = {alpha: err for alpha, err in zip(alpha_values, avg_mse_per_alpha)}
    min_avg_cv_error = next((err for a, err in avg_test_errors_dict.items()
                          if np.isclose(a, best_alpha)), float('inf'))

    # --- Calculate final metrics using the predictions ---
    # 1. Weighted norm distance using ORIGINAL probs
    distances_np = np.sqrt(np.sum((Y_pred_np - Y_np)**2, axis=1))
    norm_dist = np.sum(distances_np * original_probs_np)


    r_squared = r2_score(Y_np, Y_pred_np, sample_weight=weights_np, multioutput='variance_weighted')

    # --- Create backward-compatible return structure ---
    backward_compatible_results = {
        # --- New / Updated Keys (Recommended for clarity) ---
        "best_overall_alpha": best_alpha,
        "avg_alpha_test_errors_mse": avg_test_errors_dict, # Clarify it's MSE
        "final_ridge_metrics": {
            "norm_dist": norm_dist,
            "r_squared": r_squared,
            "predictions": Y_pred_np,
            "true_values": Y_np,
            "weights": original_probs_np, # Return original weights
            "model_type": "RidgeCV",
            "best_alpha": best_alpha
        },

        # --- Backward Compatibility Keys ---
        "best_overall_rcond": best_alpha, # Store best alpha under old key name
        "avg_test_errors": avg_test_errors_dict, # Store alpha->MSE map under old key name
        "per_fold_results": None, # Set to None as per-fold not easily available from GCV
        "avg_fold_test_error": min_avg_cv_error, # Min avg CV MSE for best alpha
        "avg_fold_train_error": min_avg_cv_error, # Set same as test for compatibility
        "norm_dist": norm_dist,
        "predictions": Y_pred_np,
        "true_values": Y_np,
        "weights": original_probs_np,
        "r_squared": r_squared,
        # Add the final_metrics dict itself under the old name if needed
        "final_metrics": {
            "norm_dist": norm_dist,
            "r_squared": r_squared,
            "predictions": Y_pred_np,
            "true_values": Y_np,
            "weights": original_probs_np,
            "model_type": "RidgeCV",
            "best_alpha": best_alpha
        }
    }

    return backward_compatible_results

def compute_weighted_pca_variance(X, W, B):
    """
    Performs weighted PCA and calculates cumulative explained variance.

    Args:
        X (torch.Tensor): Input data tensor of shape (N, D).
        W (torch.Tensor): Probability weights tensor of shape (N,).
        B (int): Target belief dimension (number of components to check).

    Returns:
        tuple: 
            - cumulative_variance (np.ndarray): Array of cumulative explained variance (D,).
            - variance_at_B (float): Cumulative variance explained by the top B components.
            - variance_at_B_minus_1 (float): Cumulative variance explained by the top B-1 components.
    """
    N, D = X.shape
    device = X.device

    # Ensure weights sum to 1
    W_norm = W / W.sum()

    # Calculate weighted mean
    mean = (X * W_norm[:, None]).sum(dim=0)

    # Center the data
    X_centered = X - mean

    # Apply square root weights
    X_weighted_centered = X_centered * torch.sqrt(W_norm)[:, None]

    # Perform SVD
    try:
        _, S, _ = torch.svd_lowrank(X_weighted_centered, q=30, niter=1000)
    except torch.linalg.LinAlgError as e:
        print(f"SVD failed: {e}. Returning NaNs for PCA variance.")
        nan_array = np.full(min(N, D), np.nan)
        return nan_array, np.nan, np.nan

    # Calculate explained variance ratios
    explained_variance = S.pow(2) / S.pow(2).sum()

    # Calculate cumulative explained variance
    cumulative_variance = torch.cumsum(explained_variance, dim=0)

    # Ensure B and B-1 are valid indices
    if B >= 1 and B <= len(cumulative_variance):
        variance_at_B = cumulative_variance[B-1].item()
    else:
        variance_at_B = np.nan # Or handle as error/warning
        print(f"Warning: Target dimension B={B} is out of bounds [1, {len(cumulative_variance)}]")

    if B - 1 >= 1 and B - 1 <= len(cumulative_variance):
        variance_at_B_minus_1 = cumulative_variance[B-2].item()
    else:
        variance_at_B_minus_1 = np.nan # Or handle appropriately
        if B-1 >= 1:
             print(f"Warning: Target dimension B-1={B-1} is out of bounds [1, {len(cumulative_variance)}]")

    return cumulative_variance.cpu().numpy(), variance_at_B, variance_at_B_minus_1



from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_activation_to_beliefs_regression_pls(
    regression_analyzer,  # Unused, kept for API consistency
    activations,
    beliefs,
    probs,
    kf,  # KFold object or iterable (IGNORED, kept for API compatibility)
    alpha_values  # Unused, kept for API compatibility
):
    """
    Run activation to beliefs regression using PLSRegression.

    Maintains backward compatibility with the return structure of the previous
    rcond/manual-CV based functions.

    Args:
        regression_analyzer: Instance (unused).
        activations: Tensor of activations (N, D).
        beliefs: Tensor of beliefs (N, B).
        probs: Tensor of original probabilities/weights (N,).
        kf: KFold object or iterable (IGNORED, kept for API compatibility).
        alpha_values: List or iterable of alpha values (IGNORED, kept for API compatibility).

    Returns:
        Dictionary containing (structured for backward compatibility):
            best_overall_rcond: Set to None (not applicable for PLS).
            avg_test_errors: Set to None (not applicable for PLS).
            final_metrics: Results from the final PLS model.
            per_fold_results: Set to None (not applicable for PLS).
            avg_fold_test_error: Set to None (not applicable for PLS).
            avg_fold_train_error: Set to None (not applicable for PLS).
            norm_dist: Weighted norm distance from the final model.
            predictions: Predictions from the final model on the full dataset.
            true_values: True belief values for the full dataset.
            weights: Original probabilities/weights for the full dataset.
            r_squared: Weighted R-squared score of the final model.
    """

    device = activations.device  # Get device from input tensor

    # --- Prepare full dataset and convert to NumPy ---

    X_np = activations.cpu().detach().numpy()
    Y_np = beliefs.cpu().detach().numpy()
    original_probs_np = probs.cpu().detach().numpy()

    # Normalize probabilities for sample weighting
    weights_np = original_probs_np / np.sum(original_probs_np)

    # PLSRegression can handle multidimensional targets natively
    model = PLSRegression(n_components=min(X_np.shape[1], Y_np.shape[1]))

    # Fit the model
    print(f"Fitting PLSRegression model with {model.n_components} components")
    print(f"X_np shape: {X_np.shape}, Y_np shape: {Y_np.shape}, weights_np shape: {weights_np.shape}")

    model.fit(X_np, Y_np)
    Y_pred_np = model.predict(X_np)

    # --- Calculate final metrics using the predictions ---
    # 1. Weighted norm distance using ORIGINAL probs
    distances_np = np.sqrt(np.sum((Y_pred_np - Y_np) ** 2, axis=1))
    norm_dist = np.sum(distances_np * original_probs_np)

    r_squared = r2_score(Y_np, Y_pred_np, sample_weight=weights_np, multioutput='variance_weighted')

    # --- Create backward-compatible return structure ---
    backward_compatible_results = {
        # --- New / Updated Keys (Recommended for clarity) ---
        "best_overall_alpha": None,  # Not applicable for PLS
        "avg_alpha_test_errors_mse": None,  # Not applicable for PLS
        "final_ridge_metrics": {
            "norm_dist": norm_dist,
            "r_squared": r_squared,
            "predictions": Y_pred_np,
            "true_values": Y_np,
            "weights": original_probs_np,  # Return original weights
            "model_type": "PLSRegression",
            "best_alpha": None  # Not applicable for PLS
        },

        # --- Backward Compatibility Keys ---
        "best_overall_rcond": None,  # Not applicable for PLS
        "avg_test_errors": None,  # Not applicable for PLS
        "per_fold_results": None,  # Not applicable for PLS
        "avg_fold_test_error": None,  # Not applicable for PLS
        "avg_fold_train_error": None,  # Not applicable for PLS
        "norm_dist": norm_dist,
        "predictions": Y_pred_np,
        "true_values": Y_np,
        "weights": original_probs_np,
        "r_squared": r_squared,
        # Add the final_metrics dict itself under the old name if needed
        "final_metrics": {
            "norm_dist": norm_dist,
            "r_squared": r_squared,
            "predictions": Y_pred_np,
            "true_values": Y_np,
            "weights": original_probs_np,
            "model_type": "PLSRegression",
            "best_alpha": None  # Not applicable for PLS
        }
    }

    return backward_compatible_results


def run_activation_to_beliefs_regression_pca(
    regression_analyzer,  # Unused, kept for API consistency
    activations,
    beliefs,
    probs,
    kf,  # KFold object or iterable (IGNORED, kept for API compatibility)
    rcond_values  # Changed from alpha_values
):
    """
    Run activation to beliefs regression using PCA followed by linear regression
    with rcond sweep for regularization.

    Maintains backward compatibility with the return structure of the previous
    rcond/manual-CV based functions.

    Args:
        regression_analyzer: Instance (unused).
        activations: Tensor of activations (N, D).
        beliefs: Tensor of beliefs (N, B).
        probs: Tensor of original probabilities/weights (N,).
        kf: KFold object or iterable (IGNORED, kept for API compatibility).
        rcond_values: List or iterable of rcond values to try for the pseudoinverse.

    Returns:
        Dictionary containing (structured for backward compatibility):
            best_overall_rcond: The best rcond value found.
            avg_test_errors: Set to None (no CV used here).
            final_metrics: Results from the final regression model.
            per_fold_results: Set to None (no CV used here).
            avg_fold_test_error: Set to None (no CV used here).
            avg_fold_train_error: Set to None (no CV used here).
            norm_dist: Weighted norm distance from the final model.
            predictions: Predictions from the final model on the full dataset.
            true_values: True belief values for the full dataset.
            weights: Original probabilities/weights for the full dataset.
            r_squared: Weighted R-squared score of the final model.
            var_explained: Variance explained by each principal component.
            dims_95_var: Number of dimensions needed to explain 95% of variance.
            dims_99_var: Number of dimensions needed to explain 99% of variance.
    """

    device = activations.device  # Get device from input tensor

    # --- Prepare full dataset and convert to NumPy for PCA ---
    DO_WEIGHTED_PCA = False # if false, use unweighted PCA
    if DO_WEIGHTED_PCA:
        X_np = activations.cpu().detach().numpy()
        Y_np = beliefs.cpu().detach().numpy()
        original_probs_np = probs.cpu().detach().numpy()
    else:
        X_np = activations.cpu().detach().numpy()
        Y_np = beliefs.cpu().detach().numpy()
        original_probs_np = probs.cpu().detach().numpy()

    # Normalize probabilities for sample weighting in PCA
    if DO_WEIGHTED_PCA:
        weights_pca_np = original_probs_np / np.sum(original_probs_np)
    else:
        weights_pca_np = np.ones(original_probs_np.shape) / original_probs_np.shape[0]

    # --- Perform weighted PCA on activations manually ---
    # Center the data
    mean_X = np.average(X_np, axis=0, weights=weights_pca_np)
    X_centered = X_np - mean_X

    # Compute the weighted covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False, aweights=weights_pca_np, bias=True)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate variance explained by each component
    total_variance = np.sum(eigenvalues)
    var_explained = eigenvalues / total_variance

    # Calculate cumulative variance explained
    cumulative_var_explained = np.cumsum(var_explained)

    # Determine number of dimensions needed for 95% and 99% variance
    dims_95_var = np.searchsorted(cumulative_var_explained, 0.95) + 1
    dims_99_var = np.searchsorted(cumulative_var_explained, 0.99) + 1
    dims_999_var = np.searchsorted(cumulative_var_explained, 0.999) + 1


    # Select the top components to keep 99% variance (or fewer if available)
    n_components = min(int(dims_95_var), X_np.shape[1]) # Cast dims_999_var to int
    principal_components = eigenvectors[:, :n_components]

    # Transform the data using PCA
    X_pca_np = np.dot(X_centered, principal_components)

    # --- Convert back to Tensors for Regression with Rcond Sweep ---
    X_pca = torch.tensor(X_pca_np, device=device, dtype=activations.dtype)
    Y = beliefs # Use original beliefs tensor
    original_probs = probs # Use original probs tensor

    # Normalize original probabilities for R-squared calculation later
    probs_norm = original_probs / original_probs.sum()

    # --- Perform Regression with Rcond Sweep on PCA components ---
    print(f"Fitting regression model after PCA ({n_components} components) with rcond sweep")
    print(f"X_pca shape: {X_pca.shape}, Y shape: {Y.shape}, Probs shape: {original_probs.shape}")

    # Add bias term to PCA components
    ones = torch.ones(X_pca.shape[0], 1, device=device)
    X_bias = torch.cat([ones, X_pca], dim=1) # Shape (N, n_components + 1)

    # Weighted design matrix (using normalized probs for stability in pinv)
    sqrt_weights_norm = torch.sqrt(probs_norm).unsqueeze(1)
    X_weighted = X_bias * sqrt_weights_norm
    Y_weighted = Y * sqrt_weights_norm

    # RE-ADDED: Compute pseudoinverses of X_weighted for all rcond values using the efficient SVD method
    try:
        # Call compute_efficient_pinv_from_svd directly on X_weighted
        pinv_Xw_dict, singular_values = compute_efficient_pinv_from_svd(X_weighted, rcond_values)
    except Exception as e:
        print(f"Error computing efficient pinv for X_weighted: {e}")
        # Handle error case, maybe return NaNs or raise
        return {"error": f"Pinverse calculation failed for X_weighted: {e}"}

    # Variables to store the best result based on lowest norm_dist
    best_norm_dist = float('inf')
    best_rcond = None
    best_beta = None
    best_predictions = None

    # Loop through rcond values to find the best one
    for rcond in rcond_values:
        # MODIFIED: Retrieve pre-computed pinv_Xw from the dictionary
        if rcond not in pinv_Xw_dict:
            print(f"Warning: rcond {rcond} not found in precomputed pinv_Xw_dict, skipping.")
            continue
        
        pinv_Xw = pinv_Xw_dict[rcond]

        # Optional: Check for NaNs/Infs if necessary (might be redundant if compute_efficient handles it)
        if torch.isnan(pinv_Xw).any() or torch.isinf(pinv_Xw).any():
                print(f"Warning: Precomputed pinv_Xw resulted in NaN/Inf for rcond={rcond}. Skipping.")
                continue

        # Calculate beta using the direct method with the precomputed pinv_Xw
        beta = pinv_Xw @ Y_weighted

        # Make predictions on the full dataset using this beta
        Y_pred = X_bias @ beta

        # Calculate weighted error (norm_dist) using ORIGINAL probabilities
        distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
        current_norm_dist = (distances * original_probs).sum().item()

        # Check if this is the best result so far
        if current_norm_dist < best_norm_dist:
            best_norm_dist = current_norm_dist
            best_rcond = rcond
            best_beta = beta
            best_predictions = Y_pred

    if best_beta is None:
        print("Warning: No valid rcond found, regression failed.")
        # Handle failure case
        return {"error": "Regression failed, no valid rcond found."}

    # --- Calculate final metrics using the best model ---
    final_norm_dist = best_norm_dist

    # Calculate R-squared using the best predictions and NORMALIZED weights
    # (R-squared typically uses normalized weights)
    Y_weighted_mean = (Y * sqrt_weights_norm).sum(dim=0) / sqrt_weights_norm.sum()
    total_var = torch.sum(((Y * sqrt_weights_norm) - Y_weighted_mean)**2)
    explained_var = torch.sum(((best_predictions * sqrt_weights_norm) - Y_weighted_mean)**2)
    final_r_squared = (explained_var / total_var).item() if total_var > 0 else 0

    # Convert final predictions and weights to NumPy
    final_predictions_np = best_predictions.cpu().detach().numpy()
    final_true_values_np = Y.cpu().detach().numpy()
    final_weights_np = original_probs.cpu().detach().numpy() # Return original weights


    # --- Create backward-compatible return structure ---
    backward_compatible_results = {
        # --- New / Updated Keys ---
        # Removed alpha related keys
        "final_pca_rcond_metrics": { # More specific name
            "norm_dist": final_norm_dist,
            "r_squared": final_r_squared,
            "predictions": final_predictions_np,
            "true_values": final_true_values_np,
            "weights": final_weights_np,
            "model_type": f"PCA ({n_components} comps) + RcondSweep",
            "best_rcond": best_rcond
        },

        # --- Backward Compatibility Keys ---
        "best_overall_rcond": best_rcond, # Use the best rcond found
        "avg_test_errors": None, # No CV
        "per_fold_results": None, # No CV
        "avg_fold_test_error": None, # No CV
        "avg_fold_train_error": None, # No CV
        "norm_dist": final_norm_dist,
        "predictions": final_predictions_np,
        "true_values": final_true_values_np,
        "weights": final_weights_np,
        "r_squared": final_r_squared,
        "var_explained": var_explained, # Keep PCA variance info
        "dims_95_var": dims_95_var,
        "dims_99_var": dims_99_var,
        "dims_999_var": dims_999_var,
        "final_metrics": { # Keep this nested dict for compatibility
            "norm_dist": final_norm_dist,
            "r_squared": final_r_squared,
            "predictions": final_predictions_np,
            "true_values": final_true_values_np,
            "weights": final_weights_np,
            "model_type": f"PCA ({n_components} comps) + RcondSweep",
            "best_rcond": best_rcond # Changed from best_alpha
        }
    }

    return backward_compatible_results