"""
Belief Regression Analysis Script

This script performs regression analysis to map neural network activations to belief states
for models trained on stochastic processes. It analyzes how well different architectures
(Transformers, LSTMs, GRUs, RNNs) learn to represent beliefs about hidden Markov models.

The analysis uses k-fold cross-validation with various regularization parameters to find
optimal mappings between activations and theoretical belief states.
"""

# Configuration
output_dir = "belief_regression_results"
N_SPLITS = 3  # K-fold cross-validation splits
DEVICE = 'cpu'  # Device for data extraction and tensor storage
REGRESSION_DEVICE = 'cpu'  # Device for RegressionAnalyzer (set to 'cpu' or 'cuda' if no MPS available)
RANDOM_STATE = 42
ONLY_INITIAL_AND_FINAL = True  # Process only first and last checkpoints

sweep_run_pairs = [
    ("manual_sweep", 0),
]

# %%
import torch
from epsilon_transformers.analysis.load_data import S3ModelLoader
from scripts.activation_analysis.data_loading import ModelDataManager
from scripts.activation_analysis.belief_states import BeliefStateGenerator
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from scripts.activation_analysis.data_loading import ActivationExtractor
from scripts.activation_analysis.config import TRANSFORMER_ACTIVATION_KEYS
from scripts.activation_analysis.config import RCOND_SWEEP_LIST
from typing import Optional, List, Dict, Tuple
import numpy as np # Make sure numpy is imported
import os # Make sure os is imported
import numpy as np
import warnings # To handle potential warnings during PCA
from tqdm.auto import tqdm
import joblib
import os
import numpy as np
from collections import defaultdict

from scripts.activation_analysis.regression import (
    RegressionAnalyzer,
    run_activation_to_beliefs_regression_kf,
)

model_data_manager = ModelDataManager(device=DEVICE, use_company_s3=True)
belief_generator = BeliefStateGenerator(model_data_manager, device=DEVICE)
reg_analyzer = RegressionAnalyzer(device=REGRESSION_DEVICE, use_efficient_pinv=True)
s3_loader = S3ModelLoader(use_company_credentials=True)

os.makedirs(output_dir, exist_ok=True)

def _combine_layer_activations(nn_acts):
    """Combine activations from all layers into a single tensor, by concatenating them
    make sure to reshape back to the original shape
    a single act is of shape (n_samples, n_ctx, d_model)
    so in the end the shape should be (n_samples, n_ctx, d_model*n_layers)"""
    flattened_acts = []
    first_layer_key = list(nn_acts.keys())[0]
    n_samples = nn_acts[first_layer_key].shape[0]
    n_ctx = nn_acts[first_layer_key].shape[1]
    
    all_acts = []
    for layer_act in nn_acts.values():
        # Reshape to (n_samples, -1) to flatten any extra dimensions
        all_acts.append(layer_act)

    cat = torch.cat(all_acts, dim=2) 
    #print(cat.shape)
    return cat

def find_duplicate_prefixes(nn_inputs):
    """
    Find duplicate prefixes in the input sequences and return their indices.
    
    Args:
        nn_inputs: Tensor of shape (batch_size, seq_len) containing token sequences
        
    Returns:
        Dictionary mapping each unique prefix tuple to a list of (seq_idx, pos) tuples
    """
    batch_size, seq_len = nn_inputs.shape
    prefix_to_indices = {}  # (prefix tuple) -> list of (seq_idx, pos) tuples
    
    # Process each sequence and position
    for seq_idx in range(batch_size):
        seq = nn_inputs[seq_idx]
        
        for pos in range(seq_len):
            # Get the prefix up to this position
            prefix = tuple(seq[:pos+1].cpu().numpy().tolist())
            
            # Add this occurrence to our mapping
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append((seq_idx, pos))
    
    return prefix_to_indices

def combine_duplicate_data(prefix_to_indices, activations, nn_probs, belief_states, debug=False, tolerance=1e-6):
    """
    Combine data for duplicate prefixes by summing probabilities.
    
    Args:
        prefix_to_indices: Dictionary mapping prefixes to lists of (seq_idx, pos) tuples
        activations: Tensor of shape (batch_size, seq_len, d_model)
        nn_probs: Tensor of shape (batch_size, seq_len)
        belief_states: Tensor of shape (batch_size, seq_len, belief_dim)
        debug: Whether to print debug information
        tolerance: Tolerance for activation differences
        
    Returns:
        Tuple of (unique_activations, summed_probs, unique_beliefs, unique_prefixes)
    """
    # Dictionary to store unique activations and summed probabilities
    unique_data = {}  # (prefix tuple) -> (activation, summed_prob, belief_state, count)
    
    # Debug information
    if debug:
        activation_diffs = {}  # (prefix tuple) -> max difference observed
        inconsistent_prefixes = []
    
    # Process each unique prefix
    for prefix, indices_list in prefix_to_indices.items():
        first_seq_idx, first_pos = indices_list[0]
        act = activations[first_seq_idx, first_pos]
        prob = nn_probs[first_seq_idx, first_pos]
        belief = belief_states[first_seq_idx, first_pos]
        count = 1
        
        # Process additional occurrences of this prefix
        for seq_idx, pos in indices_list[1:]:
            current_act = activations[seq_idx, pos]
            current_prob = nn_probs[seq_idx, pos]
            
            if debug:
                # Check if activations are the same
                diff = torch.max(torch.abs(act - current_act)).item()
                
                if diff > tolerance:
                    if prefix not in inconsistent_prefixes:
                        inconsistent_prefixes.append(prefix)
                    
                    current_max_diff = activation_diffs.get(prefix, 0)
                    activation_diffs[prefix] = max(current_max_diff, diff)
            
            # Sum the probabilities
            prob += current_prob
            count += 1
        
        # Store the combined data
        unique_data[prefix] = (act, prob, belief, count)
    
    # Print debug info if requested
    if debug:
        if inconsistent_prefixes:
            print(f"WARNING: Found {len(inconsistent_prefixes)} prefixes with inconsistent activations!")
            print(f"Max differences observed:")
            for prefix in sorted(inconsistent_prefixes, key=lambda p: activation_diffs[p], reverse=True)[:10]:
                print(f"  Prefix {prefix}: max diff = {activation_diffs[prefix]}, seen {unique_data[prefix][3]} times")
        else:
            print("All prefixes have consistent activations (within tolerance).")
            
        total_saved = sum(unique_data[p][3] - 1 for p in unique_data)
        total_items = sum(len(indices) for indices in prefix_to_indices.values())
        print(f"Deduplication saved {total_saved} activations out of {total_items}")
    
    # Convert to tensors
    unique_prefixes = list(unique_data.keys())
    unique_activations = torch.stack([unique_data[p][0] for p in unique_prefixes])
    summed_probs = torch.tensor([unique_data[p][1] for p in unique_prefixes])
    unique_beliefs = torch.stack([unique_data[p][2] for p in unique_prefixes])
    
    return unique_activations, summed_probs, unique_beliefs, unique_prefixes

def deduplicate_tensor(prefix_to_indices, tensor, aggregation_fn=None, debug=False, tolerance=1e-6):
    """
    Deduplicate a single tensor based on prefix indices.
    
    Args:
        prefix_to_indices: Dictionary mapping prefixes to lists of (seq_idx, pos) tuples
        tensor: Tensor of shape (batch_size, seq_len, ...) to deduplicate
        aggregation_fn: Function to aggregate duplicate values (default: take first occurrence)
                        Should accept a list of tensor values and return a single value
        debug: Whether to print debug information
        tolerance: Tolerance for tensor value differences
        
    Returns:
        Tuple of (unique_tensor_values, unique_prefixes)
    """
    # Dictionary to store unique tensor values
    unique_data = {}  # (prefix tuple) -> (tensor_value, count)
    
    # Debug information
    if debug:
        value_diffs = {}  # (prefix tuple) -> max difference observed
        inconsistent_prefixes = []
    
    # Process each unique prefix
    for prefix, indices_list in prefix_to_indices.items():
        first_seq_idx, first_pos = indices_list[0]
        value = tensor[first_seq_idx, first_pos]
        count = 1
        
        # If we have an aggregation function and multiple occurrences, prepare to aggregate
        if aggregation_fn is not None and len(indices_list) > 1:
            values = [value]
            
            # Collect all values for this prefix
            for seq_idx, pos in indices_list[1:]:
                current_value = tensor[seq_idx, pos]
                values.append(current_value)
                
                if debug:
                    # Check if values are the same
                    diff = torch.max(torch.abs(value - current_value)).item()
                    
                    if diff > tolerance:
                        if prefix not in inconsistent_prefixes:
                            inconsistent_prefixes.append(prefix)
                        
                        current_max_diff = value_diffs.get(prefix, 0)
                        value_diffs[prefix] = max(current_max_diff, diff)
                
                count += 1
            
            # Aggregate the values
            value = aggregation_fn(values)
        
        # Store the unique value
        unique_data[prefix] = (value, count)
    
    # Print debug info if requested
    if debug:
        if inconsistent_prefixes:
            print(f"WARNING: Found {len(inconsistent_prefixes)} prefixes with inconsistent values!")
            print(f"Max differences observed:")
            for prefix in sorted(inconsistent_prefixes, key=lambda p: value_diffs[p], reverse=True)[:10]:
                print(f"  Prefix {prefix}: max diff = {value_diffs[prefix]}, seen {unique_data[prefix][1]} times")
        else:
            print("All prefixes have consistent values (within tolerance).")
            
        total_saved = sum(unique_data[p][1] - 1 for p in unique_data)
        total_items = sum(len(indices) for indices in prefix_to_indices.values())
        print(f"Deduplication saved {total_saved} items out of {total_items}")
    
    # Convert to tensor
    unique_prefixes = list(unique_data.keys())
    unique_values = torch.stack([unique_data[p][0] for p in unique_prefixes])
    
    return unique_values, unique_prefixes

def get_nn_type(run_id):
    if 'GRU' in run_id or 'LSTM' in run_id or 'RNN' in run_id or 're' in run_id:
        return "RNN"
    else:
        return "transformer"


def deduplicate_data(inputs, probs, beliefs):
    """
    Deduplicate inputs, probs, and beliefs based on duplicate prefixes.
    
    Args:
        inputs: Input tensor to find duplicate prefixes
        probs: Probability tensor to deduplicate
        beliefs: Belief tensor to deduplicate
        
    Returns:
        tuple: (deduplicated probs, deduplicated beliefs, deduplicated indices)
    """
    prefix_to_indices = find_duplicate_prefixes(inputs)
    dedup_probs, dedup_indices = deduplicate_tensor(prefix_to_indices, probs, aggregation_fn=sum)
    # normalize the probs to sum to 1
    dedup_probs = dedup_probs / dedup_probs.sum()
    dedup_beliefs, _ = deduplicate_tensor(prefix_to_indices, beliefs, aggregation_fn=None)
    return dedup_probs, dedup_beliefs, dedup_indices, prefix_to_indices

def calculate_weighted_pca_variance(activations: np.ndarray, weights: np.ndarray) -> tuple:
    """
    Performs weighted PCA and calculates explained variance ratios.

    Args:
        activations (np.ndarray): The data array (e.g., deduplicated activations) 
                                  of shape (N, D), where N is number of samples 
                                  and D is number of features.
        weights (np.ndarray): The weights array (e.g., deduplicated probabilities) 
                              of shape (N,). Weights should be non-negative.

    Returns:
        tuple: A tuple containing:
            - cumulative_explained_variance (np.ndarray | None): 
                Array of cumulative explained variance (shape D,). Returns None if PCA fails.
            - explained_variance_ratio (np.ndarray | None): 
                Array of explained variance ratio per component (shape D,). Returns None if PCA fails.
            - sorted_eigenvalues (np.ndarray | None): 
                Array of sorted eigenvalues (shape D,). Returns None if PCA fails.
    """
    if activations.shape[0] != weights.shape[0]:
        raise ValueError(f"Number of samples mismatch: activations ({activations.shape[0]}) vs weights ({weights.shape[0]})")
    if np.any(weights < 0):
        warnings.warn("Input weights contain negative values.", RuntimeWarning)
    if not np.isclose(np.sum(weights), 1.0):
        warnings.warn("Input weights do not sum close to 1. Normalizing weights for PCA.", RuntimeWarning)
        weights = weights / np.sum(weights)
        # Ensure no division by zero if sum was zero
        if np.isnan(weights).any():
             weights = np.ones_like(weights) / weights.shape[0]


    try:
        # 1. Calculate weighted mean
        weighted_mean_acts = np.average(activations, axis=0, weights=weights)
        
        # 2. Center the activation data
        centered_acts = activations - weighted_mean_acts
        
        # 3. Compute the weighted covariance matrix
        # Use bias=True for population estimate, typical for PCA
        # Use ddof=0 explicitly which is equivalent to bias=True for np.cov
        cov_matrix = np.cov(centered_acts, rowvar=False, aweights=weights, ddof=0) 
        
        # 4. Eigen decomposition (use eigh for symmetric matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Check for complex eigenvalues (shouldn't happen for covariance, but safety check)
        if np.iscomplexobj(eigenvalues):
            warnings.warn("Complex eigenvalues encountered. Taking real part.", RuntimeWarning)
            eigenvalues = eigenvalues.real
        
        # Check for negative eigenvalues (can happen due to numerical instability)
        if np.any(eigenvalues < -1e-10): # Allow for small negative noise
            negative_count = np.sum(eigenvalues < -1e-10)
            warnings.warn(f"{negative_count} negative eigenvalues encountered. Setting them to zero.", RuntimeWarning)
            eigenvalues[eigenvalues < 0] = 0

        # 5. Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        # sorted_eigenvectors = eigenvectors[:, sorted_indices] # Eigenvectors not always needed
        
        # 6. Calculate explained variance ratio
        total_variance = np.sum(sorted_eigenvalues)
        
        if total_variance <= 1e-10: # Handle case of zero variance
             warnings.warn("Total variance is close to zero. Explained variance cannot be computed.", RuntimeWarning)
             explained_variance_ratio = np.zeros_like(sorted_eigenvalues)
             cumulative_explained_variance = np.zeros_like(sorted_eigenvalues)
        else:
            explained_variance_ratio = sorted_eigenvalues / total_variance
            # 7. Calculate cumulative explained variance
            cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        return cumulative_explained_variance, explained_variance_ratio, sorted_eigenvalues

    except np.linalg.LinAlgError as e:
        warnings.warn(f"Weighted PCA failed due to LinAlgError: {e}. Returning None.", RuntimeWarning)
        return None, None, None
    except Exception as e:
         warnings.warn(f"Unexpected error during weighted PCA: {e}. Returning None.", RuntimeWarning)
         return None, None, None
    
import collections # Or 'from collections import defaultdict' if not already done

def nested_dict_factory():
    """Returns a defaultdict that defaults to a regular dictionary."""
    return collections.defaultdict(dict)

def compute_kfold_split(flat_probs, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    from sklearn.model_selection import KFold
    import numpy as np

    # If flat_probs is a PyTorch tensor, convert it to numpy
    if isinstance(flat_probs, torch.Tensor):
        flat_probs = flat_probs.cpu().detach().numpy()
    
    # Create position indices
    all_positions = np.arange(len(flat_probs))
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    
    # Return the KFold object and positions
    # kf is a list of tuples, each tuple contains two lists: the indices of the training set and the indices of the test set
    return kf, all_positions

for sweep, run_id_int in sweep_run_pairs:
    run_dir = f'{output_dir}/{sweep}_{run_id_int}'
    os.makedirs(run_dir, exist_ok=True) # Create the directory if it doesn't exist

    runs = s3_loader.list_runs_in_sweep(sweep)
    # keep the entry of run that has f'run_{run_id_int}' in it
    run_id = [x for x in runs if f'run_{run_id_int}' in x][0]

    print(run_id)
    ckpts = s3_loader.list_checkpoints(sweep, run_id)
    model, run_config = s3_loader.load_checkpoint(sweep, run_id, ckpts[0])


    loss_df = s3_loader.load_loss_from_run(sweep, run_id)

    n_ctx = run_config["model_config"]["n_ctx"]
    run_config["n_ctx"] = n_ctx
    nn_inputs, nn_beliefs, _, nn_probs, _ = prepare_msp_data(
        run_config, run_config["process_config"]
    )

    classical_beliefs = belief_generator.generate_classical_belief_states(
    run_config, max_order=3)

    classical_nn_inputs = classical_beliefs['markov_order_3']['inputs']
    classical_nn_beliefs = classical_beliefs['markov_order_3']['beliefs']
    classical_nn_probs = classical_beliefs['markov_order_3']['probs']
    
    # Deduplicate neural network data
    dedup_probs, dedup_beliefs, dedup_indices, prefix_to_indices = deduplicate_data(
        nn_inputs, 
        nn_probs, 
        nn_beliefs
    )

    kf, all_positions = compute_kfold_split(dedup_probs)
    kf_list = list(kf.split(all_positions))
    
    # Deduplicate classical model data
    classical_dedup_probs, classical_dedup_beliefs, classical_dedup_indices, classical_prefix_to_indices = deduplicate_data(
        classical_nn_inputs, 
        classical_nn_probs, 
        classical_nn_beliefs
    )

    classical_kf, classical_all_positions = compute_kfold_split(classical_dedup_probs)
    classical_kf_list = list(classical_kf.split(classical_all_positions))

    ground_truth_data = defaultdict(dict) # Keys: ckpt -> layer -> predictions
    ground_truth_data['probs'] = dedup_probs.cpu().numpy() if torch.is_tensor(dedup_probs) else np.array(dedup_probs)
    ground_truth_data['beliefs'] = dedup_beliefs.cpu().numpy() if torch.is_tensor(dedup_beliefs) else np.array(dedup_beliefs)
    ground_truth_data['indices'] = np.array(dedup_indices, dtype=object)
    joblib.dump(ground_truth_data, f'{run_dir}/ground_truth_data.joblib')

    classical_ground_truth_data = defaultdict(dict) # Keys: ckpt -> layer -> predictions
    classical_ground_truth_data['probs'] = classical_dedup_probs.cpu().numpy()
    classical_ground_truth_data['beliefs'] = classical_dedup_beliefs.cpu().numpy()
    classical_ground_truth_data['indices'] = np.array(classical_dedup_indices, dtype=object)
    joblib.dump(classical_ground_truth_data, f'{run_dir}/markov3_ground_truth_data.joblib')
    
    checkpoints = s3_loader.list_checkpoints(sweep, run_id)

    if ONLY_INITIAL_AND_FINAL:
        selected_checkpoints = [checkpoints[0], checkpoints[-1]]
        selected_epochs = [0, len(checkpoints) - 1]
        print(f"Processing {len(selected_checkpoints)} checkpoints: first and last")
    else:
        # Process all checkpoints
        selected_checkpoints = checkpoints
        selected_epochs = list(range(len(checkpoints)))
        print(f"Processing all {len(selected_checkpoints)} checkpoints")


    for i, (epoch, ckpt) in enumerate(zip(selected_epochs, selected_checkpoints)):
        print(f"Processing checkpoint {i+1}/{len(selected_checkpoints)}: {ckpt} (epoch {epoch})")
        model, run_config = s3_loader.load_checkpoint(sweep, run_id, ckpt)

        #ckpt_ind is between / and .pt
        ckpt_ind = ckpt.split('/')[-1].split('.')[0]
        # we want the value of 'val_loss_mean' where num_tokens_seen == ckpt_ind
        try:
            filtered_df = loss_df[loss_df['epoch'] == epoch-1]
            if len(filtered_df) > 0 and 'val_loss_mean' in filtered_df.columns:
                val_loss_mean = filtered_df['val_loss_mean'].values[0]
            else:
                val_loss_mean = float('nan')
        except (KeyError, IndexError, AttributeError):
            val_loss_mean = float('nan')
        
        act_extractor = ActivationExtractor(device=DEVICE)
        nn_acts_ = act_extractor.extract_activations(
            model,
            nn_inputs,
            get_nn_type(run_id),
            relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
        )
        nn_acts = {}
        for layer, acts in nn_acts_.items():
            nn_acts[layer] = acts
        nn_acts['combined'] = _combine_layer_activations(nn_acts)

        classical_acts_ = act_extractor.extract_activations(
            model,
            classical_nn_inputs,
            get_nn_type(run_id),
            relevant_activation_keys=TRANSFORMER_ACTIVATION_KEYS,
        )
        classical_acts = {}
        for layer, acts in classical_acts_.items():
            classical_acts[layer] = acts
        classical_acts['combined'] = _combine_layer_activations(classical_acts)
        
        save_data = collections.defaultdict(nested_dict_factory) # Use the named function here
        classical_save_data = collections.defaultdict(nested_dict_factory) # Use the named function here
        
        for layer, act in nn_acts.items():
            
            # dedup the activations
            dedup_acts, dedup_indices = deduplicate_tensor(prefix_to_indices, act, aggregation_fn=None)

            classical_dedup_acts, classical_dedup_indices = deduplicate_tensor(classical_prefix_to_indices, classical_acts[layer], aggregation_fn=None)
            
            zscore_acts = (dedup_acts.numpy() - dedup_acts.numpy().mean(axis=0)) / dedup_acts.numpy().std(axis=0)
            cum_var_exp, _, _ = calculate_weighted_pca_variance(dedup_acts.numpy(), dedup_probs.numpy())
            cum_var_exp_zscore, _, _ = calculate_weighted_pca_variance(zscore_acts, dedup_probs.numpy())

            classical_zscore_acts = (classical_dedup_acts.numpy() - classical_dedup_acts.numpy().mean(axis=0)) / classical_dedup_acts.numpy().std(axis=0)
            classical_cum_var_exp, _, _ = calculate_weighted_pca_variance(classical_dedup_acts.numpy(), classical_dedup_probs.numpy())
            classical_cum_var_exp_zscore, _, _ = calculate_weighted_pca_variance(classical_zscore_acts, classical_dedup_probs.numpy())

            # Move tensors to configured device
            device = torch.device(DEVICE)
            results = run_activation_to_beliefs_regression_kf(
                reg_analyzer,
                dedup_acts.to(device),
                dedup_beliefs.to(device),
                dedup_probs.to(device),
                kf_list,
                rcond_values=RCOND_SWEEP_LIST,
            )

            classical_results = run_activation_to_beliefs_regression_kf(
                reg_analyzer,
                classical_dedup_acts.to(device),
                classical_dedup_beliefs.to(device),
                classical_dedup_probs.to(device),
                classical_kf_list,
                rcond_values=RCOND_SWEEP_LIST,
            )

            # Calculate Euclidean distance weighted by probabilities
            save_data[layer]['predicted_beliefs'] = results['predictions']
            save_data[layer]['rmse'] = results['final_metrics']['rmse']
            save_data[layer]['mae'] = results['final_metrics']['mae']
            save_data[layer]['r2'] = results['final_metrics']['r2']
            save_data[layer]['dist'] = results['final_metrics']['dist']
            save_data[layer]['mse'] = results['final_metrics']['mse']
            save_data[layer]['cum_var_exp'] = cum_var_exp
            save_data[layer]['cum_var_exp_zscore'] = cum_var_exp_zscore
            save_data[layer]['val_loss_mean'] = val_loss_mean

            # Calculate Euclidean distance weighted by probabilities
            
            # Only save predictions for epoch 0 or final epoch to save space
            if epoch == 0 or epoch == len(checkpoints) - 1:
                classical_save_data[layer]['predicted_beliefs'] = classical_results['predictions']
            else:
                classical_save_data[layer]['predicted_beliefs'] = None  # Skip storing predictions for intermediate epochs
            classical_save_data[layer]['rmse'] = classical_results['final_metrics']['rmse']
            classical_save_data[layer]['mae'] = classical_results['final_metrics']['mae']
            classical_save_data[layer]['r2'] = classical_results['final_metrics']['r2']
            classical_save_data[layer]['dist'] = classical_results['final_metrics']['dist']
            classical_save_data[layer]['mse'] = classical_results['final_metrics']['mse']
            classical_save_data[layer]['cum_var_exp'] = classical_cum_var_exp
            classical_save_data[layer]['cum_var_exp_zscore'] = classical_cum_var_exp_zscore
            classical_save_data[layer]['val_loss_mean'] = val_loss_mean
        
        # Save each checkpoint's data to a separate file
        joblib.dump(save_data, f'{run_dir}/checkpoint_{ckpt_ind}.joblib')
        joblib.dump(classical_save_data, f'{run_dir}/markov3_checkpoint_{ckpt_ind}.joblib')
