#!/usr/bin/env python3
"""Test script to isolate where the bad_alloc error occurs."""
import sys
import torch
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import mess3 as _mess3_matrix

print("Step 1: Creating MM3 process...")
process = TransitionMatrixGHMM(_mess3_matrix(x=0.05, a=0.85))
print(f"Process created. Vocab len: {process.vocab_len}")

print("\nStep 2: Deriving mixed state tree...")
n_ctx = 4
seq_len = n_ctx + 1  # 5
bos = False
msp_depth = seq_len + (1 if bos else 2)  # 7
print(f"n_ctx={n_ctx}, seq_len={seq_len}, msp_depth={msp_depth}")

try:
    msp = process.derive_mixed_state_tree(depth=msp_depth)
    print("Mixed state tree created successfully!")
    print(f"Number of paths: {len(msp.paths)}")
except Exception as e:
    print(f"ERROR in derive_mixed_state_tree: {e}")
    sys.exit(1)

print("\nStep 3: Getting paths and probs...")
final_seq_len = seq_len - (1 if bos else 0)  # 5
try:
    paths, probs = msp.get_paths_and_probs(depth=final_seq_len)
    print(f"Got {len(paths)} paths")
except Exception as e:
    print(f"ERROR in get_paths_and_probs: {e}")
    sys.exit(1)

print("\nStep 4: Converting to tensors...")
try:
    transformer_inputs = torch.tensor(paths, dtype=torch.int32)
    print(f"Tensor shape: {transformer_inputs.shape}")
except Exception as e:
    print(f"ERROR converting to tensor: {e}")
    sys.exit(1)

print("\nAll steps completed successfully!")
