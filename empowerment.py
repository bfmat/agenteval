#!/usr/bin/env python3
"""Calculate n-step empowerment over the lifetime of an agent by multiplying together copies of the combined transition matrix."""

import pickle
import sys
import numpy as np
import sparse

with open(sys.argv[1], 'rb') as f:
    cooccurrence_counts = pickle.load(f)

# Get the keys and values from the dictionary and construct a sparse array
coords, values = [np.array(ls) for ls in zip(*cooccurrence_counts.items())]
obs_dim = coords[:, (0, 2)].max() + 1
act_dim = coords[:, 1].max() + 1
env_tensor = sparse.COO(coords.T, values.astype(np.float32), shape=(obs_dim, act_dim, obs_dim))
# Normalize it to a sum of 1
env_tensor /= env_tensor.sum()
# Get the policy tensor by adding up the unnormalized env tensor over the last axis
policy_tensor = env_tensor.sum(2)
# Normalize the env tensor over the next observation axis
sum_over_next_obs = env_tensor.sum(2, keepdims=True)
norm_array = sparse.COO(sum_over_next_obs.coords, 1 / sum_over_next_obs.data, sum_over_next_obs.shape)
next_obs_norm_env_tensor = env_tensor * norm_array

# Multiply the policy tensor by the env tensor multiple times to create an (o, a, a, ... a, o) tensor
n_step_tensor = policy_tensor
max_n = 1
for n in range(2, max_n + 1):
    # Expand the env tensor to have new dimensions at the end and the n-step tensor to have new dimensions at the beginning
    expanded_env_tensor = next_obs_norm_env_tensor.reshape(tuple(list(next_obs_norm_env_tensor.shape) + ([1] * (n - 1))))
    n_step_tensor = n_step_tensor.reshape(tuple([1, 1] + list(n_step_tensor.shape)))
    # Multiply the n-step tensor by the expanded env tensor and sum over the intermediate observation axis
    n_step_tensor = n_step_tensor * expanded_env_tensor
    print(f'Sparsity before sum: {n_step_tensor.nonzero()[0].size} nonzero elements out of {n_step_tensor.size}')
    n_step_tensor = n_step_tensor.sum(2)
    print(f'Sparsity after sum: {n_step_tensor.nonzero()[0].size} nonzero elements out of {n_step_tensor.size}')
    print(f'Shape of n-step tensor for n={n}: {n_step_tensor.shape}')

# Convert it to a dense array
n_step_tensor = n_step_tensor.todense()
# Calculate the mean log probs over the last action axis
last_act_sum = n_step_tensor.sum(-1, keepdims=True)
last_act_sum[last_act_sum == 0] = 1
last_act_probs = n_step_tensor / last_act_sum
last_act_log_probs = np.log(last_act_probs)
last_act_log_probs[last_act_probs == 0] = 0
mean_log_probs = (last_act_probs * last_act_log_probs).sum(-1)
# Multiply the mean log probs over the last action by the joint probs of the prior observation and action axes
prior_obs_act_probs = n_step_tensor.sum(-1) / n_step_tensor.sum()
# Calculate the overall marginal entropy
marginal_entropy = (mean_log_probs * prior_obs_act_probs).sum() * -1

# Calculate conditional entropy, which is H(A|O,O')
sum_over_act = env_tensor.sum(1, keepdims=True)
norm_array = sparse.COO(sum_over_act.coords, 1 / sum_over_act.data, shape=sum_over_act.shape)
act_probs = env_tensor * norm_array
# Calculate the mean log probs over the action axis
act_log_probs = sparse.COO(act_probs.coords, np.log(act_probs.data), act_probs.shape)
mean_log_probs = (act_probs * act_log_probs).sum(1, keepdims=True)
obs_probs = sum_over_act / sum_over_act.sum()
conditional_entropy = (mean_log_probs * obs_probs).sum() * -1
print('Empowerment:', marginal_entropy - conditional_entropy)
