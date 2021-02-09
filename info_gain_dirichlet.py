#!/usr/bin/env python3
"""Approximate information gain using a cooccurrence matrix to create a Dirichlet distribution over transition matrices."""

import pickle
import sys
import numpy as np
from scipy.special import digamma, loggamma

def log_beta(alpha_non_one, num_ones, alpha_sum):
    """The logarithm of the Beta function in the definition of the Dirichlet entropy function."""
    return np.sum(loggamma(alpha_non_one)) + (loggamma(1) * num_ones) - loggamma(alpha_sum)

def dirichlet_entropy(alpha_minus_one, size):
    """The entropy of a Dirichlet distribution."""
    nonzero_indices = list(alpha_minus_one.keys())
    num_ones = size - len(nonzero_indices)
    alpha_non_one = np.array([alpha_minus_one[i] for i in nonzero_indices]) + 1
    alpha_sum = alpha_non_one.sum() + num_ones
    num_categories = alpha_non_one.shape[0] + num_ones
    entropy = np.real(log_beta(alpha_non_one, num_ones, alpha_sum))
    entropy += np.real((alpha_sum - num_categories) * digamma(alpha_sum))
    entropy -= np.real(np.dot(alpha_non_one - 1, digamma(alpha_non_one)))
    return entropy

with open(sys.argv[1], 'rb') as f:
    cooccurrence_counts = pickle.load(f)
obs_matrix = {}

# These are the total numbers of unique observations and actions for each environment; add your own entries for new environments

max_obs = int(sys.argv[2]) - 1

max_act = int(sys.argv[3]) - 1

total_transitions = 0
for (obs, act, next_obs), count in cooccurrence_counts.items():
    total_transitions += count

    key = (obs, act)
    if key not in obs_matrix:
        obs_matrix[key] = {}
    if next_obs not in obs_matrix[key]:
        obs_matrix[key][next_obs] = 0
    obs_matrix[key][next_obs] += count

new_total_transitions = 100_000_000
actual_transitions = 0
for key in obs_matrix:
    for next_obs in obs_matrix[key]:
        obs_matrix[key][next_obs] = int(obs_matrix[key][next_obs] > 0)
        actual_transitions += 1

initial_row_entropy = dirichlet_entropy({}, max_obs + 1)
total_info_gain = sum(initial_row_entropy - dirichlet_entropy(row, max_obs + 1) for row in obs_matrix.values())
total_prior = initial_row_entropy * (max_obs + 1) * (max_act + 1)
total_posterior = sum((dirichlet_entropy(obs_matrix[obs, act], max_obs + 1) if (obs, act) in obs_matrix else initial_row_entropy) for obs in range(max_obs + 1) for act in range(max_act + 1))
print('Information gain:', total_info_gain / new_total_transitions)
