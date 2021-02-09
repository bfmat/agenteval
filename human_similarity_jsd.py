#!/usr/bin/env python3
"""Given a transition matrix from human data and from an agent, calculate the Jensen-Shannon divergence."""

import pickle
import sys
from pathlib import Path
import numpy as np

# Calculate the total occurrences of each observation and each observation pair for the human and agent (using a list for total occurrences so it is iterated by reference)
human_total_occurrences = [0]
human_obs_occurrences = {}
human_obs_act_occurrences = {}
human_obs_pair_occurrences = {}
human_data = (Path(sys.argv[1]), human_total_occurrences, human_obs_occurrences, human_obs_act_occurrences, human_obs_pair_occurrences)
agent_total_occurrences = [0]
agent_obs_occurrences = {}
agent_obs_act_occurrences = {}
agent_obs_pair_occurrences = {}
agent_data = (Path(sys.argv[2]), agent_total_occurrences, agent_obs_occurrences, agent_obs_act_occurrences, agent_obs_pair_occurrences)
for matrix_path, total_occurrences, obs_occurrences, obs_act_occurrences, obs_pair_occurrences in [human_data, agent_data]:
    # Load the combined matrix from disk
    with open(matrix_path, 'rb') as f:
        cooccurrence_counts, scalar_to_index = pickle.load(f)
    # Invert the scalar to index dictionary, but leave 0 as 0 (because index 0 corresponds to many scalars and occurs only in the agent matrix)
    index_to_scalar = {0: 0}
    for scalar, index in scalar_to_index.items():
        if index != 0:
            index_to_scalar[index] = scalar
    # Add up the counts for all observations, observation-action pairs, and observation-observation pairs
    for (obs_index, act, next_obs_index), count in cooccurrence_counts.items():
        obs = index_to_scalar[obs_index]
        next_obs = index_to_scalar[next_obs_index]
        # Initialize the dictionary values that are not present; the "in" operator does not work correctly for defaultdict
        if obs not in obs_occurrences:
            obs_occurrences[obs] = 0
        if (obs, act) not in obs_act_occurrences:
            obs_act_occurrences[obs, act] = 0
        if (obs, next_obs) not in obs_pair_occurrences:
            obs_pair_occurrences[obs, next_obs] = 0
        obs_occurrences[obs] += count
        obs_act_occurrences[obs, act] += count
        obs_pair_occurrences[obs, next_obs] += count
        total_occurrences[0] += count

human_obs = set(human_obs_occurrences.keys())
agent_obs = set(agent_obs_occurrences.keys())
overlapping_obs = human_obs & agent_obs

# Add up the Jensen-Shannon divergence, which is the mean Kullback-Leibler divergence of each of the two distributions from the mean distribution
obs_marginal = 0
act_given_obs = 0
next_obs_given_obs = 0
for (matrix_path, [total_occurrences], obs_occurrences, obs_act_occurrences, obs_pair_occurrences), (other_matrix_path, [other_total_occurrences], other_obs_occurrences, other_obs_act_occurrences, other_obs_pair_occurrences) in [(human_data, agent_data), (agent_data, human_data)]:

    # Compute and print the proportion of observations in obs_occurrences that are also in other_obs_occurrences (weighted by the observation counts)
    overlapping_occurrences = 0
    for obs, count in obs_occurrences.items():
        if obs in other_obs_occurrences:
            overlapping_occurrences += count
    print(f'Proportion of observations in {matrix_path.stem} that are also in {other_matrix_path.stem}: {overlapping_occurrences / total_occurrences}')

    # H(O)
    for obs in obs_occurrences:
        prob = obs_occurrences[obs] / total_occurrences
        other_prob = (other_obs_occurrences[obs] / other_total_occurrences) if obs in other_obs_occurrences else 0
        # Calculate the probability under the mean of the two distributions
        mixed_prob = (prob + other_prob) / 2
        obs_marginal += 0.5 * prob * (np.log(prob) - np.log(mixed_prob))

    # H(A|O)
    for obs, act in obs_act_occurrences:
        prob = obs_act_occurrences[obs, act] / obs_occurrences[obs]
        # The observation might not occur, so just use 0 if it doesn't
        other_prob = (other_obs_act_occurrences[obs, act] / other_obs_occurrences[obs]) if (obs, act) in other_obs_act_occurrences else 0
        mixed_prob = (prob + other_prob) / 2
        # Multiply the result by the prior probability of the observation
        obs_prob = obs_occurrences[obs] / total_occurrences
        act_given_obs += 0.5 * prob * (np.log(prob) - np.log(mixed_prob)) * obs_prob

    # H(O'|O)
    for obs, next_obs in obs_pair_occurrences:
        prob = obs_pair_occurrences[obs, next_obs] / obs_occurrences[obs]
        other_prob = (other_obs_pair_occurrences[obs, next_obs] / other_obs_occurrences[obs]) if (obs, next_obs) in other_obs_pair_occurrences else 0
        mixed_prob = (prob + other_prob) / 2
        obs_prob = obs_occurrences[obs] / total_occurrences
        next_obs_given_obs += 0.5 * prob * (np.log(prob) - np.log(mixed_prob)) * obs_prob

print('JSD human similarity:', obs_marginal)
