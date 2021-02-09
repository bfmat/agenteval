#!/usr/bin/env python3
"""Write mean(sqrt(counts)) to a file."""

import pickle
import sys
from pathlib import Path
import numpy as np

matrix_path = sys.argv[1]

with open(matrix_path, 'rb') as f:
    matrix = pickle.load(f)

obs_act_matrix = {}
for (obs, act, next_obs), count in matrix.items():
    key = (obs, act)
    if key not in obs_act_matrix:
        obs_act_matrix[key] = 0
    obs_act_matrix[key] += count

sqrt_sum = 0
total = 0
unique_obs = set()
for (obs, act), count in obs_act_matrix.items():
    unique_obs.add(obs)
    sqrt_sum += np.sqrt(count)
    total += count

num_obs = len(unique_obs)
sqrt_mean = sqrt_sum / total
print('Sqrt counts information gain:', sqrt_mean)
