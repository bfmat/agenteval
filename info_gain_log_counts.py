#!/usr/bin/env python3
"""Write mean(log(counts + 1)) to a file."""

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

log_sum = 0
unique_obs = set()
total = 0
for (obs, act), count in obs_act_matrix.items():
    unique_obs.add(obs)
    log_sum += np.log(count + 1)
    total += count

num_obs = len(unique_obs)
log_mean = log_sum / total
print('Log counts information gain:', log_mean)
