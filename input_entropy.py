#!/usr/bin/env python3
"""Compute the entropy of the marginal of the observation transition matrix."""

import pickle
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

# Load the matrix and calculate the marginal occurrences of each image
with open(sys.argv[1], 'rb') as f:
    cooccurrence_counts = pickle.load(f)
obs_counts = {}
for (obs, _, _), count in cooccurrence_counts.items():
    if obs not in obs_counts:
        obs_counts[obs] = 0
    obs_counts[obs] += count
marginal = np.array(list(obs_counts.values()), dtype=float)
marginal /= marginal.sum()
log_marginal = np.log(marginal)
# Where marginal[x] == 0, set the log to 0 as well to avoid computing 0 * -inf
log_marginal[marginal == 0] = 0
entropy = -(marginal * log_marginal).sum()
print('Curiosity:', entropy)
