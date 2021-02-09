#!/usr/bin/env python3
"""Compute overlap between two matrices with the same discretization scheme."""

import pickle
import sys

obs_occurrences = []
for path in sys.argv[1:]:
    with open(path, 'rb') as f:
        counts, scalar_to_index = pickle.load(f)
    index_to_scalar = {index: scalar for scalar, index in scalar_to_index.items()}
    occurrences = set()
    for (obs_index, _, _), count in counts.items():
        if count > 0:
            occurrences.add(index_to_scalar[obs_index])
    obs_occurrences.append(occurrences)
occurrences_0, occurrences_1 = obs_occurrences
intersection = len(occurrences_0 & occurrences_1)
print('Human similarity:', intersection / len(occurrences_0 | occurrences_1))
