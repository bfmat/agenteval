# Evaluating Agents without Rewards

Source code for the research paper [Evaluating Agents without Rewards][paper].
For more information, visit the [project website][website]. This repository
contains instructions for using the dataset, computing the objective functions,
and for creating new datasets.

[website]: https://danijar.com/project/agenteval
[paper]: https://arxiv.org/pdf/2012.11538.pdf

If you find this project useful, please cite our paper:

```
@article{matusch2020agenteval,
  title={Evaluating Agents without Rewards},
  author={Matusch, Brendon and Ba, Jimmy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2012.11538},
  year={2020}
}
```

## Use our Dataset

The repository contains two of the transition tables of our study in the
`transition_tables` directory as examples. All transition tables are available
for download. The file names use the format
`{env}_{agent}_{reward,noreward}_{human,shared}_disc.pkl`. The files with human
discretization levels are used for computing the human similarity metric. The
shared discretization levels were computed as percentiles across all agents for
the respective environment.

Even though they are not required for our study, we also offer the downscaled
episodes in NPZ format and the original videos in MP4 format for download. The
NPZ files can be loaded via `np.load(filename)` and contain the keys `observ`
that holds the `(T, 8, 8)` downscaled by not-yet-discretized images, `action`
that hols the `(T,)` array of action indices, and `reward` that holds the
`(T,)` array of rewards.

[Dataset download][data] (121 GB)

[data]: https://archive.org/download/agenteval

## Compute the Objectives

The objectives are computed from the transition tables. Those can be created
from manually collected data as described below or downloaded for the agents
and environments used in our paper.

To reproduce results in our study, download the `transition_tables` directory
[here][data], move the contents to the `transition_tables` directory in
this repository, and run:

```sh
./compute_all_metrics.sh
```

Input entropy measures how spread out the agent's visitation distribution over
inputs is. It increases if the agent observes more inputs, or visits them more
uniformly.

```sh
python3 input_entropy.py transition_tables/montezuma_noop_shared_disc.pkl
```

Information gain measures how many bits of information the agent has learned
about the environment from its experience. Computing the Dirichlet information
gain used in our paper requires knowing the total number of unique observations
in the environment across agents. The two alternative information gain
approximations are also included.

You must include the numbers of unique observations and actions as the second
and third arguments to `info_gain_dirichlet.py`, since those determine the size
of the Dirichlet distribution.

```sh
python3 info_gain_dirichlet.py transition_table.pkl $num_obs $num_act
python3 info_gain_sqrt_counts.py transition_table.pkl
python3 info_gain_log_counts.py transition_table.pkl
```

Empowerment measures the agent's influence over its sensory inputs and thus the
environment. We define it as the mutual information between an action and the
input it leads to, given the last input. This is computed as the difference
between the entropy of the action given the preceding input, before and after
observing the successor input.

```sh
python3 empowerment.py transition_table.pkl
```

Human similarity measures the degree of shared coverage of the environment
between an artificial agent and a human agent. We compute it as the Jaccard
index, also known as intersection over union, between the unique images
encountered by the human and agent. The two transition tables need to use the
same discretization thresholds. In our study, we use the `*_human_disc.pkl`
files for this. We also include an alternative human similarity objective based
on the Jensen-Shannon divergence.

```sh
python3 human_similarity_jaccard.py transition_table_0.pkl transition_table_1.pkl
python3 human_similarity_jsd.py transition_table_0.pkl transition_table_1.pkl
```

## Create new Datasets

The section describes how to preprocess your own agent experience to create a
new dataset for computing our objectives. The process consists of the 3 steps
described below, namely discretizing the images, index encoding them, and
creating the transition tables.

First, discretize the images into numeric encodings using the code snippeted
below. The result is a Python list rather than a NumPy array to make use of
Python's big integer support. If you need to convert the list to a NumPy array,
be sure to use a data type that can represent large enough integers.

```python
images = np.zeros((1000, 210, 160), np.uint8)  # Any height and width works
downscaled = downscale(images, 8)              # (1000, 8, 8) np.float32
discretized = discretize(downscaled, 4)        # (1000, 8, 8) np.int32
encodings = encode(discretized, 4)             # List of Python big integers
```

Alternatively, you can also discretize a set of images using the discretization
thresholds from another agent or set of agents. This is how we generated the
human-discretized transition tables for computing human similarity.

```python
human_discretized, bins = discretize(downscale(human_images, 8), 4, return_bins=True)
agent_discretized = discretize(downscale(agent_images, 8), color_bins=bins)
```

Second, index encode the discretized images. You can use the following helper
function for this.

```python
indices = dense_indices(encodings)
```

Third, create the transition tables. The format for transition tables is a
pickled dictionary that maps tuples of current image index, action, and next
image index to a count of how often this transition occurred. Transitions that
never occur should not be included to keep the file size manageable. For
example:

```python
transition_table = {
  (1524, 4, 56213): 16,  # Agent went 16 times from image 1524 with action 4 to image 56213.
  ...
}
```

The resulting pickle file can be used as input to the scripts that compute the
different objectives.

## Dependencies

The required dependencies are `numpy`, `scipy` and
[`sparse`](https://github.com/pydata/sparse).
