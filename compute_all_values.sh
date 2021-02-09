#!/bin/sh

set -e
declare -a envs=("breakout" "seaquest" "montezuma" "minecraft")
declare -a obs_counts=(192911187 215436546 101509 37624305)
declare -a act_counts=(4 18 18 11)
declare -a agents=("noop" "random" "icm_noreward" "rnd_noreward" "ppo" "icm_reward" "rnd_reward")

for env_idx in ${!envs[@]}; do
    for agent_idx in ${!agents[@]}; do

        # Skip Minecraft with ICM
        if [ $env_idx == 3 ]
        then
            if [ $agent_idx == 2 ] || [ $agent_idx == 5 ]
            then
                continue
            fi
        fi

        echo ======================== ${envs[$env_idx]} ${agents[$agent_idx]} ========================
        ./input_entropy.py transition_tables/${envs[$env_idx]}_${agents[$agent_idx]}_shared_disc.pkl
        ./info_gain_dirichlet.py transition_tables/${envs[$env_idx]}_${agents[$agent_idx]}_shared_disc.pkl ${obs_counts[$env_idx]} ${act_counts[$env_idx]}
        ./empowerment.py transition_tables/${envs[$env_idx]}_${agents[$agent_idx]}_shared_disc.pkl
        ./human_similarity_jaccard.py transition_tables/${envs[$env_idx]}_human_human_disc.pkl transition_tables/${envs[$env_idx]}_${agents[$agent_idx]}_human_disc.pkl
        echo "\n\n\n"
    done
done
