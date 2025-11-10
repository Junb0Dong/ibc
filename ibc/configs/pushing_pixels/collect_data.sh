#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=500 \
 --policy=oracle_push \
 --task=PUSH \
 --dataset_path=ibc/data/test_dataset/simulation_pushing_state.zarr \
 --replicas=1  \
 --use_image_obs=False
