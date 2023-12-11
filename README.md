## Introduction

This repository replicates the result of this paper https://openreview.net/forum?id=bgIZDxd2bM

## To replicate the results
1) Clone this github into your workspace
2) In order, run ```bash DSC180A_JEDI/build_envs.sh```, ```bash DSC180A_JEDI/download_classifiers.sh```, ```bash DSC180A_JEDI/download_pretrained_models.sh```. This will create four folders in your workspace. \
   ```.venv``` is the folder containing the virtual environment with all required dependencies. ```apex``` contains the necessary packages for distributed learning. ```classifiers``` and ```ckpts``` stored weights \
   for pretrained models
3) For training new model, cd into DSC180A_JEDI directory and run ```bash train_vae_joint_split_data_DDP_yelp.sh```.
