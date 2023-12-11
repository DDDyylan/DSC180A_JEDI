## Introduction

This repository replicates the result of this paper https://openreview.net/forum?id=bgIZDxd2bM

## To replicate the results
1) Clone this github into your workspace
2) In order, run \
```bash DSC180A_JEDI/build_envs.sh``` \
```bash DSC180A_JEDI/download_classifiers.sh```
```bash DSC180A_JEDI/download_pretrained_models.sh```\
This will create four folders in your workspace. \
   ```.venv``` is the folder containing the virtual environment with all required dependencies. ```apex``` contains the necessary packages for distributed learning. ```classifiers``` and ```ckpts``` stored weights \
   for pretrained models
   The overall structure will look like the following:
   ```
   workspace
   |--.venv
   |--apex
   |--ckpts
   |--classifiers
   |--DSC180A_JEDI
   |   |--src
   ```
3) Activate environment by running ```source .venv/bin/activate```
4) For training new model, cd into DSC180A_JEDI directory and run ```bash train.sh```. It will automatically train the model and store the checkpoints in the output directory specify in train.sh. Make sure to change the ```output_dir``` to the absolute path of the folder you want to store the model. The folder will also contains the the evaluation result on the test set.
5) To simply evaluate preexisting checkpoint which is located in ```ckpts``` folder, you can run ```bash evaluate.sh```. Make sure to change the ```output_dir``` in ```evaluate.sh``` to be the absolute path of the location you want to store the results. Also make sure to change ```checkpoint_dir``` to the absolute or relative path where you want to store the evaluation results. It will contains two ```.txt``` files. The first file will contains statistics that shows the performance of the model in text reconstruction. The second file will contains the generated sentences produce by the model.
