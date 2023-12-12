## Introduction

This repository replicates the result of this paper https://openreview.net/forum?id=bgIZDxd2bM

Deep generative models are broadly implemented and applied for reconstruction, generation, and representation tasks. Diffusion models, in particular, achieved amazing results in the image generation, while left insufficiently studied in natural language processing. Joint Autoencoding Diffusion (JEDI) combines Variational Autoencoders (VAEs) with diffusion models to achieve data reconstruction, generation, and representation all in one. We replicated the application of JEDI in text reconstruction and generation and named this specific branch of application JEDI-TEXT. JEDI-TEXT realized a Bleu score of 94 in text reconstruction and a perplexity of 23.4 in generation. It creates a text-diffusion framework with high versatility to various dataset and downstream tasks.

## To replicate the results
$\color{red} \text{Reminder: This project requires space for checkpoints and pretrained encoder/decoder.}$
$\color{red} \text{Please make sure you are in a workplace with enough space.}$

1) Clone this github into your workspace
2) In order, run \
```bash DSC180A_JEDI/build_envs.sh``` \
The line above will take a relatively long time to finish because it will run the setup.py for apex. \
```bash DSC180A_JEDI/download_classifiers.sh``` \
```bash DSC180A_JEDI/download_pretrained_models.sh``` \
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
4) Activate environment by running ```source .venv/bin/activate``` 
5) For __training__ new model, cd into DSC180A_JEDI directory and run 
```bash train.sh``` \
It will automatically train the model and store the checkpoints in the output directory specify in train.sh. Make sure to __change the ```output_dir``` to the  path of the folder you want to store the model__. By defualt, a __JEDI_output/LM__ folder will be created in your workspace. The folder will also contains the the evaluation result on the test set. \
If you have multiple GPU device, set ``CUDA_VISIBLE_DEVICE`` to the corresponding one you will use. By default it is set to 0. 
7) To simply evaluate preexisting checkpoint which is located in ```ckpts``` folder, you can run \
```bash evaluate.sh``` \
Make sure to __change the ```output_dir``` in ```evaluate.sh``` to be the path of the location you want to store the results__. By default a __JEDI_output/experiment__ folder will be created in your workspace. It will contains two ```.txt``` files. The first file will contains statistics that shows the performance of the model in text reconstruction. The second file will contains the generated sentences produce by the model. \
If you have multiple GPU device, set ``CUDA_VISIBLE_DEVICE`` to the corresponding one you will use. By default it is set to 0.
