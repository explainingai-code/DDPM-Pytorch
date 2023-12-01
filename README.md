Denoising Diffusion Probabilistic Models Implementation in Pytorch
========

This repository implements DDPM with training and sampling methods of DDPM and unet architecture mimicking the stable diffusion unet used in diffusers library from huggingface from scratch.

## Data preparation
For setting up the mnist dataset:

Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation


# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/explainingai-code/DDPM-Pytorch.git```
* ```cd DDPM-Pytorch```
* ```pip install -r requirements.txt```
* For training/sampling use the below commands passing the desired configuration file as the config argument in case you want to play with it. 
* ```python -m tools.train_ddpm``` for training ddpm
* ```python -m tools.sample_ddpm``` for generating images

## Configuration
* ```config/default.yaml``` - Allows you to play with different components of ddpm  


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of DDPM the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory

During sampling the following output will be saved
* Sampled image grid for all timesteps in ```task_name/samples/*.png``` 


## Sample Output for DDPM on Mnist






