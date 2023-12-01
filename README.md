Denoising Diffusion Probabilistic Models Implementation in Pytorch
========

This repository implements DDPM with training and sampling methods of DDPM and unet architecture mimicking the stable diffusion unet used in diffusers library from huggingface from scratch.

## Sample Output by trained DDPM on Mnist

<img src="https://github.com/explainingai-code/DDPM-Pytorch/assets/144267687/a8095bc2-a525-40ad-a0ae-ec53da4145b5" width="300">


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


