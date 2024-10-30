<h1 align="center">S2S Precipitation Forecasting in the Mediterranean Area</h1>
This research project was conducted at the Institute for Environmental Studies from May to October as part of my end-of-studies internship for MVA and SDI Masters.

## I- Project Overview
### a. Description
This project focuses on **predicting precipitation during the cold season (October-March) in the Mediterranean region**, specifically at lead times of 1 to 4 weeks. 

The project aims to develop two distinct models to generate probabilistic forecasts for weekly precipitation:

1. **To build upon Earthformer**, a 3D convolutional network transformer, adapting it for S2S forecasting in the Mediterranean region and generating ensemble members through weight initialization.

2. **To develop a 3D Variational Autoencoder (VAE)**, leveraging its stochastic structure to generate ensembles by sampling from the latent space.

### b. Results


## II- Repo stucture
This repository is structured in 2 main codebase : Earthformer and VAE 




```
📦src
 ┣ 📂data
 ┃ ┣ 📜mediteranean_dataset.py
 ┃ ┗ 📜north_h_dataset.py
 ┣ 📂evaluation
 ┣ 📂model
 ┗ 📂utils
 ```



## III- Installation


```
python -m venv pyenv
pip install git+https://github.com/amazon-science/earth-forecasting-transformer.git
pip install torchmetrics
pip install torch torchvision torchaudio
pip install pytorch_lightining
pip install wandb
`````
