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
### a. Earthformer



```
📦 src
 ┣ 📂 configs
 ┣ 📂 data
 ┃ ┣ 📜 area_dataset.py
 ┃ ┣ 📜 dataset.py
 ┃ ┗ 📜 temporal_aggregator.py
 ┣ 📂 evaluation
 ┃ ┣ 📜 ensemble_eval.py
 ┃ ┗ 📜 test.py
 ┣ 📂 interpretability
 ┣ 📂 model
 ┃ ┣ 📂 experiments
 ┃ ┣ 📜 earthformer_model.py
 ┃ ┣ 📜 earthformer_prob_model.py
 ┣ 📂 nets
 ┃ ┗ 📜 cuboid_transformer.py
 ┣ 📂 s2s_evaluation
 ┗ 📂 utils
   ┣ 📜 __init__.py
   ┣ 📜 climatology.py
   ┣ 📜 entire_era.py
   ┣ 📜 enums.py
   ┣ 📜 hierarchical_aggregator.py
   ┣ 📜 spi.py
 ```

### b. VAE
```
📦src
 ┣ 📂data
 ┃  ┣ 📜 __init__.py
 ┃  ┣ 📜 ensemble_eval.py
 ┃  ┣ 📜 eval.py
 ┃  ┗ 📜 main.py
 ┣ 📂 model_vae
 ┃ ┣ 📜 __init__.py
 ┃ ┣ 📜 vae_bis.py         # Alternative VAE implementation/variant
 ┃ ┣ 📜 vae_model.py       # Core VAE model definition
 ┃ ┣ 📜 vae_net.py         # Neural network architecture for VAE
 ┃ ┗ 📜 vae_train_job.sh   # Training script for VAE
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
