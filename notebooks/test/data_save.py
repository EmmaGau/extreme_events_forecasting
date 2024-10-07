
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np 
import torch 
import os
import sys

# Ajoutez le répertoire parent de 'data' au sys.path
sys.path.append(os.path.abspath("/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src"))

# Maintenant vous pouvez importer le module
from data.dataset import DatasetEra
from utils.temporal_aggregator import TemporalAggregatorFactory

import wandb 
import xclim


data_dirs = {'mediteranean': {'tp':"/home/egauillard/data/PR_era5_MED_1degr_19400101_20240229_new.nc",
                                  't2m':"/home/egauillard/data/T2M_era5_MED_1degr_19400101-20240229.nc"},
                 
                 'north_hemisphere': {"stream": "/home/egauillard/data/STREAM250_era5_NHExt_1degr_19400101_20240229_new.nc",
                                      "sst": "/home/egauillard/data/SST_era5_NHExt_1degr_19400101-20240229_new.nc",
                                      "msl": "/home/egauillard/data/MSLP_era5_NHExt_1degr_19400101_20240229_new.nc",
                                      },
                 'tropics': {"ttr": "/home/egauillard/data/OLR_era5_tropics_1degr_19400101_20240229.nc"}}

wandb_config = {
    'dataset': {
        'variables_tropics': ["ttr"],
        'variables_nh': ["stream", "msl", "sst"],
        'variables_med': ['tp', "t2m"],
        'target_variables': ['tp'],
        'relevant_years': [1940,1941],
        'relevant_months': [10,11,12,1,2,3],
        'scaling_years': [1940,2005],
        'land_sea_mask': '/home/egauillard/data/ERA5_land_sea_mask_1deg.nc',
        'spatial_resolution': 1,
        'predict_sea_land': False,
        'out_spatial_resolution': 1,
        'sum_pr': True,
        "coarse_t": True,
        "coarse_s": True,
        "coarse_t_target": True,
        "coarse_s_target": True
    },
    'scaler': {
        'mode': 'standardize',
    },
    'temporal_aggregator': {
        'in_len': 3,
        'out_len': 3,
        'resolution_input': 7,
        'resolution_output': 7,
        'gap': 1,
        'lead_time_gap': 0
    }
}



#coarse 
temp_aggregator_factory = TemporalAggregatorFactory(wandb_config['temporal_aggregator'])


train_dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory)

train_dataset.data.to_netcdf('coarse_dataset.nc')
train_dataset.target.to_netcdf('coarse_target.nc')

# # fine
wandb_config['dataset']['coarse_t'] = False
wandb_config['dataset']['coarse_s'] = False
wandb_config['dataset']['coarse_t_target'] = False
wandb_config['dataset']['coarse_s_target'] = False

train_dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory)
train_dataset.data.to_netcdf('fine_dataset.nc')
train_dataset.target.to_netcdf('fine_target.nc')



# coarse t fine s 
wandb_config['dataset']['coarse_t'] = True
wandb_config['dataset']['coarse_s'] = False
wandb_config['dataset']['coarse_t_target'] = True
wandb_config['dataset']['coarse_s_target'] = False

train_dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory)
train_dataset.data.to_netcdf('coarse_t_fine_s_dataset.nc')
train_dataset.target.to_netcdf('coarse_t_fine_s_target.nc')


    

    

