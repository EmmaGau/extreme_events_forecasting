from data.dataset import DatasetEra
from data.temporal_aggregator import TemporalAggregatorFactory
import xarray as xr

"""This script is used to save the entire era5 dataset (1940-2023) for the precipitations in the mediteranean region.
It is used to compute the terciles for the probabilistic evaulation of the model."""

if __name__ == "__main__":
    data_dirs = {'mediteranean': {'tp':"/home/egauillard/data/PR_era5_MED_1degr_19400101_20240229_new.nc",
                                    't2m':"/home/egauillard/data/T2M_era5_MED_1degr_19400101-20240229.nc"},
                    
                    'north_hemisphere': {"stream": "/home/egauillard/data/STREAM250_era5_NHExt_1degr_19400101_20240229_new.nc",
                                        "sst": "/home/egauillard/data/SST_era5_NHExt_1degr_19400101-20240229_new.nc",
                                        "msl": "/home/egauillard/data/MSLP_era5_NHExt_1degr_19400101_20240229_new.nc",
                                        },
                    'tropics': {"ttr": "/home/egauillard/data/OLR_era5_tropics_1degr_19400101_20240229.nc"}}

    wandb_config = {
    'dataset': {
        'variables_tropics': [],
        'variables_nh': [],
        'variables_med': ['tp','t2m'],
        'target_variables': ['tp'],
        'relevant_years': [1940,2023],
        'relevant_months': [10,11,12,1,2,3],
        'scaling_years': [1940,2005],
        'land_sea_mask': '/home/egauillard/data/ERA5_land_sea_mask_1deg.nc',
        'spatial_resolution': 1,
        'predict_sea_land': False,
        'out_spatial_resolution': 10,
        'sum_pr': True,
        "coarse_t":False,
        "coarse_s": False,
        "coarse_t_target": False,
        "coarse_s_target": False
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
    # Initialize dataset and dataloaders
    temp_aggregator_factory = TemporalAggregatorFactory(wandb_config['temporal_aggregator'])

    # save 10deg  resolution 
    wandb_config['dataset']['out_spatial_resolution'] = 10
    dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory)
    target_10 = dataset.target_class.data.to_netcdf("/home/egauillard/data/tp_1940_2023_10deg_7res_winter.nc")

    # save 1deg resolution
    wandb_config['dataset']['out_spatial_resolution'] = 1
    dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory)
    target_1 = dataset.target_class.data.to_netcdf("/home/egauillard/data/tp_1940_2023_1deg_7res_winter.nc")

