import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
from omegaconf import OmegaConf
from typing import Union, List
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath("/home/egauillard/extreme_events_forecasting/vae_probabilistic/src"))
from model_vae.vae_model import VAE3DLightningModule

from data.dataset import DatasetEra
from data.temporal_aggregator import TemporalAggregatorFactory
from model.earthformer_model import CuboidERAModule
torch.set_default_dtype(torch.float32)
torch.set_default_device('cpu')



class S2SInference:
    def __init__(self, checkpoint_paths: Union[List[str], str], config_path: str, test_dataset: DatasetEra, model_type: str):
        self.checkpoint_paths = [checkpoint_paths] if isinstance(checkpoint_paths, str) else checkpoint_paths
        self.config_path = config_path
        self.test_dataset = test_dataset
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.load_models()
    
    def load_models(self):
        oc = OmegaConf.load(self.config_path)
        sample = next(iter(self.test_dataset))
        input_shape = list(sample[0].shape)
        output_shape = list(sample[1].shape)

        for checkpoint_path in self.checkpoint_paths:
            if self.model_type == "earthformer":
                model = CuboidERAModule(
                    total_num_steps=1,
                    config_file_path=self.config_path,
                    save_dir="temp",
                    input_shape=input_shape,
                    output_shape=output_shape
                )
                pass
            elif self.model_type == "vae":
                model = VAE3DLightningModule(
                    config_file_path=self.config_path,
                    save_dir="temp",
                    input_dims=input_shape,
                    output_dims=output_shape
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict['state_dict'])
            model.to(self.device)
            model.eval()
            self.models.append(model)

        self.config = oc

    def load_model(self, checkpoint_paths, config_path):
        # maybe autorise self.models to be a list of models or a single model
        oc = OmegaConf.load(self.config_path)
        sample = next(iter(self.test_dataset))
        input_shape = list(sample[0].shape)
        output_shape = list(sample[1].shape)

        if model_type == "earthformer":
            self.model = CuboidERAModule(
                total_num_steps=1,
                config_file_path=self.config_path,
                save_dir="temp",
                input_shape=input_shape,
                output_shape=output_shape
            )
        elif model_type == "vae":
            self.model = VAE3DLightningModule(config_file_path = self.config_path,
                                          save_dir = "temp",
                                         input_dims = input_shape,
                                          output_dims = output_shape)
        for checkpoint_path in self.checkpoint_paths:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.config = oc
    
    def run_inference(self):
        self.ensemble_pred = []
        self.all_ground_truth = []
        self.all_climatology = []
        self.all_target_time_indexes = []

        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=4
        )
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Processing batches"):
                input_data, target_data, season_float, year_float, clim_target, input_time_indexes, target_time_indexes = batch
                
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                batch_predictions = []
                if self.model_type == "vae":
                    for _ in range(50):  # Nombre fixe d'échantillons pour VAE
                        predictions, _, _, _ = self.models[0](input_data, target_data)
                        batch_predictions.append(predictions.cpu().numpy())
                else:  # earthformer
                    for model in self.models:
                        predictions = model((input_data, target_data, season_float, year_float, clim_target, input_time_indexes, target_time_indexes))[0]
                        batch_predictions.append(predictions.cpu().numpy())
                
                
                self.ensemble_pred.append(np.stack(batch_predictions, axis=1))  # Shape: (batch_size, nb_ensemble_members, ...)
                self.all_ground_truth.append(target_data.cpu().numpy())
                self.all_climatology.append(clim_target.numpy())
                self.all_target_time_indexes.append(self.test_dataset.aggregator.decode_time_indexes(target_time_indexes.numpy()))

        self.ensemble_pred = np.concatenate(self.ensemble_pred, axis=0)  # Shape: (total_samples, nb_ensemble_members, ...)
        self.all_ground_truth = np.concatenate(self.all_ground_truth, axis=0)
        self.all_climatology = np.concatenate(self.all_climatology, axis=0)
        self.all_target_time_indexes = np.concatenate(self.all_target_time_indexes, axis=0)

        print(f"Ensemble shape: {self.ensemble_pred.shape}")
        print(f"Number of ensemble members: {self.ensemble_pred.shape[1]}")

    def create_s2s_datasets(self):
        target_data = self.test_dataset.target_class.data
        lat_coords = target_data.latitude.values
        lon_coords = target_data.longitude.values
        resolution_output = self.test_dataset.resolution_output

        time_coords = pd.to_datetime(self.all_target_time_indexes.flatten())
        
        coords = {
            'time': time_coords,
            'latitude': lat_coords,
            'longitude': lon_coords,
        }
        
        ensemble_coords = coords.copy()
        ensemble_coords['realization'] = range(1, self.ensemble_pred.shape[1] + 1)

        data_vars_ensemble = {}
        data_vars_truth = {}
        data_vars_clim_test = {}
        data_vars_clim_mean = {}
        data_vars_clim_std = {}

        for i, var in enumerate(self.test_dataset.target_variables):
            data_ensemble = self.ensemble_pred[:, :, :, :, :, i].reshape(-1, self.ensemble_pred.shape[1], len(lat_coords), len(lon_coords))
            data_truth = self.all_ground_truth[:, :, :, :, i].reshape(-1, len(lat_coords), len(lon_coords))
            data_clim = self.all_climatology[:, :, :, :, i].reshape(-1, len(lat_coords), len(lon_coords))

            data_vars_ensemble[var] = xr.DataArray(
                data_ensemble,
                coords=ensemble_coords,
                dims=['time', 'realization', 'latitude', 'longitude']
            )
            data_vars_truth[var] = xr.DataArray(
                data_truth,
                coords=coords,
                dims=['time', 'latitude', 'longitude']
            )
            data_vars_clim_test[var] = xr.DataArray(
                data_clim,
                coords=coords,
                dims=['time', 'latitude', 'longitude']
            )
            
            # Préparation de clim_mean et clim_std
            clim_mean_data = np.array([self.test_dataset.target_class.climatology["mean"][var].sel(dayofyear=time.dayofyear).values for time in time_coords])
            clim_std_data = np.array([self.test_dataset.target_class.climatology["std"][var].sel(dayofyear=time.dayofyear).values for time in time_coords])
            
            data_vars_clim_mean[var] = xr.DataArray(
                clim_mean_data,
                coords=coords,
                dims=['time', 'latitude', 'longitude']
            )
            data_vars_clim_std[var] = xr.DataArray(
                clim_std_data,
                coords=coords,
                dims=['time', 'latitude', 'longitude']
            )

        ds_ensemble = xr.Dataset(data_vars_ensemble, coords=ensemble_coords)
        ds_ground_truth = xr.Dataset(data_vars_truth, coords=coords)
        ds_climatology = xr.Dataset(data_vars_clim_test, coords=coords)
        ds_clim_mean = xr.Dataset(data_vars_clim_mean, coords=coords)
        ds_clim_std = xr.Dataset(data_vars_clim_std, coords=coords)

        # Apply inverse scaling (excluding clim_mean and clim_std)
        ds_ensemble = self.test_dataset.target_class.inverse_transform(ds_ensemble)
        ds_ground_truth = self.test_dataset.target_class.inverse_transform(ds_ground_truth)
        ds_climatology = self.test_dataset.target_class.inverse_transform(ds_climatology)

        # Now restructure to S2S format
        lead_times = np.arange(1, self.ensemble_pred.shape[2] + 1) * resolution_output + self.test_dataset.aggregator.lead_time_gap
        forecast_times = time_coords[::len(lead_times)] - pd.Timedelta(days=resolution_output)

        def restructure_to_s2s(ds, include_realization=True):
            new_coords = {
                'forecast_time': forecast_times,
                'lead_time': lead_times,
                'latitude': lat_coords,
                'longitude': lon_coords,
            }
            if include_realization:
                new_coords['realization'] = ds.realization

            new_data_vars = {}
            for var in ds.data_vars:
                data = ds[var].values
                if var == "tp":
                    data *= resolution_output  # Convert from mean to sum
                if include_realization:
                    data = data.reshape(len(forecast_times), len(lead_times), -1, len(lat_coords), len(lon_coords))
                    dims = ['forecast_time', 'lead_time', 'realization', 'latitude', 'longitude']
                else:
                    data = data.reshape(len(forecast_times), len(lead_times), len(lat_coords), len(lon_coords))
                    dims = ['forecast_time', 'lead_time', 'latitude', 'longitude']
                new_data_vars[var] = xr.DataArray(data, coords=new_coords, dims=dims)
            
            return xr.Dataset(new_data_vars, coords=new_coords)

        ds_ensemble_s2s = restructure_to_s2s(ds_ensemble, include_realization=True)
        ds_ground_truth_s2s = restructure_to_s2s(ds_ground_truth, include_realization=False)
        ds_climatology_s2s = restructure_to_s2s(ds_climatology, include_realization=False)
        ds_clim_mean_s2s = restructure_to_s2s(ds_clim_mean, include_realization=False)
        ds_clim_std_s2s = restructure_to_s2s(ds_clim_std, include_realization=False)

        return ds_ensemble_s2s, ds_ground_truth_s2s, ds_climatology_s2s, ds_clim_mean_s2s, ds_clim_std_s2s

    def save_results(self, ds_ensemble, ds_ground_truth, ds_climatology, ds_clim_mean, ds_clim_std):
        base_dir = os.path.dirname(self.checkpoint_paths[0])
        save_dir = os.path.join(base_dir, f"s2s_inference_{self.model_type}")
        os.makedirs(save_dir, exist_ok=True)

        ds_ensemble.to_netcdf(os.path.join(save_dir, "ensemble_predictions.nc"))
        ds_ground_truth.to_netcdf(os.path.join(save_dir, "ground_truth.nc"))
        ds_climatology.to_netcdf(os.path.join(save_dir, "climatology.nc"))
        ds_clim_mean.to_netcdf(os.path.join(save_dir, "climatology_mean.nc"))
        ds_clim_std.to_netcdf(os.path.join(save_dir, "climatology_std.nc"))

    def run_evaluation(self):
        self.run_inference()
        return self.create_s2s_datasets()

if __name__ == "__main__":
    # checkpoint_paths = ["/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_202509_s_fine_tp_in8_0/checkpoints/loss/model-loss-epoch=005_valid_loss_epoch=1.04.ckpt",
    # "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_203734_s_fine_tp_in8_1/checkpoints/loss/model-loss-epoch=017_valid_loss_epoch=1.04.ckpt",
    # "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_215848_s_fine_tp_in8_2/checkpoints/loss/model-loss-epoch=014_valid_loss_epoch=1.04.ckpt",
    # "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_223218_s_fine_tp_in8_3/checkpoints/loss/model-loss-epoch=008_valid_loss_epoch=1.04.ckpt",
    # "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_225453_s_fine_tp_in8_4/checkpoints/loss/model-loss-epoch=014_valid_loss_epoch=1.04.ckpt"]

    # checkpoint_paths = ["/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241011_162428_s_fine_tp_in8_gap7_20/checkpoints/loss/model-loss-epoch=002_valid_loss_epoch=0.94.ckpt",
    # "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241012_212441_s_fine_tp_in8_gap7_25/checkpoints/loss/model-loss-epoch=008_valid_loss_epoch=0.94.ckpt"]
    # checkpoint_paths = ["/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_202509_s_fine_tp_in8_0/checkpoints/loss/last.ckpt",
    #                     "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_203734_s_fine_tp_in8_1/checkpoints/loss/last.ckpt",
    #                     "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_215848_s_fine_tp_in8_2/checkpoints/loss/last.ckpt",
    #                     "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_223218_s_fine_tp_in8_3/checkpoints/loss/last.ckpt",
    #                     "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20241013_225453_s_fine_tp_in8_4/checkpoints/loss/last.ckpt"]
    checkpoint_paths = "/home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240920_194025_ld9000_beta7_gamma1000_every_coarse/checkpoints/loss/epoch=20-val_loss=21.11.ckpt"
    if isinstance(checkpoint_paths, list):
        checkpoint_path = checkpoint_paths[0]
    else:
        checkpoint_path = checkpoint_paths
    
    exp_dir = checkpoint_path.split('/checkpoints/')[0]
    print(f"Experiment directory: {exp_dir}")
    config_path = os.path.join(exp_dir, 'cfg.yaml')

    oc = OmegaConf.load(config_path)
    dataset_cfg = OmegaConf.to_object(oc.data)
    test_config = dataset_cfg.copy()

    data_dirs = dataset_cfg['data_dirs']
    dataset_cfg['temporal_aggregator']['gap'] = 7

    all_ds_ensemble = []
    all_ds_ground_truth = []
    all_ds_climatology = []
    all_ds_clim_mean = []
    all_ds_clim_std = []

    model = CuboidERAModule(
        total_num_steps=1,
        config_file_path=config_path,
        save_dir="temp",
        input_shape=[6,111,360,6],
        output_shape=[6,15,50,1]
    )

    for year in [2016, 2017, 2018, 2019, 2020]:
        print(f"Processing year: {year}")
        test_config['dataset']['relevant_years'] = [year-1, year+1]
        test_config['dataset']["relevant_months"] = [10,11,12,1,2,3]
        temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])

        test_dataset = DatasetEra(test_config, data_dirs, temp_aggregator_factory, f"{year}-01-02")

        s2s_inference = S2SInference(checkpoint_paths, config_path, test_dataset, model_type="earthformer")
        ds_ensemble, ds_ground_truth, ds_climatology, ds_clim_mean, ds_clim_std = s2s_inference.run_evaluation()

        all_ds_ensemble.append(ds_ensemble)
        all_ds_ground_truth.append(ds_ground_truth)
        all_ds_climatology.append(ds_climatology)
        all_ds_clim_mean.append(ds_clim_mean)
        all_ds_clim_std.append(ds_clim_std)

    # Concatenate datasets along forecast_time dimension
    combined_ds_ensemble = xr.concat(all_ds_ensemble, dim='forecast_time')
    combined_ds_ground_truth = xr.concat(all_ds_ground_truth, dim='forecast_time')
    combined_ds_climatology = xr.concat(all_ds_climatology, dim='forecast_time')
    combined_ds_clim_mean = xr.concat(all_ds_clim_mean, dim='forecast_time')
    combined_ds_clim_std = xr.concat(all_ds_clim_std, dim='forecast_time')

    # Save combined results
    save_dir = os.path.join(exp_dir, f"s2s_inference_vae_combined")
    os.makedirs(save_dir, exist_ok=True)

    combined_ds_ensemble.to_netcdf(os.path.join(save_dir, "ensemble_predictions_combined.nc"))
    combined_ds_ground_truth.to_netcdf(os.path.join(save_dir, "ground_truth_combined.nc"))
    combined_ds_climatology.to_netcdf(os.path.join(save_dir, "climatology_combined.nc"))
    combined_ds_clim_mean.to_netcdf(os.path.join(save_dir, "climatology_mean_combined.nc"))
    combined_ds_clim_std.to_netcdf(os.path.join(save_dir, "climatology_std_combined.nc"))

    print(f"Combined results saved in {save_dir}")