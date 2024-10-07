import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
import cartopy.crs as ccrs
from sklearn.metrics import mean_squared_error, r2_score
import argparse 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import random
import sys
sys.path.append(os.path.abspath("/home/egauillard/extreme_events_forecasting/vae_probabilistic/src"))
from model.vae_model import VAE3DLightningModule
# Ajoutez le répertoire parent de 'data' au sys.path
sys.path.append(os.path.abspath("/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src"))
# Maintenant vous pouvez importer le module
from data.dataset import DatasetEra
from utils.statistics import DataScaler, DataStatistics
from utils.temporal_aggregator import TemporalAggregatorFactory

class Evaluation:
    def __init__(self, checkpoint_path, config_path, test_dataset, scaler,nb_samples = 50):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.test_dataset = test_dataset
        print(self.test_dataset.relevant_years)
        print(self.test_dataset.target_class.data)
        #(TODO) changer scaler et statistics
        self.scaler = scaler

        self.save_folder = self.create_save_folder()
        self.model = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
        self.unity = {"tp": "mm", "t2m": "K"}
        self.resolution_output = test_dataset.resolution_output
        self.out_spatial_resolution = test_dataset.out_spatial_resolution

        self.target_class = test_dataset.target_class
        computer = DataStatistics(years = test_dataset.scaling_years, months = test_dataset.relevant_months, coarse_temporal = self.target_class.coarse_t, coarse_spatial = self.target_class.coarse_s)
        self.statistics = computer._get_stats(test_dataset.target_class)
        self.nb_samples = nb_samples
    
    def inverse_scaling(self, data):
        return self.scaler.inverse_transform(data, self.statistics)

    def create_save_folder(self):
        checkpoint_dir, checkpoint_id = os.path.split(self.checkpoint_path)

        checkpoint_dir, checkpoint_id = os.path.split(self.checkpoint_path)
        exp_dir = checkpoint_dir.split('/checkpoints')[0]
        if '/checkpoints' in exp_dir:
            exp_dir = exp_dir.rsplit('/checkpoints', 1)[0]
        print(f"Checkpoints found in path. Experiment directory: {exp_dir}")

                save_folder = os.path.join(exp_dir, 'inference_plots', checkpoint_id)
        os.makedirs(save_folder, exist_ok=True)
        print(f"Save folder created at: {save_folder}")
        return save_folder

    def load_model(self):
        oc = OmegaConf.load(self.config_path)
        sample = next(iter(self.test_dataset))
        input_shape = list(sample[0].shape)
        output_shape = list(sample[1].shape)

        self.model = VAE3DLightningModule(config_file_path = self.config_path,
                                          save_dir = "temp",
                                         input_dims = input_shape,
                                          output_dims = output_shape)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.config = oc

    def run_inference(self):
        self.ensemble_pred = []
        self.all_ground_truth = []
        self.all_inputs = []
        self.all_input_time_indexes = []
        self.all_target_time_indexes = []

        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=4
        )
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Processing batches"):
                input_data, target_data, season_float, clim_target, year_float, input_time_indexes, target_time_indexes = batch
                
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                batch_predictions = []
                for _ in range(self.nb_samples):
                    predictions, target, mu, log_var = self.model(input_data, target_data)
                    batch_predictions.append(predictions.cpu().numpy())

                
                self.ensemble_pred.append(np.stack(batch_predictions, axis=1))  # Shape: (batch_size, num_samples, ...)
                self.all_ground_truth.append(target_data.cpu().numpy())
                self.all_inputs.append(input_data.cpu().numpy())
                self.all_input_time_indexes.append(self.test_dataset.aggregator.decode_time_indexes(input_time_indexes.numpy()))
                self.all_target_time_indexes.append(self.test_dataset.aggregator.decode_time_indexes(target_time_indexes.numpy()))

        self.ensemble_pred = np.concatenate(self.ensemble_pred, axis=0)  # Shape: (total_samples, num_samples, ...)
        self.all_inputs = np.concatenate(self.all_inputs, axis=0)
        self.all_ground_truth = np.concatenate(self.all_ground_truth, axis=0)
        self.all_input_time_indexes = np.concatenate(self.all_input_time_indexes, axis=0)
        self.all_target_time_indexes = np.concatenate(self.all_target_time_indexes, axis=0)

        # Calculate climatology
        self.climatology_mean = self.test_dataset.compute_climatology()["mean"]
        self.climatology_std = self.test_dataset.compute_climatology()["std"]

    def create_data_arrays(self):
        target_data = self.test_dataset.target_class.data
        lat_coords = target_data.latitude.values
        lon_coords = target_data.longitude.values

        self.all_datasets_ensemble = []
        self.all_datasets_ground_truth = []
        self.all_datasets_climatology = []
        self.all_datasets_climatology_std = []

        for sample in range(self.ensemble_pred.shape[0]):
            time_coords = pd.to_datetime(self.all_target_time_indexes[sample])

            coords = {
                'time': time_coords,
                'latitude': lat_coords,
                'longitude': lon_coords,
                'ensemble': range(self.ensemble_pred.shape[1])
            }

            data_vars_ensemble = {}
            data_vars_ground_truth = {}
            data_vars_climatology = {}
            data_vars_climatology_std = {}

            for i, var in enumerate(self.test_dataset.target_variables):
                data_vars_ensemble[var] = xr.DataArray(
                    self.ensemble_pred[sample, :, :, :, :, i],
                    coords=coords,
                    dims=['ensemble', 'time', 'latitude', 'longitude']
                )
                data_vars_ground_truth[var] = xr.DataArray(
                    self.all_ground_truth[sample, :, :, :, i],
                    coords={k: v for k, v in coords.items() if k != 'ensemble'},
                    dims=['time', 'latitude', 'longitude']
                )

                climatology_data = np.zeros_like(self.all_ground_truth[sample, :, :, :, i])
                climatology_std_data = np.zeros_like(self.all_ground_truth[sample, :, :, :, i])
                for t, time in enumerate(time_coords):
                    climatology_data[t, :, :] = self.climatology_mean[var].sel(dayofyear=time.dayofyear).values
                    climatology_std_data[t, :, :] = self.climatology_std[var].sel(dayofyear=time.dayofyear).values
                
                data_vars_climatology[var] = xr.DataArray(
                    climatology_data,
                    coords={k: v for k, v in coords.items() if k != 'ensemble'},
                    dims=['time', 'latitude', 'longitude']
                )
                data_vars_climatology_std[var] = xr.DataArray(
                    climatology_std_data,
                    coords={k: v for k, v in coords.items() if k != 'ensemble'},
                    dims=['time', 'latitude', 'longitude']
                )

            ds_ensemble = xr.Dataset(data_vars_ensemble, coords=coords)
            ds_ground_truth = xr.Dataset(data_vars_ground_truth, coords={k: v for k, v in coords.items() if k != 'ensemble'})
            ds_climatology = xr.Dataset(data_vars_climatology, coords={k: v for k, v in coords.items() if k != 'ensemble'})
            ds_climatology_std = xr.Dataset(data_vars_climatology_std, coords={k: v for k, v in coords.items() if k != 'ensemble'})

            # Apply inverse scaling
            ds_ensemble = self.inverse_scaling(ds_ensemble)
            ds_ground_truth = self.inverse_scaling(ds_ground_truth)

            # Multiply by 1000 for precipitation
            if "tp" in self.test_dataset.target_variables:
                ds_ensemble["tp"] *= 1000
                ds_ground_truth["tp"] *= 1000
                ds_climatology["tp"] *= 1000
                ds_climatology_std["tp"] *= 1000

            self.all_datasets_ensemble.append(ds_ensemble)
            self.all_datasets_ground_truth.append(ds_ground_truth)
            self.all_datasets_climatology.append(ds_climatology)
            self.all_datasets_climatology_std.append(ds_climatology_std)

    def generate_plots(self):
        n_samples = len(self.all_datasets_ensemble)
        random_samples = random.sample(range(n_samples), min(5, n_samples))

        for var_name in self.test_dataset.target_variables:
            for sample in random_samples:
                ensemble_dataset = self.all_datasets_ensemble[sample]
                truth_dataset = self.all_datasets_ground_truth[sample]
                clim_dataset = self.all_datasets_climatology[sample]

                time_steps = ensemble_dataset.time.values
                n_lead_times = len(time_steps)

                fig, axs = plt.subplots(3, n_lead_times, figsize=(5*n_lead_times, 15), 
                                        subplot_kw={'projection': ccrs.PlateCarree()})

                if n_lead_times == 1:
                    axs = axs.reshape(3, 1)

                lats = [30] + list(ensemble_dataset[var_name].latitude.values) + [45]
                lons = [-10] + list(ensemble_dataset[var_name].longitude.values) + [40]

                cmap = 'Blues' if var_name == "tp" else 'Reds'

                for lead_time, time in enumerate(time_steps):
                    pred = ensemble_dataset[var_name].sel(time=time).mean('ensemble')
                    truth = truth_dataset[var_name].sel(time=time)
                    clim = clim_dataset[var_name].sel(time=time)

                    vmin = min(pred.min().item(), truth.min().item(), clim.min().item())
                    vmax = max(pred.max().item(), truth.max().item(), clim.max().item())

                    for row, data in enumerate([pred, truth, clim]):
                        ax = axs[row, lead_time]
                        
                        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                                    extent=[lons[0], lons[-1], lats[0], lats[-1]],
                                    transform=ccrs.PlateCarree())
                        
                        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
                        ax.add_feature(cfeature.BORDERS, linestyle=':', color='black', linewidth=0.5)
                        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
                        ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='lightblue', alpha=0.3)
                        
                        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                        linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                        gl.xlocator = mticker.FixedLocator(range(-10, 41, 10))
                        gl.ylocator = mticker.FixedLocator(range(30, 46, 5))
                        
                        python_datetime = pd.Timestamp(time).to_pydatetime()
                        if row == 0:
                            title = f'Prediction average - {str(time)[:10]}'
                        elif row == 1:
                            title = f'Ground Truth average - {str(time)[:10]}'
                        else:
                            title = f'Climatology average - {str(time)[:10]}'
                        ax.set_title(title, fontsize=10)

                plt.tight_layout(w_pad=0.5, h_pad=1.0)
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(var_name, rotation=270, labelpad=15)
                fig.set_size_inches(5*n_lead_times + 1, 15)

                plt.savefig(os.path.join(self.save_folder, f'{var_name}_sample_{sample}_all_lead_times.png'), 
                            bbox_inches='tight', dpi=300)
                plt.close()

    def calculate_mse(self):
        self.mse_model = {}
        self.mse_climatology = {}
        self.relative_mse_model = {}
        self.relative_mse_climatology = {}
        self.rmse_model = {}
        self.rmse_climatology = {}
        self.skill_score = {}
        self.std_model = {}
        self.std_climatology = {}
        self.r2_model = {}
        self.r2_climatology = {}

        for var in self.test_dataset.target_variables:
            self.mse_model[var] = []
            self.mse_climatology[var] = []
            self.relative_mse_model[var] = []
            self.relative_mse_climatology[var] = []
            self.rmse_model[var] = []
            self.rmse_climatology[var] = []
            self.skill_score[var] = []
            self.std_model[var] = []
            self.std_climatology[var] = []
            self.r2_model[var] = []
            self.r2_climatology[var] = []
            
            for lead_time in range(len(self.all_datasets_ground_truth[0].time)):
                truth = np.concatenate([ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_ground_truth])
                ensemble_preds = np.array([np.concatenate([ds[var].isel(time=lead_time).isel(ensemble = member_idx).values.flatten() for ds in self.all_datasets_ensemble]) for member_idx in range(self.nb_samples)])
                clim = np.concatenate([ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_climatology])
                
                print(f"Shape of truth: {truth.shape}")
                print(f"Shape of ensemble_preds: {ensemble_preds.shape}")
                print(f"Shape of clim: {clim.shape}")
                
                # Assurez-vous que les dimensions correspondent
                if ensemble_preds.shape[0] != len(self.all_datasets_ensemble):
                    ensemble_preds = ensemble_preds.T

                # Calculer la MSE pour chaque membre de l'ensemble, on fait la moyenne spatialement
                mse_ensemble = np.mean((truth[:,np.newaxis] - ensemble_preds)**2, axis=0)
                
                # Prendre la moyenne des MSE de l'ensemble
                mse_model = np.mean(mse_ensemble)
                mse_clim = np.mean((truth - clim)**2)

                # Calculer RMSE et écart-type pour chaque membre de l'ensemble
                rmse_ensemble = np.sqrt(mse_ensemble)
                rmse_model = np.mean(rmse_ensemble)
                std_rmse = np.std(rmse_ensemble)

                std_clim_rmse = np.std(np.sqrt((truth - clim)**2))

                # Calculer R2 pour chaque membre de l'ensemble et prendre la moyenne
                r2_ensemble = [r2_score(truth, pred) for pred in ensemble_preds.T]
                r2_model = np.mean(r2_ensemble)
                r2_clim = r2_score(truth, clim)

                range_squared = (np.max(truth) - np.min(truth))**2
                relative_rmse = np.sqrt(mse_model / range_squared)

                self.mse_model[var].append(mse_model)
                self.mse_climatology[var].append(mse_clim)
                self.relative_mse_model[var].append(mse_model / range_squared)
                self.relative_mse_climatology[var].append(mse_clim / range_squared)
                self.rmse_model[var].append(rmse_model)
                self.rmse_climatology[var].append(np.sqrt(mse_clim))
                self.skill_score[var].append(1 - (mse_model / mse_clim))
                self.std_model[var].append(std_rmse)
                self.std_climatology[var].append(std_clim_rmse)
                self.r2_model[var].append(r2_model)
                self.r2_climatology[var].append(r2_clim)

        # Transformer tous les RMSE par cellule de grille
        # self.rmse_climatology = {key: [v / self.out_spatial_resolution for v in value] for key, value in self.rmse_climatology.items()}
        # self.rmse_model = {key: [v / self.out_spatial_resolution for v in value] for key, value in self.rmse_model.items()}
        
    def plot_mse_curves(self):
        for var in self.test_dataset.target_variables:
            lead_times = [i*self.test_dataset.resolution_output for i in range(len(self.all_datasets_ground_truth[0].time))]
            lead_time_gap = self.test_dataset.aggregator.lead_time_gap
            x_labels = [f"{t + lead_time_gap}-{t+self.test_dataset.resolution_output+ lead_time_gap}" for t in lead_times]

            # Plot MSE
            plt.figure(figsize=(12, 7))
            plt.plot(lead_times, self.mse_model[var], 'o-', label=f'Ensemble Mean MSE (n={self.nb_samples})', markersize=6)
            plt.plot(lead_times, self.mse_climatology[var], 's-', label='Climatology', markersize=6)
            plt.title(f'Mean Squared Error vs Lead Time - {var}', fontsize=16)
            plt.xlabel('Lead Time', fontsize=14)
            plt.ylabel('MSE', fontsize=14)
            plt.xticks(lead_times, x_labels, rotation=45, ha='right')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(fontsize=12, loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, f'{var}_mse_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Plot RMSE
            plt.figure(figsize=(12, 7))
            plt.plot(lead_times, self.rmse_model[var], 'o-', label=f'Ensemble Mean RMSE (n={self.nb_samples})', markersize=6)
            plt.plot(lead_times, self.rmse_climatology[var], 's-', label='Climatology RMSE', markersize=6)
            plt.title(f'Root Mean Squared Error vs Lead Time - {var}', fontsize=16)
            plt.xlabel('Lead Time', fontsize=14)
            plt.ylabel(f'RMSE in {self.unity[var]}', fontsize=14)
            plt.xticks(lead_times, x_labels, rotation=45, ha='right')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(fontsize=12, loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, f'{var}_rmse_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Plot R2
            plt.figure(figsize=(12, 7))
            plt.plot(lead_times, self.r2_model[var], 'o-', label=f'Ensemble Mean R2 (n={self.nb_samples})', markersize=6)
            plt.plot(lead_times, self.r2_climatology[var], 's-', label='Climatology R2', markersize=6)
            plt.title(f'R2 Score vs Lead Time - {var}', fontsize=16)
            plt.xlabel('Lead Time', fontsize=14)
            plt.ylabel('R2 Score', fontsize=14)
            plt.xticks(lead_times, x_labels, rotation=45, ha='right')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(fontsize=12, loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, f'{var}_r2_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Plot Skill Score
            plt.figure(figsize=(12, 7))
            plt.plot(lead_times, self.skill_score[var], 'o-', label=f'Ensemble Skill Score (n={self.nb_samples})', markersize=6)
            plt.axhline(y=0, color='r', linestyle='--', label='No Skill Line')
            plt.title(f'Skill Score vs Lead Time - {var}', fontsize=16)
            plt.xlabel('Lead Time', fontsize=14)
            plt.ylabel('Skill Score', fontsize=14)
            plt.xticks(lead_times, x_labels, rotation=45, ha='right')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(fontsize=12, loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, f'{var}_skill_score.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def calculate_spatial_mse(self):
        self.spatial_mse_model = {}
        self.spatial_rmse_model = {}
        self.spatial_mse_climatology = {}
        self.spatial_rmse_climatology = {}
        
        for var in self.test_dataset.target_variables:
            mse_model = np.zeros_like(self.all_datasets_ground_truth[0][var].isel(time=0).values)
            mse_climatology = np.zeros_like(self.all_datasets_ground_truth[0][var].isel(time=0).values)
            truth_min = np.inf
            truth_max = -np.inf
            
            for ds_truth, ds_pred, ds_clim in zip(self.all_datasets_ground_truth, self.all_datasets_ensemble, self.all_datasets_climatology):
                truth = ds_truth[var].values
                pred_ensemble = ds_pred[var].values
                clim = ds_clim[var].values

                # do the mean over the time steps 
                mse_ensemble = np.mean((truth[np.newaxis,: ,: ,:] - pred_ensemble)**2, axis=(1))
                
                # Prendre la moyenne des MSE de l'ensemble
                mse_model += np.mean(mse_ensemble, axis=0)
                mse_climatology += np.mean((truth - clim)**2, axis=0)
                truth_min = min(truth_min, np.min(truth))
                truth_max = max(truth_max, np.max(truth))
            
            mse_model /= len(self.all_datasets_ground_truth)
            mse_climatology /= len(self.all_datasets_ground_truth)
            self.spatial_mse_model[var] = mse_model
            self.spatial_mse_climatology[var] = mse_climatology
            range_squared = (truth_max - truth_min)**2
            self.spatial_rmse_model[var] = np.sqrt(mse_model)
            self.spatial_rmse_climatology[var] = np.sqrt(mse_climatology)

    def plot_spatial_mse(self):
        for var in self.test_dataset.target_variables:
            lats = [30] + list(self.all_datasets_ground_truth[0].latitude.values) + [45]
            lons = [-10] + list(self.all_datasets_ground_truth[0].longitude.values) + [40]
            def symmetric_limits(data):
                abs_max = max(abs(data.min()), abs(data.max()))
                return -abs_max, abs_max
            def pos_limits(data):
                lim_max = max(data)
                return 0, lim_max

            vmin_rmse, vmax_rmse = pos_limits(np.concatenate([
                self.spatial_rmse_model[var].flatten(),
                self.spatial_rmse_climatology[var].flatten()
            ]))

                # Fonction pour créer un plot
            def create_plot(data, title, filename, vmin, vmax, label=None, cmap : str = 'RdBu_r'):
                fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
                im = ax.imshow(data, cmap=cmap, transform=ccrs.PlateCarree(),
                            extent=[lons[0], lons[-1], lats[0], lats[-1]],
                            vmin=vmin, vmax=vmax)
                ax.coastlines(resolution='50m', color='black', linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=':', color='black', linewidth=0.5)
                ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
                ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='lightblue', alpha=0.3)
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.xlocator = mticker.FixedLocator(range(-10, 41, 10))
                gl.ylocator = mticker.FixedLocator(range(30, 46, 5))
                gl.top_labels = False
                gl.right_labels = False
                
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, extend='both')
                if label:
                    cbar.set_label(label, fontsize=12)
                
                plt.title(title, fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_folder, filename), 
                            bbox_inches='tight', dpi=300)
                plt.close()

            # Plot Relative MSE
            create_plot(self.spatial_rmse_model[var], f'RMSE average ensemble Model - {var}', f'{var}_spatial_rmse_model.png', vmin_rmse, vmax_rmse, f'RMSE in {self.unity[var]}', 'YlOrRd')
            create_plot(self.spatial_rmse_climatology[var], f'RMSE Climatology - {var}', f'{var}_spatial_rmse_climatology.png', vmin_rmse, vmax_rmse, f'RMSE in {self.unity[var]}', 'YlOrRd')
            create_plot(self.spatial_rmse_model[var] - self.spatial_rmse_climatology[var], f'Difference RMSE Ensemble Model - Climatology - {var}', f'{var}_spatial_rmse_diff.png', -3, 3, f'RMSE in {self.unity[var]}', 'RdYlBu')

    def save_mse_to_csv(self):
        df = pd.DataFrame()
        for var in self.test_dataset.target_variables:
            df[f'{var}_ensemble_mse'] = self.mse_model[var]
            df[f'{var}_climatology_mse'] = self.mse_climatology[var]
            df[f'{var}_ensemble_relative_mse'] = self.relative_mse_model[var]
            df[f'{var}_climatology_relative_mse'] = self.relative_mse_climatology[var]
            df[f'{var}_ensemble_rmse'] = self.rmse_model[var]
            df[f'{var}_climatology_rmse'] = self.rmse_climatology[var]
            df[f'{var}_skill_score'] = self.skill_score[var]
            df[f'{var}_ensemble_std_rmse'] = self.std_model[var]
            df[f'{var}_climatology_std_rmse'] = self.std_climatology[var]
            df[f'{var}_ensemble_r2'] = self.r2_model[var]
            df[f'{var}_climatology_r2'] = self.r2_climatology[var]
        
        means = df.mean()
        df.loc['---'] = '---'
        df.loc['mean'] = means

        df.to_csv(os.path.join(self.save_folder, 'ensemble_mse_results.csv'), index=False)
    
    def get_save_paths(self):
        dic = {}
        exp_dir = self.checkpoint_path.split('/checkpoints/')[0]

        dic['ensemble_pred'] = os.path.join(self.save_folder, 'all_ensemble_predictions.nc')
        dic['truth'] = os.path.join(self.save_folder, 'all_ground_truths.nc')
        dic['climatology'] = os.path.join(self.save_folder, 'all_climatology.nc')
        dic['climatology_std'] = os.path.join(self.save_folder, 'all_climatology_std.nc')
        dic["truth_era"] = os.path.join(exp_dir, '1940_2024_target.nc')
        dic["save_folder"] = self.save_folder
        return dic 

    def save_results(self):
        # Créer des listes pour stocker les datasets
        all_ensembles = []
        all_truths = []
        all_clims = []
        all_clim_stds = []
        
        # Parcourir tous les échantillons
        for i, (ensemble, truth, clim, clim_std) in enumerate(zip(self.all_datasets_ensemble, self.all_datasets_ground_truth, self.all_datasets_climatology, self.all_datasets_climatology_std)):
            # Ajouter une dimension 'sample' à chaque dataset
            ensemble = ensemble.expand_dims(sample=[i])
            truth = truth.expand_dims(sample=[i])
            clim = clim.expand_dims(sample=[i])
            clim_std = clim_std.expand_dims(sample=[i])
            
            all_ensembles.append(ensemble)
            all_truths.append(truth)
            all_clims.append(clim)
            all_clim_stds.append(clim_std)

        # Combiner tous les datasets
        combined_ensembles = xr.concat(all_ensembles, dim='sample')
        combined_truths = xr.concat(all_truths, dim='sample')
        combined_clims = xr.concat(all_clims, dim='sample')
        combined_clim_stds = xr.concat(all_clim_stds, dim='sample')

        # Sauvegarder les datasets combinés
        combined_ensembles.to_netcdf(os.path.join(self.save_folder, 'all_ensemble_predictions.nc'))
        combined_truths.to_netcdf(os.path.join(self.save_folder, 'all_ground_truths.nc'))
        combined_clims.to_netcdf(os.path.join(self.save_folder, 'all_climatology.nc'))
        combined_clim_stds.to_netcdf(os.path.join(self.save_folder, 'all_climatology_std.nc'))
        
        self.target_class.data.to_netcdf(os.path.join(self.save_folder, 'target.nc'))

        # Calculer et sauvegarder les MSE
        self.calculate_mse()
        self.save_mse_to_csv()

        # Générer les graphiques
        self.plot_mse_curves()
        self.calculate_spatial_mse()
        self.plot_spatial_mse()

    def run_evaluation(self):
        self.load_model()
        self.run_inference()
        self.create_data_arrays()
        self.generate_plots()
        self.save_results()


# Usage
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_path', type=str, required=True)
    # args = parser.parse_args()
    # checkpoint_path = args.checkpoint_path
    checkpoint_path = "/home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240925_112910/checkpoints/epoch=19-val_prediction_loss=0.79-val_kld_loss=0.24.ckpt"

    exp_dir = checkpoint_path.split('/checkpoints/')[0]
    print(f"Experiment directory: {exp_dir}")
    config_path = os.path.join(exp_dir, 'cfg.yaml')

    oc = OmegaConf.load(config_path)
    dataset_cfg = OmegaConf.to_object(oc.data)
    test_config = dataset_cfg.copy()
    test_config['dataset']['relevant_years'] = [2016, 2024]
    lead_time = dataset_cfg['temporal_aggregator']['out_len']

    data_dirs = dataset_cfg['data_dirs']
    scaler = DataScaler(dataset_cfg['scaler'])
    dataset_cfg['temporal_aggregator']['gap'] = dataset_cfg['temporal_aggregator']['gap'] = dataset_cfg['temporal_aggregator']['resolution_output']* lead_time

    temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])

    test_dataset = DatasetEra(test_config, data_dirs, temp_aggregator_factory, scaler)

    # save the whole dataset to get a climatology for the percentiles
    all_config = dataset_cfg.copy()
    all_config['dataset']['relevant_years'] = [1940, 2024]

    dataset =  DatasetEra(all_config, data_dirs, temp_aggregator_factory, scaler)
    target_class = dataset.reverse_scaling(dataset.target_class).to_netcdf(os.path.join(exp_dir, '1940_2024_target.nc'))

    # unscale the data

    eval = Evaluation(checkpoint_path, config_path, test_dataset, scaler)
    print("ready to eval")
    eval.run_evaluation()

  