import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
import cartopy.crs as ccrs
from sklearn.metrics import mean_squared_error
from  model.earthformer_model import CuboidERAModule
from utils.temporal_aggregator import TemporalAggregatorFactory
from utils.statistics import DataScaler
import argparse 
from data.dataset import DatasetEra
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from utils.statistics import DataStatistics
import random


class Evaluation:
    def __init__(self, checkpoint_path, config_path, test_dataset, scaler):
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
    
    def inverse_scaling(self, data):
        return self.scaler.inverse_transform(data, self.statistics)

    def create_save_folder(self):
        checkpoint_dir, _ = os.path.split(self.checkpoint_path)

        if '/checkpoints/' in checkpoint_dir:
            exp_dir = checkpoint_dir.split('/checkpoints/')[0]
            print(f"Checkpoints found in path. Experiment directory: {exp_dir}")
        else:
            # Handle the case where 'checkpoints' is not in the path (fallback for older structure)
            exp_dir = os.path.dirname(os.path.dirname(self.checkpoint_path))
            print(f"Checkpoints not found in path. Fallback experiment directory: {exp_dir}")

        save_folder = os.path.join(exp_dir, 'inference_plots')
        os.makedirs(save_folder, exist_ok=True)
        print(f"Save folder created at: {save_folder}")
        return save_folder

    def load_model(self):
        oc = OmegaConf.load(self.config_path)
        sample = next(iter(self.test_dataset))
        input_shape = list(sample[0].shape)
        output_shape = list(sample[1].shape)

        self.model = CuboidERAModule(
            total_num_steps=1,
            config_file_path=self.config_path,
            save_dir="temp",
            input_shape=input_shape,
            output_shape=output_shape
        )
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.config = oc

    def calculate_skill_score(self, predictions, ground_truth, climatology):
        mse_model = mean_squared_error(ground_truth, predictions)
        mse_climatology = mean_squared_error(ground_truth, climatology)
        return 1 - (mse_model / mse_climatology)

    def run_inference(self):
        all_predictions = []
        all_ground_truth = []
        all_inputs = []
        all_input_time_indexes = []
        all_target_time_indexes = []
        clim_test = []

        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=4
        )

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Processing batches"):
                print("first batch")
                input_data, target_data, season_float, clim_target, year_float, input_time_indexes, target_time_indexes = batch
                
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                predictions,  loss, input, target,  clim = self.model((input_data, target_data, clim_target, season_float, year_float, input_time_indexes, target_time_indexes))
                print("predictions", predictions)
                print("loss", loss)
                print("input", input)
                print("target", target)
                print("clim", clim)
                
                all_inputs.append(input_data.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_ground_truth.append(target_data.cpu().numpy())
                all_input_time_indexes.append(self.test_dataset.aggregator.decode_time_indexes(input_time_indexes.numpy()))
                all_target_time_indexes.append(self.test_dataset.aggregator.decode_time_indexes(target_time_indexes.numpy()))
                clim_test.append(clim.cpu().numpy())

        self.all_inputs = np.concatenate(all_inputs, axis=0)
        self.all_predictions = np.concatenate(all_predictions, axis=0)
        self.all_ground_truth = np.concatenate(all_ground_truth, axis=0)
        self.all_input_time_indexes = np.concatenate(all_input_time_indexes, axis=0)
        self.all_target_time_indexes = np.concatenate(all_target_time_indexes, axis=0)
        self.clim_test = np.concatenate(clim_test, axis=0)

        # Calculate climatology
        self.climatology = self.test_dataset.compute_climatology()["mean"]


    def create_data_arrays(self):
        target_data = self.test_dataset.target_class.data
        lat_coords = target_data.latitude.values
        lon_coords = target_data.longitude.values

        self.all_datasets_predictions = []
        self.all_datasets_ground_truth = []
        self.all_datasets_climatology = []
        self.all_datasets_clim_test = []  # Nouvelle liste pour clim_test

        for sample in range(self.all_predictions.shape[0]):
            # Convertir les time_indexes en objets datetime pour cet échantillon
            time_coords = pd.to_datetime(self.all_target_time_indexes[sample])

            coords = {
                'time': time_coords,
                'latitude': lat_coords,
                'longitude': lon_coords
            }

            data_vars_predictions = {}
            data_vars_ground_truth = {}
            data_vars_climatology = {}
            data_vars_clim_test = {}  # Nouveau dictionnaire pour clim_test

            for i, var in enumerate(self.test_dataset.target_variables):
                data_vars_predictions[var] = xr.DataArray(
                    self.all_predictions[sample, :, :, :, i],
                    coords=coords,
                    dims=['time', 'latitude', 'longitude']
                )
                data_vars_ground_truth[var] = xr.DataArray(
                    self.all_ground_truth[sample, :, :, :, i],
                    coords=coords,
                    dims=['time', 'latitude', 'longitude']
                )
                data_vars_clim_test[var] = xr.DataArray(  # Nouvelle DataArray pour clim_test
                    self.clim_test[sample, :, :, :, i],
                    coords=coords,
                    dims=['time', 'latitude', 'longitude']
                )

                climatology_data = np.zeros_like(self.all_predictions[sample, :, :, :, i])
                for t, time in enumerate(time_coords):
                    climatology_data[t, :, :] = self.climatology[var].sel(dayofyear = time.dayofyear).values
                
                data_vars_climatology[var] = xr.DataArray(
                    climatology_data,
                    coords=coords,
                    dims=['time', 'latitude', 'longitude']
                )

            # Créer les Datasets pour cet échantillon
            ds_predictions = xr.Dataset(data_vars_predictions, coords=coords)
            ds_ground_truth = xr.Dataset(data_vars_ground_truth, coords=coords)
            ds_climatology = xr.Dataset(data_vars_climatology, coords=coords)
            ds_clim_test = xr.Dataset(data_vars_clim_test, coords=coords)  # Nouveau Dataset pour clim_test

            print("ds_predictions", ds_predictions)
            print("ds_ground_truth", ds_ground_truth)
            print("ds_climatology", ds_climatology)
            print("ds_clim_test", ds_clim_test)

            # Appliquer l'inverse scaling séparément
            ds_predictions = self.inverse_scaling(ds_predictions)
            ds_ground_truth = self.inverse_scaling(ds_ground_truth)
            ds_clim_test = self.inverse_scaling(ds_clim_test)  # Appliquer l'inverse scaling à clim_test
            print("ds_predictions", ds_predictions)
            print("ds_ground_truth", ds_ground_truth)
            print("ds_climatology", ds_climatology)
            print("ds_clim_test", ds_clim_test)

            self.all_datasets_predictions.append(ds_predictions)
            self.all_datasets_ground_truth.append(ds_ground_truth)
            self.all_datasets_climatology.append(ds_climatology)
            self.all_datasets_clim_test.append(ds_clim_test)  # Ajouter clim_test à la liste

    def generate_plots(self):
        n_samples = len(self.all_datasets_predictions)
        random_samples = random.sample(range(n_samples), min(30, n_samples))

        for var_name in self.test_dataset.target_variables:
            for sample in random_samples:
                pred_dataset = self.all_datasets_predictions[sample]
                truth_dataset = self.all_datasets_ground_truth[sample]
                clim_dataset = self.all_datasets_climatology[sample]

                time_steps = pred_dataset.time.values
                n_lead_times = len(time_steps)

                fig, axs = plt.subplots(3, n_lead_times, figsize=(5*n_lead_times, 15), 
                                        subplot_kw={'projection': ccrs.PlateCarree()})

                if n_lead_times == 1:
                    axs = axs.reshape(3, 1)

                lats = [30] + list(pred_dataset[var_name].latitude.values) + [45]
                lons = [-10] + list(pred_dataset[var_name].longitude.values) + [40]

                cmap = 'Blues' if var_name == "tp" else 'Reds'

                for lead_time, time in enumerate(time_steps):
                    pred = pred_dataset[var_name].sel(time=time)
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
                        
                        import pandas as pd
                        python_datetime = pd.Timestamp(time).to_pydatetime()
                        if row == 0:
                            title = f'Prediction average - {str(time)[:10]}'
                        elif row == 1:
                            title = f'Ground Truth average - {str(time)[:10]}'
                        else:
                            title = f'Climatology average - {str(time)[:10]}'
                        ax.set_title(title, fontsize=10)

                # Adjust layout to remove extra space
                plt.tight_layout(w_pad=0.5, h_pad=1.0)

                # Add a common colorbar to the right side
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(var_name, rotation=270, labelpad=15)

                # Adjust the figure size to accommodate the colorbar
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

        climatology_errors = {var: [] for var in self.test_dataset.target_variables}
        for sample_idx in range(len(self.all_datasets_ground_truth)):
            for var in self.test_dataset.target_variables:
                errors = []
                for lead_time in range(len(self.all_datasets_ground_truth[0].time)):
                    truth = self.all_datasets_ground_truth[sample_idx][var].isel(time=lead_time).values
                    clim = self.all_datasets_climatology[sample_idx][var].isel(time=lead_time).values
                    error = np.mean((truth - clim)**2)  # MSE
                    errors.append(error)
                climatology_errors[var].append(errors)

        # Analyse des erreurs
        for var in self.test_dataset.target_variables:
            errors_array = np.array(climatology_errors[var])
            mean_errors = np.mean(errors_array, axis=0)
            std_errors = np.std(errors_array, axis=0)
            
            print(f"Variable: {var}")
            for lead_time, (mean, std) in enumerate(zip(mean_errors, std_errors)):
                print(f"  Lead Time {lead_time}: Mean Error = {mean:.4f}, Std Error = {std:.4f}")
            
            # Visualisation
            plt.figure(figsize=(12, 6))
            plt.errorbar(range(len(mean_errors)), mean_errors, yerr=std_errors, fmt='o-')
            plt.title(f'Mean Climatology Error vs Lead Time - {var}')
            plt.xlabel('Lead Time')
            plt.ylabel('Mean Error')
            plt.savefig(os.path.join(self.save_folder, f'{var}_climatology_error_analysis.png'))
            plt.close()
        
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
            
            for lead_time in range(len(self.all_datasets_ground_truth[0].time)):
                print('lead time debugging')
                print(lead_time)

                print("truth", [ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_ground_truth])
                print("pred", [ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_predictions])
                print("clim", [ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_climatology])
                truth = np.concatenate([ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_ground_truth])
                pred = np.concatenate([ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_predictions])
                clim = np.concatenate([ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_climatology])
                
                mse_model = np.mean((truth - pred)**2)
                mse_clim = np.mean((truth - clim)**2)
                # trouver la std du rmse 
                std_rmse = np.std(np.sqrt((truth - pred)**2))
                std_clim_rmse = np.std(np.sqrt((truth - clim)**2))
                    

                range_squared = (np.max(truth) - np.min(truth))**2
                relative_rmse = np.sqrt(mse_model / range_squared)
                rmse = np.sqrt(mse_model)

                print(f"Variable: {var}, Lead Time: {lead_time}")
                print(f"  MSE: {mse_model}")
                print(f"  Climatology MSE: {mse_clim}")
                print(f"  RMSE: {rmse}")
                print(f"  Range Squared: {range_squared}")
                print(f"  Relative RMSE: {relative_rmse}")
                print(f"  Truth - Mean: {np.mean(truth)}, Std: {np.std(truth)}, Min: {np.min(truth)}, Max: {np.max(truth)}")
                print(f"  Pred  - Mean: {np.mean(pred)}, Std: {np.std(pred)}, Min: {np.min(pred)}, Max: {np.max(pred)}")
                print()
                            
                
                self.std_model[var].append(std_rmse)
                self.std_climatology[var].append(std_clim_rmse)
                self.mse_model[var].append(mse_model)
                self.mse_climatology[var].append(mse_clim)
                self.relative_mse_model[var].append(mse_model / range_squared)
                self.relative_mse_climatology[var].append(mse_clim / range_squared)
                self.rmse_model[var].append(np.sqrt(mse_model))
                self.rmse_climatology[var].append(np.sqrt(mse_clim))
                self.skill_score[var].append(1- (mse_model / mse_clim))
        
        
            if var == "tp":
                self.rmse_model[var] = [x * 1000 for x in self.rmse_model["tp"]]
                self.rmse_climatology[var] = [x * 1000 for x in self.rmse_climatology["tp"]]
        # transformer tous les rmse par grid cell 
        self.rmse_climatology = {key: [v / self.out_spatial_resolution for v in value] for key, value in self.rmse_climatology.items()}
        self.rmse_model = {key: [v / self.out_spatial_resolution for v in value] for key, value in self.rmse_model.items()}



    def plot_mse_curves(self):
        for var in self.test_dataset.target_variables:
            lead_times = [i*self.test_dataset.resolution_output for i in range(len(self.all_datasets_ground_truth[0].time))]
            x_labels = [f"{t}-{t+self.test_dataset.resolution_output}" for t in lead_times]

            # Plot Relative MSE
            plt.figure(figsize=(12, 7))
            plt.plot(lead_times, self.relative_mse_model[var], 'o-', label='Model', markersize=6)
            plt.plot(lead_times, self.relative_mse_climatology[var], 's-', label='Climatology', markersize=6)
            plt.title(f'Relative MSE vs Lead Time - {var}', fontsize=16)
            plt.xlabel('Lead Time', fontsize=14)
            plt.ylabel('Relative MSE', fontsize=14)
            plt.xticks(lead_times, x_labels, rotation=45, ha='right')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(fontsize=12, loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, f'{var}_relative_mse_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Plot MSE
            plt.figure(figsize=(12, 7))
            plt.plot(lead_times, self.mse_model[var], 'o-', label='Model', markersize=6)
            plt.plot(lead_times, self.mse_climatology[var], 's-', label='Climatology', markersize=6)
            plt.title(f'MSE vs Lead Time - {var}', fontsize=16)
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
            plt.plot(lead_times, self.rmse_model[var], 'o-', label='Model', markersize=6)
            plt.plot(lead_times, self.rmse_climatology[var], 's-', label='Climatology', markersize=6)
            plt.title(f'RMSE vs Lead Time - {var}', fontsize=16)
            plt.xlabel('Lead Time', fontsize=14)
            plt.ylabel(f'RMSE in {self.unity[var]} ', fontsize=14)
            plt.xticks(lead_times, x_labels, rotation=45, ha='right')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(fontsize=12, loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, f'{var}_rmse_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()


    def calculate_spatial_mse(self):
        self.spatial_mse_model = {}
        self.spatial_relative_mse_model = {}
        self.spatial_rmse_model = {}
        self.spatial_mse_climatology = {}
        self.spatial_relative_mse_climatology = {}
        self.spatial_rmse_climatology = {}
        
        for var in self.test_dataset.target_variables:
            mse_model = np.zeros_like(self.all_datasets_ground_truth[0][var].isel(time=0).values)
            mse_climatology = np.zeros_like(self.all_datasets_ground_truth[0][var].isel(time=0).values)
            truth_min = np.inf
            truth_max = -np.inf
            
            for ds_truth, ds_pred, ds_clim in zip(self.all_datasets_ground_truth, self.all_datasets_predictions, self.all_datasets_climatology):
                truth = ds_truth[var].values
                pred = ds_pred[var].values
                clim = ds_clim[var].values
                mse_model += np.mean((truth - pred)**2, axis=0)
                mse_climatology += np.mean((truth - clim)**2, axis=0)
                truth_min = min(truth_min, np.min(truth))
                truth_max = max(truth_max, np.max(truth))
            
            mse_model /= len(self.all_datasets_ground_truth)
            mse_climatology /= len(self.all_datasets_ground_truth)
            self.spatial_mse_model[var] = mse_model
            self.spatial_mse_climatology[var] = mse_climatology
            range_squared = (truth_max - truth_min)**2
            self.spatial_relative_mse_model[var] = mse_model / range_squared
            self.spatial_relative_mse_climatology[var] = mse_climatology / range_squared
            self.spatial_rmse_model[var] = np.sqrt(mse_model) 
            self.spatial_rmse_climatology[var] = np.sqrt(mse_climatology)
            if var == "tp":
                self.spatial_rmse_model[var] *= 1000
                self.spatial_rmse_climatology[var] *= 1000
        


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

            # Calculer les valeurs min et max symétriques pour chaque type de plot
            vmin_relative, vmax_relative = pos_limits(np.concatenate([
                self.spatial_relative_mse_model[var].flatten(),
                self.spatial_relative_mse_climatology[var].flatten()
            ]))
            
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
            create_plot(self.spatial_relative_mse_model[var], f'Relative MSE Model - {var}', f'{var}_spatial_relative_mse_model.png', vmin_relative, vmax_relative, 'Relative MSE', 'RdBu_r')
            create_plot(self.spatial_relative_mse_climatology[var], f'Relative MSE Climatology - {var}', f'{var}_spatial_relative_mse_climatology.png', vmin_relative, vmax_relative, 'Relative MSE', 'RdBu_r')
            create_plot(self.spatial_rmse_model[var], f'RMSE Model - {var}', f'{var}_spatial_rmse_model.png', vmin_rmse, vmax_rmse, f'RMSE in {self.unity[var]}', 'RdBu_r')
            create_plot(self.spatial_rmse_climatology[var], f'RMSE Climatology - {var}', f'{var}_spatial_rmse_climatology.png', vmin_rmse, vmax_rmse, f'RMSE in {self.unity[var]}', 'RdBu_r')
            create_plot(self.spatial_rmse_model[var] - self.spatial_rmse_climatology[var], f'Difference RMSE Model - Climatology - {var}', f'{var}_spatial_rmse_diff.png', -3, 3, f'RMSE in {self.unity[var]}', 'bwr')


    def save_mse_to_csv(self):
        df = pd.DataFrame()
        for var in self.test_dataset.target_variables:
            df[f'{var}_model'] = self.mse_model[var]
            df[f'{var}_climatology'] = self.mse_climatology[var]
            df[f'{var}_relative_model'] = self.relative_mse_model[var]
            df[f'{var}_relative_climatology'] = self.relative_mse_climatology[var]
            df[f'{var}_rmse_model'] = self.rmse_model[var]
            df[f'{var}_rmse_climatology'] = self.rmse_climatology[var]
            df[f'{var}_skill_score'] = self.skill_score[var]
            df[f'{var}_std_rmse_model'] = self.std_model[var]
            df[f'{var}_std_rmse_climatology'] = self.std_climatology[var]
        # rajouter les moyennes de chaqque colonne 

        means = df.mean()
    
        # Ajouter une ligne de séparation (optionnel)
        df.loc['---'] = '---'

        # Ajouter la ligne des moyennes
        df.loc['mean'] = means


        df.to_csv(os.path.join(self.save_folder, 'mse_results.csv'), index=False)
        

    def save_results(self):
       # Créer des listes pour stocker les datasets
        all_preds = []
        all_truths = []
        all_clims = []
        # Parcourir tous les échantillons
        for i, (pred, truth, clim) in enumerate(zip(self.all_datasets_predictions, self.all_datasets_ground_truth, self.all_datasets_climatology)):
            # Ajouter une dimension 'sample' à chaque dataset
            pred = pred.expand_dims(sample=[i])
            truth = truth.expand_dims(sample=[i])
            clim = clim.expand_dims(sample=[i])
            
            all_preds.append(pred)
            all_truths.append(truth)
            all_clims.append(clim)

        # Combiner tous les datasets de vérité terrain
        combined_truths = xr.concat(all_truths, dim='sample')
        combined_preds = xr.concat(all_preds, dim='sample')
        combined_clims = xr.concat(all_clims, dim='sample')

        # Sauvegarder les datasets combinés
        combined_preds.to_netcdf(os.path.join(self.save_folder, 'all_predictions.nc'))
        combined_truths.to_netcdf(os.path.join(self.save_folder, 'all_ground_truths.nc'))
        combined_clims.to_netcdf(os.path.join(self.save_folder, 'all_climatology.nc'))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path

    # checkpoint_path = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240818_171920_every_coarse_2005_10/checkpoints/skill/model-skill-epoch=012-valid_skill_score=0.03.ckpt"

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
    all_dataset =  DatasetEra(all_config, data_dirs, temp_aggregator_factory, scaler).target_class.data.to_netcdf(os.path.join(exp_dir, '1940_2024_target.nc'))

    eval = Evaluation(checkpoint_path, config_path, test_dataset, scaler)
    print("ready to eval")
    eval.run_evaluation()

  