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


class Evaluation:
    def __init__(self, checkpoint_path, config_path, test_dataset, scaler):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.test_dataset = test_dataset
        self.scaler = scaler
        self.save_folder = self.create_save_folder()
        self.model = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.statistics = test_dataset.statistics
    
    def inverse_scaling(self, data):
        return self.scaler.inverse_transform(data, self.statistics)

    def create_save_folder(self):
        exp_dir = os.path.dirname(os.path.dirname(self.checkpoint_path))
        save_folder = os.path.join(exp_dir, 'inference_plots')
        os.makedirs(save_folder, exist_ok=True)
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

        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=4
        )

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Processing batches"):
                input_data, target_data, season_float, year_float, input_time_indexes, target_time_indexes = batch
                
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)
                
                predictions, _, _, _ = self.model((input_data, target_data, season_float, year_float, input_time_indexes, target_time_indexes))
                
                all_inputs.append(input_data.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_ground_truth.append(target_data.cpu().numpy())
                all_input_time_indexes.append(self.test_dataset.aggregator.decode_time_indexes(input_time_indexes.numpy()))
                all_target_time_indexes.append(self.test_dataset.aggregator.decode_time_indexes(target_time_indexes.numpy()))


        self.all_inputs = np.concatenate(all_inputs, axis=0)
        self.all_predictions = np.concatenate(all_predictions, axis=0)
        self.all_ground_truth = np.concatenate(all_ground_truth, axis=0)
        self.all_input_time_indexes = np.concatenate(all_input_time_indexes, axis=0)
        self.all_target_time_indexes = np.concatenate(all_target_time_indexes, axis=0)

        # Calculate climatology
        self.climatology = self.test_dataset.compute_climatology()["mean"]


    def create_data_arrays(self):
        target_data = self.test_dataset.target_class.data
        lat_coords = target_data.latitude.values
        lon_coords = target_data.longitude.values

        self.all_datasets_predictions = []
        self.all_datasets_ground_truth = []
        self.all_datasets_climatology = []


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


            # Appliquer l'inverse scaling séparément
            ds_predictions = self.inverse_scaling(ds_predictions)
            ds_ground_truth = self.inverse_scaling(ds_ground_truth)

            self.all_datasets_predictions.append(ds_predictions)
            self.all_datasets_ground_truth.append(ds_ground_truth)
            self.all_datasets_climatology.append(ds_climatology)

        # to do ds_climatology :
        #pick the dates that are present in ds_predictions and use the value of the variables in self.climatology at those dates to create the climatology dataset


    def generate_plots(self):
        resolution_output = self.test_dataset.resolution_output
        for var_name in self.test_dataset.target_variables:
            for sample, (pred_dataset, truth_dataset) in enumerate(zip(self.all_datasets_predictions, self.all_datasets_ground_truth)):
                time_steps = pred_dataset.time.values
                n_lead_times = len(time_steps)

                fig, axs = plt.subplots(2, n_lead_times, figsize=(6*n_lead_times, 10), 
                                        subplot_kw={'projection': ccrs.PlateCarree()})
                
                if n_lead_times == 1:
                    axs = axs.reshape(2, 1)

                # Définir les limites de la carte
                lats = [30] + list(pred_dataset[var_name].latitude.values) + [45]
                lons = [-10] + list(pred_dataset[var_name].longitude.values) + [40]

                # Choisir la palette de couleurs en fonction de la variable
                cmap = 'Blues' if var_name == "tp" else 'Reds'

                for lead_time, time in enumerate(time_steps):
                    pred = pred_dataset[var_name].sel(time=time)
                    truth = truth_dataset[var_name].sel(time=time)

                    vmin = min(pred.min().item(), truth.min().item())
                    vmax = max(pred.max().item(), truth.max().item())

                    for row, data in enumerate([pred, truth]):
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
                        
                        # Supposons que 'time' est un objet datetime ou similaire
                        import pandas as pd

                        # Convertir numpy.datetime64 en datetime Python
                        python_datetime = pd.Timestamp(time).to_pydatetime()
                        title = f'Prediction average - {str(time)[:10]}' if row == 0 else f'Ground Truth average - {str(time)[:10]}'
                        ax.set_title(title, fontsize=10)

                # Ajuster la disposition des sous-graphiques
                plt.tight_layout()

                # Ajouter une barre de couleur commune
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(var_name, rotation=270, labelpad=15)

                plt.savefig(os.path.join(self.save_folder, f'{var_name}_sample_{sample}_all_lead_times.png'), 
                            bbox_inches='tight', dpi=300)
                plt.close()

    def calculate_mse(self):
        self.mse_model = {}
        self.mse_climatology = {}
        self.relative_mse_model = {}
        self.relative_mse_climatology = {}
        
        for var in self.test_dataset.target_variables:
            self.mse_model[var] = []
            self.mse_climatology[var] = []
            self.relative_mse_model[var] = []
            self.relative_mse_climatology[var] = []
            
            for lead_time in range(len(self.all_datasets_ground_truth[0].time)):
                truth = np.concatenate([ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_ground_truth])
                pred = np.concatenate([ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_predictions])
                clim = np.concatenate([ds[var].isel(time=lead_time).values.flatten() for ds in self.all_datasets_climatology])
                
                mse_model = np.mean((truth - pred)**2)
                mse_clim = np.mean((truth - clim)**2)
                
                range_squared = (np.max(truth) - np.min(truth))**2
                
                self.mse_model[var].append(mse_model)
                self.mse_climatology[var].append(mse_clim)
                self.relative_mse_model[var].append(mse_model / range_squared)
                self.relative_mse_climatology[var].append(mse_clim / range_squared)

    def plot_mse_curves(self):
        for var in self.test_dataset.target_variables:
            plt.figure(figsize=(12, 7))
            
            lead_times = range(len(self.relative_mse_model[var]))
            
            plt.plot(lead_times, self.relative_mse_model[var], 'o-', label='Model', markersize=6)
            plt.plot(lead_times, self.relative_mse_climatology[var], 's-', label='Climatology', markersize=6)
            
            plt.title(f'Relative MSE vs Lead Time - {var}', fontsize=16)
            plt.xlabel('Lead Time', fontsize=14)
            plt.ylabel('Relative MSE', fontsize=14)
            
            # Ajuster l'axe des abscisses
            plt.xticks(lead_times)
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # Améliorer la légende
            plt.legend(fontsize=12, loc='upper left')
            
            # Ajuster les marges
            plt.tight_layout()
            
            # Sauvegarder la figure
            plt.savefig(os.path.join(self.save_folder, f'{var}_relative_mse_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

        for var in self.test_dataset.target_variables:
            plt.figure(figsize=(12, 7))
            
            lead_times = range(len(self.mse_model[var]))
            
            plt.plot(lead_times, self.mse_model[var], 'o-', label='Model', markersize=6)
            plt.plot(lead_times, self.mse_climatology[var], 's-', label='Climatology', markersize=6)
            
            
            plt.title(f'MSE vs Lead Time - {var}', fontsize=16)
            plt.xlabel('Lead Time', fontsize=14)
            plt.ylabel('MSE', fontsize=14)
            
            # Ajuster l'axe des abscisses
            plt.xticks(lead_times)
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # Améliorer la légende
            plt.legend(fontsize=12, loc='upper left')
            
            # Ajuster les marges
            plt.tight_layout()
            
            # Sauvegarder la figure
            plt.savefig(os.path.join(self.save_folder, f'{var}mse_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()


    

    def calculate_spatial_mse(self):
        self.spatial_mse = {}
        self.spatial_relative_mse = {}
        for var in self.test_dataset.target_variables:
            mse = np.zeros_like(self.all_datasets_ground_truth[0][var].isel(time=0).values)
            truth_min = np.inf
            truth_max = -np.inf
            for ds_truth, ds_pred in zip(self.all_datasets_ground_truth, self.all_datasets_predictions):
                truth = ds_truth[var].values
                pred = ds_pred[var].values
                mse += np.mean((truth - pred)**2, axis=0)
                truth_min = min(truth_min, np.min(truth))
                truth_max = max(truth_max, np.max(truth))
            mse /= len(self.all_datasets_ground_truth)
            self.spatial_mse[var] = mse
            range_squared = (truth_max - truth_min)**2
            self.spatial_relative_mse[var] = mse / range_squared

    def plot_spatial_mse(self):
        for var, mse in self.spatial_relative_mse.items():
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

            # Définir les limites de la carte
            lats = [30] + list(self.all_datasets_ground_truth[0].latitude.values) + [45]
            lons = [-10] + list(self.all_datasets_ground_truth[0].longitude.values) + [40]

            # Tracer la carte de MSE
            im = ax.imshow(mse, cmap='YlOrRd', transform=ccrs.PlateCarree(),
                        extent=[lons[0], lons[-1], lats[0], lats[-1]])

            # Ajouter les caractéristiques de la carte
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', color='black', linewidth=0.5)
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='lightblue', alpha=0.3)

            # Ajouter une grille
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.xlocator = mticker.FixedLocator(range(-10, 41, 10))
            gl.ylocator = mticker.FixedLocator(range(30, 46, 5))
            gl.top_labels = False
            gl.right_labels = False

            # Ajouter une barre de couleur
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
            cbar.set_label(f'MSE - {var}', fontsize=12)

            # Définir le titre
            plt.title(f' Relative Spatial MSE - {var}', fontsize=16)

            # Ajuster la disposition
            plt.tight_layout()

            # Sauvegarder la figure
            plt.savefig(os.path.join(self.save_folder, f'{var}_spatial_mse.png'), 
                        bbox_inches='tight', dpi=300)
            plt.close()

    def save_mse_to_csv(self):
        df = pd.DataFrame()
        for var in self.test_dataset.target_variables:
            df[f'{var}_model'] = self.mse_model[var]
            df[f'{var}_climatology'] = self.mse_climatology[var]
            df[f'{var}_relative_model'] = self.relative_mse_model[var]
            df[f'{var}_relative_climatology'] = self.relative_mse_climatology[var]
        
        df.to_csv(os.path.join(self.save_folder, 'mse_results.csv'), index=False)
        

    def save_results(self):
        # Initialiser des listes pour stocker tous les datasets
        all_preds = []
        all_truths = []

        # Parcourir tous les échantillons
        for i, (pred, truth) in enumerate(zip(self.all_datasets_predictions, self.all_datasets_ground_truth)):
            # Ajouter une dimension 'sample' à chaque dataset
            pred = pred.expand_dims(sample=[i])
            truth = truth.expand_dims(sample=[i])
            
            all_preds.append(pred)
            all_truths.append(truth)

        # Combiner tous les datasets de prédiction
        combined_preds = xr.concat(all_preds, dim='sample')
        
        # Combiner tous les datasets de vérité terrain
        combined_truths = xr.concat(all_truths, dim='sample')

        # Sauvegarder les datasets combinés
        combined_preds.to_netcdf(os.path.join(self.save_folder, 'all_predictions.nc'))
        combined_truths.to_netcdf(os.path.join(self.save_folder, 'all_ground_truths.nc'))

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
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'cfg.yaml')

    oc = OmegaConf.load(config_path)
    dataset_cfg = OmegaConf.to_object(oc.data)
    test_config = dataset_cfg.copy()
    test_config['dataset']['relevant_years'] = [2011, 2023]  # Your TEST_YEARS
    data_dirs = dataset_cfg['data_dirs']
    scaler = DataScaler(dataset_cfg['scaler'])
    dataset_cfg['temporal_aggregator']['gap'] = dataset_cfg['temporal_aggregator']['resolution_output']
    temp_aggregator_factory = TemporalAggregatorFactory(dataset_cfg['temporal_aggregator'])

    test_dataset = DatasetEra(test_config, data_dirs, temp_aggregator_factory, scaler)

    evaluation = Evaluation(checkpoint_path, config_path, test_dataset, scaler)
    evaluation.run_evaluation()