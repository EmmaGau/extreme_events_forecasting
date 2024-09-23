import numpy as np
import xarray as xr
import pandas as pd
import os
import xarray as xr
import numpy as np
import xarray as xr
import os
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import os
import numpy as np
import matplotlib.colors as mcolors

import numpy as np
import xarray as xr
import pandas as pd
import os
import scipy.stats as stats

#(TODO) génerer histogram rank + RPSS et BSS avec la climatology gaussienne + changer les couleurs des maps + bss centré en 0 + refactor le code et utilise transform data + les map de time series
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

class EnsembleVisualization:
    def __init__(self, prediction_files, climatology_mean_file, climatology_std_file, ground_truth_file, save_folder):
        self.prediction_files = prediction_files
        self.climatology_mean_file = climatology_mean_file
        self.climatology_std_file = climatology_std_file
        self.ground_truth_file = ground_truth_file
        self.save_folder = save_folder
        self.data = self.prepare_data()
        
    def transform_data(self, da):
        mask = ~np.isnan(da).any(dim=['latitude', 'longitude'])
        
        def get_valid_times(sample_mask):
            return np.where(sample_mask)[0]
        
        valid_time_indices = xr.apply_ufunc(
            get_valid_times,
            mask,
            input_core_dims=[['time']],
            output_core_dims=[['valid_time']],
            vectorize=True
        )
        
        new_da = da.isel(time=valid_time_indices)
        return new_da
    
    def prepare_data(self)->dict:
        """Prepare the data for visualization.

        Returns:
            dict: A dictionary containing various data arrays and statistics.
        """
        pred_ds = [self.transform_data(xr.open_dataset(f)['tp']) for f in self.prediction_files]
        climatology_mean = self.transform_data(xr.open_dataset(self.climatology_mean_file)['tp'])
        climatology_std = self.transform_data(xr.open_dataset(self.climatology_std_file)['tp'])
        gt_ds = self.transform_data(xr.open_dataset(self.ground_truth_file)['tp'])
        
        ensemble_mean = sum(pred_ds) / len(pred_ds)
        ensemble_std = xr.concat(pred_ds, dim='ensemble').std(dim='ensemble')
        
        ensemble_error = ensemble_mean - gt_ds
        climatology_error = climatology_mean - gt_ds
        
        ensemble_rmse = np.sqrt((ensemble_error**2))
        climatology_rmse = np.sqrt((climatology_error**2))
        
        return {
            'pred_ds': pred_ds,
            'climatology_mean': climatology_mean,
            'climatology_std': climatology_std,
            'gt_ds': gt_ds,
            'ensemble_mean': ensemble_mean,
            'ensemble_std': ensemble_std,
            'ensemble_rmse': ensemble_rmse,
            'climatology_rmse': climatology_rmse
        }
    
    def plot_rank_histogram(self, var):
        ranks = sum([pred < self.data['gt_ds'] for pred in self.data['pred_ds']])
        rank_hist, _ = np.histogram(ranks.values.flatten(), bins=len(self.data['pred_ds'])+1, range=(-0.5, len(self.data['pred_ds'])+0.5))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(rank_hist)), rank_hist, color='purple', alpha=0.7)
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Rank Histogram - {var}', fontsize=16)
        
        ideal_height = np.mean(rank_hist)
        ax.axhline(y=ideal_height, color='r', linestyle='--', label='Ideal uniform distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/{var}_rank_histogram.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_time_series(self, var, lat_idx=0, lon_idx=4):
        # Extract data for the selected grid cell
        gt_values = self.data["gt_ds"].isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        climatology_values = self.data["climatology_mean"].isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        ensemble_values = [ds.isel(latitude=lat_idx, longitude=lon_idx).values.flatten() for ds in self.data['pred_ds']]

        # Plot for every ensemble on different figures
        fig, axs = plt.subplots(len(ensemble_values), 1, figsize=(10, 4*len(ensemble_values)))
        for i, ensemble in enumerate(ensemble_values):
            axs[i].plot(np.arange(ensemble.size), ensemble, alpha=0.5, label=f"Ensemble {i+1}")
            axs[i].plot(climatology_values, color='black', label='Climatology')
            axs[i].plot(gt_values, color='red', label='Ground Truth')
            axs[i].legend()
            axs[i].set_title(f'Ensemble {i+1}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(var)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/{var}_time_series.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_time_series_spread(self, var, lat_idx=0, lon_idx=0):
        # Extract data for the selected grid cell
        gt_values = self.data["gt_ds"].isel(latitude=lat_idx, longitude=lon_idx)
        climatology_mean = self.data["climatology_mean"].isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        climatology_std = self.data["climatology_std"].isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        ensemble_mean = self.data["ensemble_mean"].isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        ensemble_std = self.data["ensemble_std"].isel(latitude=lat_idx, longitude=lon_idx).values.flatten()

        time = gt_values.time.values.flatten()
        gt_values = gt_values.values.flatten()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot climatology
        ax.fill_between(time, climatology_mean - climatology_std, climatology_mean + climatology_std, 
                        alpha=0.3, color='blue', label='Climatology Range')
        ax.plot(time,climatology_mean, color='blue', label='Climatology Mean')

        # Plot ensemble
        ax.fill_between(time,ensemble_mean - ensemble_std, ensemble_mean + ensemble_std, 
                        alpha=0.3, color='green', label='Ensemble Range')
        ax.plot(time,ensemble_mean, color='green', label='Ensemble Mean')

        # Plot ground truth
        ax.plot(time,gt_values, color='red', label='Ground Truth')

        ax.set_xlabel('Time')
        ax.set_ylabel(var)
        ax.set_title(f'Time Series Spread - {var} (Lat: {lat_idx}, Lon: {lon_idx})')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/{var}_time_series_spread.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_ensemble_sharpness(self, var):
        ensemble_mean = self.data['ensemble_mean'].mean(dim=['latitude', 'longitude'])
        ensemble_std = self.data['ensemble_std'].mean(dim=['latitude', 'longitude'])
        climatology_mean = self.data['climatology_mean'].mean(dim=['latitude', 'longitude'])
        climatology_std = self.data['climatology_std'].mean(dim=['latitude', 'longitude'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.linspace(ensemble_mean.min().values - 4*ensemble_std.max().values,
                        ensemble_mean.max().values + 4*ensemble_std.max().values, 100)
        
        for t in range(len(ensemble_mean)):
            ensemble_pdf = stats.norm.pdf(x, ensemble_mean[t], ensemble_std[t])
            climatology_pdf = stats.norm.pdf(x, climatology_mean[t], climatology_std[t])
            ax.plot(x, ensemble_pdf, 'b-', alpha=0.1)
            ax.plot(x, climatology_pdf, 'r-', alpha=0.1)
        
        ax.set_xlabel(var, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Sharpness Plot - {var}', fontsize=16)
        ax.legend(['Ensemble', 'Climatology'])
        
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/{var}_sharpness.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_ensemble_vs_climatology_rmse(self, var):
        ensemble_rmse = self.data['ensemble_rmse'].mean(dim=['latitude', 'longitude'])
        climatology_rmse = self.data['climatology_rmse'].mean(dim=['latitude', 'longitude'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(ensemble_rmse, climatology_rmse)
        
        max_rmse = max(ensemble_rmse.max().values, climatology_rmse.max().values)
        ax.plot([0, max_rmse], [0, max_rmse], 'r--', label='1:1 line')
        
        ax.set_xlabel('Ensemble RMSE', fontsize=12)
        ax.set_ylabel('Climatology RMSE', fontsize=12)
        ax.set_title(f'Ensemble vs Climatology RMSE - {var}', fontsize=16)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/{var}_ensemble_vs_climatology_rmse.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_spatial_rmse_maps(self, var):
        ensemble_rmse = self.data['ensemble_rmse']
        climatology_rmse = self.data['climatology_rmse']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        vmin = min(ensemble_rmse.min().values, climatology_rmse.min().values)
        vmax = max(ensemble_rmse.max().values, climatology_rmse.max().values)
        
        for ax, rmse, title in zip([ax1, ax2], 
                                   [ensemble_rmse, climatology_rmse],
                                   ['Ensemble RMSE', 'Climatology RMSE']):
            im = ax.imshow(rmse, cmap='viridis', transform=ccrs.PlateCarree(),
                           extent=[rmse.longitude.min(), rmse.longitude.max(), 
                                   rmse.latitude.min(), rmse.latitude.max()], 
                           vmin=vmin, vmax=vmax)
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', color='black', linewidth=0.5)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, label=f'{title} - {var}')
            ax.set_title(f'{title} - {var}\nVariance: {rmse.var().values:.4f}', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_folder}/{var}_spatial_rmse_maps.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_all_plots(self, var):
        self.plot_rank_histogram(var)
        self.plot_time_series(var)
        self.plot_time_series_spread(var)
        self.plot_ensemble_sharpness(var)
        self.plot_ensemble_vs_climatology_rmse(var)
        self.plot_spatial_rmse_maps(var)
        
class ModelEvaluation:
    def __init__(self, prediction_files, ground_truth_file, climatology_file, climatology_std_file, save_dir, entire_era_file):
        self.prediction_datasets = [self.transform_data(xr.open_dataset(file)) for file in prediction_files]
        self.ground_truth_dataset = self.transform_data(xr.open_dataset(ground_truth_file))
        self.climatology_dataset = self.transform_data(xr.open_dataset(climatology_file))
        self.climatology_std_dataset = self.transform_data(xr.open_dataset(climatology_std_file))
        self.era_entire_dataset = xr.open_dataset(entire_era_file)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_folder = os.path.join(self.save_dir, prediction_files[0].split('/')[-3])
        os.makedirs(self.save_folder, exist_ok=True)
        
        self.target_variables = list(self.ground_truth_dataset.data_vars)
        self.extreme_thresholds = self._calculate_extreme_thresholds()
        self.drought_thresholds = self._calculate_drought_thresholds()
        self.nb_ensemble = len(self.prediction_datasets)

    def transform_data(self, da):
        mask = ~np.isnan(da).any(dim=['latitude', 'longitude'])
        
        def get_valid_times(sample_mask):
            return np.where(sample_mask)[0]
        
        valid_time_indices = xr.apply_ufunc(
            get_valid_times,
            mask,
            input_core_dims=[['time']],
            output_core_dims=[['valid_time']],
            vectorize=True
        )
        
        new_da = da.isel(time=valid_time_indices)
        return new_da

    def _calculate_extreme_thresholds(self):
        return {var: self.era_entire_dataset[var].quantile(0.66, dim='time') for var in self.target_variables}

    def _calculate_drought_thresholds(self):
        return {var: self.era_entire_dataset[var].quantile(0.33, dim='time') for var in self.target_variables}

    def _replace_time_lt(self, da):
        lead_time = np.arange(1, len(da.valid_time) + 1)
        da_new = da.assign_coords(lead_time=('valid_time', lead_time)).swap_dims({'valid_time': 'lead_time'})
        return da_new.drop_vars('valid_time')
    
    def calculate_skill_scores(self,ensemble_score, climatology_score, valid_samples):
        """
        Calcule le Skill Score (RPSS ou BSS) à partir des scores de l'ensemble et de la climatologie.
        
        :param ensemble_score: Score de l'ensemble (RPS ou BS)
        :param climatology_score: Score de la climatologie (RPS ou BS)
        :param valid_samples: Masque des échantillons valides
        :return: Skill Score (RPSS ou BSS)
        """
        # Ajouter une petite valeur à climatology_score pour éviter la division par zéro
        epsilon = 1e-10
        climatology_score_safe = climatology_score + epsilon
        
        # Calculer le Skill Score
        skill_score = 1 - (ensemble_score / climatology_score_safe)
        
        # Gérer les cas spéciaux
        skill_score = xr.where(skill_score < -1, -1, skill_score)  # Limiter le score minimum à -1
        skill_score = xr.where((ensemble_score == 0) & (climatology_score == 0), 0, skill_score)  # Score de 0 si les deux sont zéro
        
        # Appliquer le masque des échantillons valides
        skill_score = skill_score.where(valid_samples > 0)
        
        return skill_score
    
    def safe_divide(self,a, b):
        """
        Effectue une division sûre, en convertissant le résultat en float64.
        """
        return (a.astype('float64') / b).astype('float64')
    
    def calculate_scores(self):
        self.rpss = {var: [] for var in self.target_variables}
        self.brier_scores_ensemble = {var: [] for var in self.target_variables}
        self.brier_scores_climatology = {var: [] for var in self.target_variables}
        drought_thresholds = self._calculte_droughts_thresholds()
        
        for var in self.target_variables:
            rps_ensemble = 0
            rps_climatology = 0
            valid_samples = 0
            bs_ensemble = 0
            bs_climatology = 0
            bs_drought_ensemble = 0
            bs_drought_climatology = 0
                
            for sample in range(len(self.ground_truth_dataset.sample)):
                truth = self.ground_truth_dataset[var].isel(sample=sample)
                clim = self.climatology_dataset[var].isel(sample=sample)
                
                preds = [pred_dataset[var].isel(sample=sample) for pred_dataset in self.prediction_datasets]
                
                truth_valid = truth.dropna(dim='time', how='all')
                clim_valid = clim.dropna(dim='time', how='all')
                preds_valid = [pred.dropna(dim='time', how='all') for pred in preds]
                
                if len(truth_valid) == 0:
                    continue

                sample_dates = truth_valid.time.values
                # remplacer time par lead time :

                truth_valid = self._replace_time_lt(truth_valid)
                clim_valid = self._replace_time_lt(clim_valid)
                preds_valid = [self._replace_time_lt(pred) for pred in preds_valid]
                
                categories = self._calculate_historical_categories(var, sample_dates)
                
                prob_ensemble = self._calculate_ensemble_probabilities(preds_valid, categories)
                prob_climatology = self._calculate_probabilities(clim_valid, categories)
                obs_categorical = self._categorize_observations(truth_valid, categories)

                threshold = self.extreme_thresholds[var]
                # Calcul des probabilités d'événements extrêmes pour l'ensemble
                prob_extreme_ensemble = (xr.concat(preds_valid, dim='ensemble') > threshold).mean(dim=['ensemble'])
                
                # Calcul des probabilités d'événements extrêmes pour la climatologie
                prob_extreme_climatology = (clim_valid > threshold).astype(int)
                
                # Observations des événements extrêmes
                obs_extreme = (truth_valid > threshold).astype(int)
                
                bs_ensemble += self._calculate_brier_score(prob_extreme_ensemble, obs_extreme)
                bs_climatology += self._calculate_brier_score(prob_extreme_climatology, obs_extreme)

                threshold_drought = drought_thresholds[var]
                prob_drought_ensemble = (xr.concat(preds_valid, dim='ensemble') < threshold_drought).mean(dim=['ensemble'])
                prob_drought_climatology = (clim_valid < threshold_drought).astype(int)
                obs_drought = (truth_valid < threshold_drought).astype(int)
                
                bs_drought_ensemble += self._calculate_brier_score(prob_drought_ensemble, obs_drought)
                bs_drought_climatology += self._calculate_brier_score(prob_drought_climatology, obs_drought)
                
                
                rps_ensemble += self._calculate_rps(prob_ensemble, obs_categorical)
                rps_climatology += self._calculate_rps(prob_climatology, obs_categorical)
                valid_samples += 1
            
            rpss = self.calculate_skill_scores(rps_ensemble, rps_climatology, valid_samples)
            bss = self.calculate_skill_scores(bs_ensemble, bs_climatology, valid_samples)
            bss_drought = self.calculate_skill_scores(bs_drought_ensemble, bs_drought_climatology, valid_samples)
    

            rps_ensemble = self.safe_divide(rps_ensemble, valid_samples)
            rps_climatology = self.safe_divide(rps_climatology, valid_samples)
            bs_ensemble = self.safe_divide(bs_ensemble, valid_samples)
            bs_climatology = self.safe_divide(bs_climatology, valid_samples)

            self.save_scores(var, rpss, bss, rps_ensemble, rps_climatology, bs_ensemble, bs_climatology)
            self.plot_rpss_maps(var,rpss,bss,bss_drought)

    def _calculate_ensemble_probabilities(self, preds, categories):
        probs = [self._calculate_probabilities(pred, categories) for pred in preds]
        return sum(probs) / len(probs)
    
    def _calculate_brier_score(self, forecast_probabilities, observations):
        return ((forecast_probabilities - observations) ** 2)
    
    def _calculate_probabilities(self, data, categories):
        quantile_0 = categories.isel(quantile=0)
        quantile_1 = categories.isel(quantile=1)
        
        below = (data <= quantile_0)
        middle = (data > quantile_0) & (data <= quantile_1)
        above = (data > quantile_1)
        middle = middle.assign_coords(quantile= 0.5)
        
        probs = xr.concat([below, middle, above], dim=pd.Index(['below', 'middle', 'above'], name='category'))
        return probs

    def _categorize_observations(self, observations, categories):
        below = (observations <= categories.isel(quantile=0)).astype(int)
        middle = ((observations > categories.isel(quantile=0)) & (observations <= categories.isel(quantile=1))).astype(int)
        above = (observations > categories.isel(quantile=1)).astype(int)
        middle = middle.assign_coords(quantile= 0.5)
        
        probs = xr.concat([below, middle, above], dim=pd.Index(['below', 'middle', 'above'], name='category'))
        return probs

    def _calculate_rps(self, forecast_probabilities, obs_probabilities):
        cumulative_forecast = forecast_probabilities.cumsum(dim='category')
        cumulative_obs = obs_probabilities.cumsum(dim='category')
        return ((cumulative_forecast - cumulative_obs) ** 2).mean(dim='category')
    
    def _calculate_historical_categories(self, var, sample_dates):
        historical_data = []
        seen_times = set()

        for date in sample_dates:
            pd_date = pd.Timestamp(date)
            same_day_month_samples = self.ground_truth_dataset[var].sel(
                time=self.ground_truth_dataset.time.dt.dayofyear == pd_date.dayofyear
            )
            for sample_idx in range(len(same_day_month_samples.sample)):
                sample_data = same_day_month_samples.isel(sample=sample_idx)
                sample_data_non_nan = sample_data.dropna(dim='time', how='all')
                sample_times = set(sample_data_non_nan.time.values)

                new_times = sample_times - seen_times
                if new_times:
                    seen_times.update(new_times)  # Mettre à jour les dates vues
                    historical_data.append(sample_data_non_nan.sel(time=list(new_times)))

        if historical_data:
            historical_data = xr.concat(historical_data, dim='time')
            terciles = historical_data.quantile([0.3333, 0.6667], dim='time')

            return terciles
        else:
            return xr.DataArray(np.nan, dims=['quantile', 'lat', 'lon'], coords={'quantile': [0.3333, 0.6667]})
        
    def plot_rpss_maps(self, var, rpss, bss_upper, bss_drought):
        def create_centered_map(data, title, filename, cbar_label):
            data_mean = data.mean(dim='lead_time')
            
            lats = [30] + list(data_mean.latitude.values) + [45]
            lons = [-10] + list(data_mean.longitude.values) + [40]

            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Centrer l'échelle de couleur autour de 0
            vmax = max(abs(data_mean.min().values), abs(data_mean.max().values), 0.2)
            vmin = -vmax
            
            im = ax.imshow(data_mean, cmap="coolwarm", transform=ccrs.PlateCarree(),
                        extent=[lons[0], lons[-1], lats[0], lats[-1]], vmin=vmin, vmax=vmax)
            
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', color='black', linewidth=0.5)
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='lightblue', alpha=0.3)
            
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            lat_margin = 0.5
            ax.set_extent([data_mean.longitude.min(), data_mean.longitude.max(),
                        data_mean.latitude.min() - lat_margin, data_mean.latitude.max() + lat_margin], 
                        crs=ccrs.PlateCarree())
            gl.xlocator = mticker.FixedLocator(np.arange(lons[0], lons[-1], 10))
            gl.ylocator = mticker.FixedLocator(np.arange(lats[0], lats[-1], 5))
            gl.top_labels = False
            gl.right_labels = False
            
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
            cbar.set_label(cbar_label, fontsize=12)
            
            plt.title(title, fontsize=16)
            ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], crs=ccrs.PlateCarree())
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder, filename), 
                        bbox_inches='tight', dpi=300)
            plt.close()

        # RPSS map
        create_centered_map(rpss, 
                            f'Mean RPSS - {var} - {self.nb_ensemble} ensemble',
                            f'{var}_rpss_map.png',
                            f'RPSS - {var}')

        # BSS (upper tercile) map
        create_centered_map(bss, 
                            f'Mean BSS (upper tercile) - {var} - {self.nb_ensemble} ensemble',
                            f'{var}_bss_map.png',
                            f'BSS - {var}')

        # BSS (Drought) map
        create_centered_map(bss_drought, 
                            f'Mean BSS (Drought) - {var} - {self.nb_ensemble} ensemble',
                            f'{var}_bss_drought_map.png',
                            f'BSS (Drought) - {var}')
                

    def save_scores(self, var, rpss, bss, rps_ensemble, rps_climatology, bs_ensemble, bs_climatology):
        rpss.name = f'{var}_rpss'
        bss.name = f'{var}_bss'
        
        # Sauvegarder les scores complets (avec toutes les dimensions)
        rpss.to_netcdf(os.path.join(self.save_folder, f'{var}_rpss.nc'))
        bss.to_netcdf(os.path.join(self.save_folder, f'{var}_bss.nc'))
        
        # Calculer et sauvegarder les moyennes spatiales
        rpss_mean = rpss.mean(dim=['latitude', 'longitude'])
        bss_mean = bss.mean(dim=['latitude', 'longitude'])
        
        df = pd.DataFrame({
            'lead_time': rpss.lead_time.values,
            f'{var}_rpss': rpss_mean.values,
            f'{var}_bss': bss_mean.values
        })

            # Calculer les moyennes pour rps et bs
        rps_ensemble_mean = rps_ensemble.mean(dim=['latitude', 'longitude'])
        rps_climatology_mean = rps_climatology.mean(dim=['latitude', 'longitude'])
        bs_ensemble_mean = bs_ensemble.mean(dim=['latitude', 'longitude'])
        bs_climatology_mean = bs_climatology.mean(dim=['latitude', 'longitude'])

        # Ajouter ces scores au DataFrame
        df[f'{var}_rps_ensemble'] = rps_ensemble_mean.values
        df[f'{var}_rps_climatology'] = rps_climatology_mean.values
        df[f'{var}_bs_ensemble'] = bs_ensemble_mean.values
        df[f'{var}_bs_climatology'] = bs_climatology_mean.values

        # Calculer la moyenne globale pour chaque score
        mean_row = pd.DataFrame({
            'lead_time': ['mean'],
            f'{var}_rpss': [rpss_mean.mean().values.item()],
            f'{var}_bss': [bss_mean.mean().values.item()],
            f'{var}_rps_ensemble': [rps_ensemble_mean.mean().values.item()],
            f'{var}_rps_climatology': [rps_climatology_mean.mean().values.item()],
            f'{var}_bs_ensemble': [bs_ensemble_mean.mean().values.item()],
            f'{var}_bs_climatology': [bs_climatology_mean.mean().values.item()]
        })

        # Concaténer la ligne de moyenne avec le DataFrame principal
        df = pd.concat([df, mean_row], ignore_index=True)
        
        # Sauvegarder le DataFrame dans un fichier CSV
        df.to_csv(os.path.join(self.save_folder, f'{var}_scores.csv'), index=False)

        
        print(f"Scores for {var} saved successfully.")

# Usage remains the same


if __name__ == '__main__':
    import argparse
    # Setup argparse to handle a list of prediction files
    parser = argparse.ArgumentParser()
    parser.add_argument('--predfiles', type=str, nargs='+', required=True, help="List of prediction files")
    args = parser.parse_args()

    # This will be a list of strings
    prediction_files = args.predfiles

    # Define other file paths
    climatology_file = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_145406_every_fine_19/inference_plots/all_climatology.nc"
    ground_truth_file = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_145406_every_fine_19/inference_plots/all_ground_truths.nc"
    entire_era_file = "/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/model/experiments/earthformer_era_20240822_145406_every_fine_19/inference_plots/target.nc"

    save_folder = '/home/egauillard/extreme_events_forecasting/primary_code/evaluation_results'

    # Pass the list of prediction files and other paths to your ModelEvaluation class
    evaluator = ModelEvaluation(prediction_files, ground_truth_file, climatology_file, save_folder, entire_era_file)
    evaluator.calculate_scores()

    ensemble_vis = EnsembleVisualization(prediction_files, climatology_file, ground_truth_file, save_folder)
    ensemble_vis.create_all_plots("tp")
