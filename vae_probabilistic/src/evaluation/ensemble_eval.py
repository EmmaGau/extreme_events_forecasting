import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
        
class EnsembleEvaluation:
    def __init__(self, pred_files, ground_truth_file, climatology_file, climatology_std_file, entire_era_file, var, save_dir):
        self.pred_ds = self.load_predictions(pred_files, var)
        self.ground_truth = self.transform_data(xr.open_dataset(ground_truth_file)[var])
        self.climatology = self.transform_data(xr.open_dataset(climatology_file)[var])
        self.climatology_std = self.transform_data(xr.open_dataset(climatology_std_file)[var])
        self.era_entire = xr.open_dataset(entire_era_file)[var]
        self.var = var
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.extreme_threshold = self.era_entire.quantile(0.66, dim='time')
        self.drought_threshold = self.era_entire.quantile(0.33, dim='time')
        self.categories = self.era_entire.quantile([0.3333, 0.6667], dim='time')

    def load_predictions(self, pred_files, var):
        if isinstance(pred_files, str):
            # Single NetCDF file with all ensembles
            return self.transform_data(xr.open_dataset(pred_files)[var])
        elif isinstance(pred_files, list):
            # List of NetCDF files, one for each ensemble member
            ensemble_list = [self.transform_data(xr.open_dataset(f)[var]) for f in pred_files]
            return xr.concat(ensemble_list, dim=pd.Index(range(len(ensemble_list)), name='ensemble'))
        else:
            raise ValueError("pred_files must be either a string (path to single NetCDF file) or a list of strings (paths to multiple NetCDF files)")

    def transform_data(self, da):
        mask = ~np.isnan(da).any(dim=['latitude', 'longitude'])
        valid_time_indices = xr.apply_ufunc(
            lambda x: np.where(x)[0],
            mask,
            input_core_dims=[['time']],
            output_core_dims=[['valid_time']],
            vectorize=True
        )
        return da.isel(time=valid_time_indices)


    def calculate_scores(self):
        climatology_ensemble = self._create_climatology_ensemble()
        obs_categorical = self._categorize_observations(self.ground_truth, self.categories)

        prob_ensemble = self._calculate_ensemble_probabilities(self.pred_ds, self.categories)
        prob_climatology = self._calculate_ensemble_probabilities(climatology_ensemble, self.categories)

        prob_extreme_ensemble = (self.pred_ds > self.extreme_threshold).mean(dim='ensemble')
        prob_extreme_climatology = (climatology_ensemble > self.extreme_threshold).mean(dim='ensemble')
        obs_extreme = (self.ground_truth > self.extreme_threshold).astype(int)

        prob_drought_ensemble = (self.pred_ds < self.drought_threshold).mean(dim='ensemble')
        prob_drought_climatology = (climatology_ensemble < self.drought_threshold).mean(dim='ensemble')
        obs_drought = (self.ground_truth < self.drought_threshold).astype(int)

        rps_ensemble = self._calculate_rps(prob_ensemble, obs_categorical)
        rps_climatology = self._calculate_rps(prob_climatology, obs_categorical)
        bs_ensemble = self._calculate_brier_score(prob_extreme_ensemble, obs_extreme)
        bs_climatology = self._calculate_brier_score(prob_extreme_climatology, obs_extreme)
        bs_drought_ensemble = self._calculate_brier_score(prob_drought_ensemble, obs_drought)
        bs_drought_climatology = self._calculate_brier_score(prob_drought_climatology, obs_drought)

        # Moyenne sur les échantillons avant de transformer les scores
        scores = [rps_ensemble, rps_climatology, bs_ensemble, bs_climatology, bs_drought_ensemble, bs_drought_climatology]
        scores = [score.mean(dim='sample') for score in scores]
        scores = [self.transform_scores(score) for score in scores]

        rpss = self.calculate_skill_score(scores[0], scores[1])
        bss = self.calculate_skill_score(scores[2], scores[3])
        bss_drought = self.calculate_skill_score(scores[4], scores[5])

        self.save_scores(rpss, bss, bss_drought, *scores)
        self.plot_maps(rpss, bss, bss_drought)
        self.create_all_plots()


    def _create_climatology_ensemble(self):
        ensemble_size = 10
        random_samples = xr.DataArray(
            np.random.normal(0, 1, self.climatology.shape + (ensemble_size,)),
            coords=self.climatology.coords,
            dims=self.climatology.dims + ('ensemble',)
        )
        return self.climatology + self.climatology_std * random_samples

    def _calculate_ensemble_probabilities(self, data, categories):
        probs = [
            (data <= categories.isel(quantile=0)).mean(dim='ensemble'),
            ((data > categories.isel(quantile=0)) & (data <= categories.isel(quantile=1))).mean(dim='ensemble'),
            (data > categories.isel(quantile=1)).mean(dim='ensemble')
        ]
        probs[1] = probs[1].assign_coords(quantile=0.5)
        return xr.concat(probs, dim=pd.Index(['below', 'middle', 'above'], name='category'))

    def _categorize_observations(self, observations, categories):
        cats = [
            (observations <= categories.isel(quantile=0)),
            (observations > categories.isel(quantile=0)) & (observations <= categories.isel(quantile=1)),
            (observations > categories.isel(quantile=1))
        ]
        cats[1] = cats[1].assign_coords(quantile=0.5)
        return xr.concat(cats, dim=pd.Index(['below', 'middle', 'above'], name='category')).astype(int)

    def _calculate_rps(self, forecast_probabilities, obs_probabilities):
        return ((forecast_probabilities.cumsum(dim='category') - obs_probabilities.cumsum(dim='category')) ** 2).mean(dim='category')

    def _calculate_brier_score(self, forecast_probabilities, observations):
        return (forecast_probabilities - observations) ** 2

    def transform_scores(self, da):
        lead_time = xr.DataArray(np.arange(da.sizes['valid_time']), dims='valid_time', name='lead_time')
        da = da.assign_coords(lead_time=lead_time)
        return da.groupby('lead_time').mean(dim=['valid_time'])

    def calculate_skill_score(self, score_ensemble, score_climatology):
        return 1 - (score_ensemble / (score_climatology + 1e-6))

    # we need to do the mean over the samples as well 
    def save_scores(self, rpss, bss, bss_droughts, rps_ensemble, rps_climatology, bs_ensemble, bs_climatology, bs_drought_ensemble, bs_drought_climatology):
        scores = [rpss, bss, bss_droughts, rps_ensemble, rps_climatology, bs_ensemble, bs_climatology, bs_drought_ensemble, bs_drought_climatology]
        names = ['rpss', 'bss', 'bss_drought', 'rps_ensemble', 'rps_climatology', 'bs_ensemble', 'bs_climatology', 'bs_drought_ensemble', 'bs_drought_climatology']
        
        df = pd.DataFrame({f'{self.var}_{name}': score.mean(dim=['latitude', 'longitude']).values 
                        for score, name in zip(scores, names)})
        df['lead_time'] = rpss.lead_time.values
        df = pd.concat([df, df.mean().to_frame().T.assign(lead_time='mean')])
        df.to_csv(os.path.join(self.save_dir, f'{self.var}_scores.csv'), index=False)

    def plot_maps(self, rpss, bss, bss_drought):
        for data, title in zip([rpss, bss, bss_drought], ['RPSS', 'BSS (upper tercile)', 'BSS (Drought)']):
            self.create_centered_map(data, f'Mean {title} - {self.var}', f'{self.var}_{title.lower()}_map.png', f'{title} - {self.var}')

    def create_centered_map(self, data, title, filename, cbar_label):
        data_mean = data.mean(dim='lead_time')
        lats = [30] + list(data_mean.latitude.values) + [45]
        lons = [-10] + list(data_mean.longitude.values) + [40]

        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        vmax = max(abs(data_mean.min().values), abs(data_mean.max().values), 0.2)
        im = ax.imshow(data_mean, cmap="coolwarm", transform=ccrs.PlateCarree(),
                       extent=[lons[0], lons[-1], lats[0], lats[-1]], vmin=-vmax, vmax=vmax)

        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', color='black', linewidth=0.5)
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='lightblue', alpha=0.3)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(range(-10, 41, 10))
        gl.ylocator = mticker.FixedLocator(range(30, 46, 5))
        gl.top_labels = gl.right_labels = False

        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, label=cbar_label)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()

    def plot_rank_histogram(self):
        ranks = sum([self.pred_ds.isel(ensemble=i) < self.ground_truth for i in range(self.pred_ds.sizes['ensemble'])])
        rank_hist, _ = np.histogram(ranks.values.flatten(), bins=self.pred_ds.sizes['ensemble']+1, range=(-0.5, self.pred_ds.sizes['ensemble']+0.5))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(rank_hist)), rank_hist, color='purple', alpha=0.7)
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Rank Histogram - {self.var}', fontsize=16)
        
        ideal_height = np.mean(rank_hist)
        ax.axhline(y=ideal_height, color='r', linestyle='--', label='Ideal uniform distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.var}_rank_histogram.png'), bbox_inches='tight', dpi=300)
        plt.close()

    def plot_time_series(self, lat_idx=0, lon_idx=0):
        time = self.ground_truth.time.values.flatten()
        gt_values = self.ground_truth.isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        climatology_values = self.climatology.isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        ensemble_values = [self.pred_ds.isel(latitude=lat_idx, longitude=lon_idx).isel(ensemble=i).values.flatten() for i in range(min(5, self.pred_ds.sizes['ensemble']))]

        fig, axs = plt.subplots(len(ensemble_values), 1, figsize=(10, 4*len(ensemble_values)))
        for i, ensemble in enumerate(ensemble_values):
            axs[i].plot(np.arange(ensemble.size), ensemble, alpha=0.5, label=f"Ensemble {i+1}")
            axs[i].plot(climatology_values, color='black', label='Climatology')
            axs[i].plot(gt_values, color='red', label='Ground Truth')
            axs[i].legend()
            axs[i].set_title(f'Ensemble {i+1}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel(self.var)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.var}_time_series.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_time_series_spread(self, lat_idx=0, lon_idx=0):
        time = self.ground_truth.time.values.flatten()
        gt_values = self.ground_truth.isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        climatology_mean = self.climatology.isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        climatology_std = self.climatology_std.isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        ensemble_mean = self.pred_ds.mean(dim='ensemble').isel(latitude=lat_idx, longitude=lon_idx).values.flatten()
        ensemble_std = self.pred_ds.std(dim='ensemble').isel(latitude=lat_idx, longitude=lon_idx).values.flatten()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot climatology
        ax.fill_between(time, 
                        climatology_mean - climatology_std, 
                        climatology_mean + climatology_std, 
                        alpha=0.3, color='blue', label='Climatology Range')
        ax.plot(time, climatology_mean, 
                color='blue', label='Climatology Mean')

        # Plot ensemble
        ax.fill_between(time, 
                        ensemble_mean - ensemble_std, 
                        ensemble_mean + ensemble_std, 
                        alpha=0.3, color='green', label='Ensemble Range')
        ax.plot(time, ensemble_mean, 
                color='green', label='Ensemble Mean')

        # Plot ground truth
        ax.plot(time, gt_values, 
                color='red', label='Ground Truth')

        ax.set_xlabel('Time')
        ax.set_ylabel(self.var)
        ax.set_title(f'Time Series Spread - {self.var} (Lat: {lat_idx}, Lon: {lon_idx})')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.var}_time_series_spread.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()

    def create_all_plots(self):
        self.plot_rank_histogram()
        self.plot_time_series()
        self.plot_time_series_spread()


if __name__ == '__main__':
    import argparse
    # Setup argparse to handle a list of prediction files
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--predfiles', type=str, nargs='+', required=True, help="List of prediction files")
    # args = parser.parse_args()

    # # This will be a list of strings
    # prediction_files = args.predfiles

    pred_file = "/home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240925_112910/inference_plots/all_ensemble_predictions.nc"
    var = "tp"
    # Define other file paths
    climatology_file = "/home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240925_112910/inference_plots/all_climatology.nc"
    ground_truth_file = "/home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240925_112910/inference_plots/all_ground_truths.nc"
    entire_era_file = "/home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240925_112910/1940_2024_target.nc"
    climatology_std_file = "/home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240925_112910/inference_plots/all_climatology_std.nc"
    save_folder = '/home/egauillard/extreme_events_forecasting/vae_probabilistic/experiments/VAE_20240925_112910/inference_plots/'

    # Pass the list of prediction files and other paths to your ModelEvaluation class
    evaluator = EnsembleEvaluation(pred_file, ground_truth_file, climatology_file, climatology_std_file, entire_era_file, var, save_folder)
    evaluator.calculate_scores()

    # ensemble_vis = EnsembleVisualization(prediction_files, climatology_file, ground_truth_file, save_folder)
    # ensemble_vis.create_all_plots("tp")
