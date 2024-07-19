import xarray as xr
import numpy as np
import os
from typing import List, Dict
from utils.tools import AreaDataset

import xarray as xr
import os
from typing import List

class DataScaler:
    def __init__(self, config) -> None:
        self.mode = config['mode']
    
    def normalize(self,data, min, max):
        return (data - min) / (max - min)
    
    def standardize(self,data, mean, std):
        return (data - mean) / std

    def remove_outliers_23std(self, data, clipped_min, clipped_max):
        data = np.clip(data.data[0, :].astype(float), clipped_min, clipped_max)
        bottom = clipped_max - clipped_min
        bottom[bottom == 0] = "nan"
        data = (data - clipped_min) / bottom
        return np.nan_to_num(data, 0)

    def scale(self, data, stats):
        mean,std,min,max = stats.values()
        if self.mode == "normalize":
            return self.normalize(data, min, max)
        if self.mode == "std23":
            return self.remove_outliers_23std(data, min, max)
        elif self.mode == "standardize":
            return self.standardize(data, mean, std)
        else:
            raise ValueError(f"Unknown mode {self.mode}")


class DataStatistics:
    def __init__(self, years: List[int], months: List[int]):
        self.years = years
        self.months = months
    
    def _get_years_months_str(self):
        years_range = f"{min(self.years)}-{max(self.years)}"
        months_str = '-'.join(map(str, self.months))
        return f"{years_range}_{months_str}"

    def get_vars_stats_str(self, data_class: AreaDataset):
        vars = data_class.vars
        return "_".join(vars)

    def _get_stats(self, data_class: AreaDataset) -> Dict[str, xr.DataArray]:
        area = data_class.area
        spatial_resolution = data_class.spatial_resolution
        temporal_resolution = data_class.temporal_resolution

        path_base = f"/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/statistics/"
        path_suffix = f"{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_{temporal_resolution}days"

        paths = {
            "mean": f"{path_base}mean_{path_suffix}.nc",
            "std": f"{path_base}std_{path_suffix}.nc",
            "min": f"{path_base}min_{path_suffix}.nc",
            "max": f"{path_base}max_{path_suffix}.nc"
        }

        if all(os.path.exists(path) for path in paths.values()):
            return {key: xr.open_dataset(path) for key, path in paths.items()}
        else:
            stats = self._compute_stats(data_class)
            self.save_stats(data_class, stats)
            return stats

    def _compute_stats(self, data_class: AreaDataset):
        data = data_class.data

        # Sélectionner les années et mois spécifiés
        data = data.sel(time=data.time.dt.year.isin(self.years))
        data = data.sel(time=data.time.dt.month.isin(self.months))

        # Grouper par jour de l'année (ignorant l'année)
        grouped = data.groupby('time.dayofyear')
        

        stats = {
            "mean": grouped.mean(dim='time'),
            "std": grouped.std(dim='time'),
            "min": grouped.min(dim='time'),
            "max": grouped.max(dim='time')
        }
        # rajouter a std un epsilon pour ceux qui valent 0
        epsilon = 1e-6
        stats["std"] = xr.where(stats["std"] == 0, epsilon, stats["std"])

        return stats

    def save_stats(self, data_class, stats):
        area = data_class.area
        spatial_resolution = data_class.spatial_resolution
        temporal_resolution = data_class.temporal_resolution
        
        if "statistics" not in os.listdir():
            os.mkdir("statistics")

        path_base = f"/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/statistics/"
        path_suffix = f"{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_{temporal_resolution}days"

        for stat_name, stat_data in stats.items():
            path = f"{path_base}{stat_name}_{path_suffix}.nc"
            stat_data.to_netcdf(path)


if __name__ == "__main__":
    years = [2005,2011]
    months = [6,7,8]
    area = "mediteranean"
    vars = ['tp']
    target = 'tp'
    spatial_resolution = 1
    temporal_resolution = 1

    data_path = "/home/egauillard/data/PR_era5_MED_1degr_19400101_20240229_new.nc"
    data = xr.open_dataset(data_path)

    data_class = AreaDataset(area, data, spatial_resolution,temporal_resolution, years, months, vars, target)
    
    stats = DataStatistics(years, months)
    stats_dict = stats._get_stats(data_class)