import xarray as xr
import numpy as np
import os
from typing import List, Dict
from utils.tools import AreaDataset

import xarray as xr
import os
from typing import List

# add an option parameter coarse scaling instead of having a scaling per day, we do the mean over all days of the year
# then we only have one statistics, also add this option to the path of the file to save the statistics

class DataScaler:
    def __init__(self, config):
        self.mode = config['mode']

    def inverse_transform(self, data, statistics, var):
        if self.mode == 'standardize':
            return self._inverse_standardize(data, statistics, var)
        else:
            raise ValueError(f"Unsupported scaling mode: {self.mode}")
    
    def transform(self, data, statistics, var):
        if self.mode == 'standardize':
            return self._standardize(data, statistics, var)
        else:
            raise ValueError(f"Unsupported scaling mode: {self.mode}")
    
    def _standardize(self, data: xr.DataArray, statistics: xr.Dataset, var: str):
        if 'dayofyear' in statistics["mean"][var].dims:
            dayofyear = data.time.dt.dayofyear
            standardized_data = (data[var]- statistics["mean"][var].sel(dayofyear=dayofyear)) / statistics["std"][var].sel(dayofyear=dayofyear)
        else:
            standardized_data = (data[var] - statistics["mean"][var]) / statistics["std"][var]
        return standardized_data

    def _inverse_standardize(self, data: xr.DataArray, statistics: xr.Dataset, var: str):
        if 'dayofyear' in statistics["mean"][var].dims:
            dayofyear = data.time.dt.dayofyear
            return (data[var] * statistics["std"][var].sel(dayofyear=dayofyear)) + statistics["mean"][var].sel(dayofyear=dayofyear)
        else:
            return (data [var]* statistics["std"][var]) + statistics["mean"][var]


#  add an parameter that spatial coarse scaling
# temporal zonal scaling as well 
# essaye sur tout le northern hemisphere 
# for the spatial : we do the mean over an area of grid point 
class DataStatistics:
    def __init__(self, years: List[int], months: List[int], coarse_temporal: bool = False, coarse_spatial: bool = False):
        self.years = years
        self.months = months
        self.coarse_temporal = coarse_temporal
        self.coarse_spatial = coarse_spatial
    
    def _get_years_months_str(self):
        years_range = f"{min(self.years)}-{max(self.years)}"
        months_str = '-'.join(map(str, self.months))
        return f"{years_range}_{months_str}"

    def get_vars_stats_str(self, data_class: 'AreaDataset'):
        vars = data_class.vars
        return "_".join(vars)

    def _get_stats(self, data_class: 'AreaDataset') -> Dict[str, xr.DataArray]:
        area = data_class.area
        spatial_resolution = data_class.spatial_resolution
        temporal_resolution = data_class.temporal_resolution

        path_base = f"/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/statistics/"
        path_suffix = f"{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_{temporal_resolution}days"
        
        if self.coarse_temporal:
            path_suffix += "_coarse"
        if self.coarse_spatial:
            path_suffix += "_coarse_spatial"
        if data_class.sum_pr:
            path_suffix += "_sum_pr"

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

    def _compute_stats(self, data_class: 'AreaDataset'):
        data = data_class.data

        # Sélectionner les années et mois spécifiés
        data = data.sel(time=data.time.dt.year.isin(self.years))
        data = data.sel(time=data.time.dt.month.isin(self.months))

        if self.coarse_temporal:
            if self.coarse_spatial:
                stats = {
                    "mean": data.mean(dim=['time', 'latitude', 'longitude']),
                    "std": data.std(dim=['time', 'latitude', 'longitude']),
                    "min": data.min(dim=['time', 'latitude', 'longitude']),
                    "max": data.max(dim=['time', 'latitude', 'longitude'])
                }
            else: 
                stats = {
                    "mean": data.mean(dim='time'),
                    "std": data.std(dim='time'),
                    "min": data.min(dim='time'),
                    "max": data.max(dim='time')
                }
        else:
            # Grouper par jour de l'année (ignorant l'année)
            grouped = data.groupby('time.dayofyear')
            stats = {
                "mean": grouped.mean(dim='time'),
                "std": grouped.std(dim='time'),
                "min": grouped.min(dim='time'),
                "max": grouped.max(dim='time')
            }

        # Ajouter un epsilon à std pour ceux qui valent 0
        epsilon = 1e-6
        stats["std"] = xr.where(stats["std"] == 0, epsilon, stats["std"])

        return stats
    
    def save_stats(self, data_class: 'AreaDataset', stats: Dict[str, xr.DataArray]):
        area = data_class.area
        spatial_resolution = data_class.spatial_resolution
        temporal_resolution = data_class.temporal_resolution
        
        if "statistics" not in os.listdir():
            os.mkdir("statistics")

        path_base = f"/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/statistics/"
        path_suffix = f"{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_{temporal_resolution}days"
        
        if self.coarse_temporal:
            path_suffix += "_coarse"
        if self.coarse_spatial:
            path_suffix += "_coarse_spatial"
        if data_class.sum_pr:
            path_suffix += "_sum_pr"

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