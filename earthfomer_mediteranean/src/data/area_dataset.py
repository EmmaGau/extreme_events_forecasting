import xarray as xr
import os
from typing import List
import xarray as xr
from typing import List, Dict
import numpy as np

class AreaDataset:
    def __init__(self, area: str, data: xr.Dataset,coarse_t: bool, coarse_s:bool,
                 temporal_resolution: Dict[str, int], spatial_resolution: int, years: List[int],
                 months: List[int], vars: List[str], target: str, sum_pr : bool = False, is_target: bool = False,
                 scaling_years: List[int] = None, scaling_mode: str = None):
        """This dataset class takes in input the xr.Dataset of the area (med, tropics or NH) and 
        preprocesses it to get the right temporal and spatial resolution. It then computes the statistics
        of the class using the scaling years, and the coarse temporal and spatial resolution. Finally, it scales the data
        using the statistics computed.

        Args:
            area (str): _description_
            data (xr.Dataset): _description_
            coarse_t (bool): _description_
            coarse_s (bool): _description_
            temporal_resolution (Dict[str, int]): _description_
            spatial_resolution (int): _description_
            years (List[int]): _description_
            months (List[int]): _description_
            vars (List[str]): _description_
            target (str): _description_
            sum_pr (bool, optional): _description_. Defaults to False.
            is_target (bool, optional): _description_. Defaults to False.
            scaling_years (List[int], optional): _description_. Defaults to None.
            scaling_mode (str, optional): _description_. Defaults to None.
        """
        self.area = area
        self.data = data
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.years = years
        self.months = months
        self.vars = vars
        self.target = target
        self.sum_pr = sum_pr
        self.is_target = is_target
        self.coarse_t = coarse_t
        self.coarse_s = coarse_s
        self.scaling_years = scaling_years
        self.scaling_mode = scaling_mode

        # first transform the data to get the right temporal and spatial resolution
        self._preprocess()
        # compute the statistics of the class using the scaling years, and the coarse temporal and spatial resolution
        self._compute_statistics()
        # only select the relevant years after computing the statistics to be computed on all the training years
        self.data = self.data.sel(time=self.data.time.dt.year.isin(self.years))
        # select the months needed (winter extended season)
        self.data = self.data.sel(time=self.data.time.dt.month.isin(self.months))
        # scale the data 
        self.scaled_data = self.scaling_transform(self.data)

    def _preprocess(self):
        if "tp" in self.vars:
            self.data["tp"] = self.data["tp"] * 1000
        self._change_temporal_resolution(self.temporal_resolution)
        self._change_spatial_resolution(self.spatial_resolution)

    def _change_temporal_resolution(self, temporal_resolution: int):
        self.data = self.data.drop_duplicates(dim='time')
        dates_to_remove = temporal_resolution - 1
        all_dates = self.data.time.values
        # The first dates are removed because they are NAN values after the rolling mean
        dates_to_keep = all_dates[dates_to_remove:]
        new_data = xr.Dataset()

        for var in self.vars:
            if var != "time_bnds":
                if np.issubdtype(self.data[var].dtype, np.number):
                    # Apply the non centered rolling mean (on the right)
                    rolled = self.data[var].rolling(time=temporal_resolution, center=False).mean()
                    # Select only the dates to keep
                    new_data[var] = rolled.sel(time=dates_to_keep)
                else:
                    new_data[var] = self.data[var].sel(time=dates_to_keep)
        # update the data
        self.data = new_data

    def _change_spatial_resolution(self,spatial_resolution: int):
        if spatial_resolution != 1:
            regridded_data = self.data.coarsen(latitude=spatial_resolution, longitude=spatial_resolution, boundary="trim").mean()
            if self.sum_pr:
                # for the precipitation we need to sum the values
                if 'tp' in self.data.variables:
                    tp_sum = self.data['tp'].coarsen(latitude=spatial_resolution, longitude=spatial_resolution, boundary="trim").sum()
                    regridded_data['tp'] = tp_sum
        
            self.data = regridded_data
    
    def _compute_statistics(self):
        years_in_data = set(self.data.time.dt.year.values)
        missing_years = set(self.scaling_years) - years_in_data

        if missing_years:
            raise ValueError(f"The following years are missing from the data: {missing_years}")
        scaling_data = self.data.sel(time=self.data.time.dt.year.isin(self.scaling_years))
        # one mean and std for every time point and grid cell
        if self.coarse_t and self.coarse_s:
            self.statistics = {
                "mean": scaling_data.mean(dim=['time', 'latitude', 'longitude']),
                "std": scaling_data.std(dim=['time', 'latitude', 'longitude'])
            }
        # one mean and std per grid cell, for every time point
        elif self.coarse_t:
            self.statistics = {
                "mean": scaling_data.mean(dim='time'),
                "std": scaling_data.std(dim='time')
            }
        # one mean and std per time point, for every grid cell
        elif self.coarse_s:
            grouped = scaling_data.groupby('time.dayofyear')
            self.statistics = {
                "mean": grouped.mean(dim=['time', 'latitude', 'longitude']),
                "std": grouped.std(dim=['time', 'latitude', 'longitude'])
            }
        # one mean and std per time point, per every grid cell
        else:
            grouped = scaling_data.groupby('time.dayofyear')
            self.statistics = {
                "mean": grouped.mean(dim='time'),
                "std": grouped.std(dim='time')
            }

        # Add a small epsilon to std where it equals 0
        epsilon = 1e-6
        self.statistics["std"] = xr.where(self.statistics["std"] == 0, epsilon, self.statistics["std"])

        # compute the climatology at the same time as the statistics
        self.climatology = {"mean":scaling_data.groupby('time.dayofyear').mean(dim='time'), "std": scaling_data.groupby('time.dayofyear').std(dim='time')}
        # compute the scaled climatology
        self.scaled_climatology = (self.climatology["mean"]- self.statistics["mean"]) / self.statistics["std"]
    
    def save_statistics(self):
        path_base = f"/home/egauillard/extreme_events_forecasting/earthfomer_mediteranean/src/statistics/"
        path_suffix = f"{min(self.scaling_years)}-{max(self.scaling_years)}_{'-'.join(map(str, self.months))}_{self.get_vars_stats_str()}_{self.area}_{self.spatial_resolution}deg_{self.temporal_resolution}days"
        
        if self.coarse_t:
            path_suffix += "_coarse_t"
        if self.coarse_s:
            path_suffix += "_coarse_s"
        if self.sum_pr:
            path_suffix += "_sum_pr"

        for stat_name, stat_data in self.statistics.items():
            path = f"{path_base}{stat_name}_{path_suffix}.nc"
            stat_data.to_netcdf(path)
    
    def scaling_transform(self, data):
        if self.scaling_mode == 'standardize':
            if 'dayofyear' in self.statistics["mean"].dims:
                dayofyear = data.time.dt.dayofyear
                return (data - self.statistics["mean"].sel(dayofyear=dayofyear)) / self.statistics["std"].sel(dayofyear=dayofyear)
            else:
                return (data - self.statistics["mean"]) / self.statistics["std"]
        else:
            raise ValueError(f"Unsupported scaling mode: {self.scaling_mode}")

    def inverse_transform(self, data):
        if self.scaling_mode == 'standardize':
            if 'dayofyear' in self.statistics["mean"].dims:
                dayofyear = data.time.dt.dayofyear
                return (data * self.statistics["std"].sel(dayofyear=dayofyear)) + self.statistics["mean"].sel(dayofyear=dayofyear)
            else:
                return (data * self.statistics["std"]) + self.statistics["mean"]
        else:
            raise ValueError(f"Unsupported scaling mode: {self.scaling_mode}")

    def get_vars_stats_str(self):
        return "_".join(self.vars)



if __name__ == "__main__":
    data_path = "/home/egauillard/data/PR_era5_MED_1degr_19400101_20240229_new.nc"
    data = xr.open_dataset(data_path)

    temporal_resolution = 7
    spatial_resolution = 3
    years = [2005,2007]
    months = [6,7,8]
    vars = ['tp']
    target = 'tp'
    sum_pr = True
    is_target = False
    coarse_t = False
    coarse_s = False
    scaling_years = list(range(1940, 2005 + 1)) # must be the whole
    scaling_mode = 'standardize'

    area_dataset = AreaDataset(area='mediteranean', data=data, temporal_resolution=temporal_resolution, spatial_resolution=spatial_resolution, years=years, months=months, vars=vars, target=target, sum_pr=sum_pr, is_target=is_target, coarse_t=coarse_t, coarse_s=coarse_s, scaling_years=scaling_years, scaling_mode=scaling_mode)