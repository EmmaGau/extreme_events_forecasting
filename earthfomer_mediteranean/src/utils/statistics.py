from typing import List, Dict
import xarray as xr
import os 
from enums import Resolution


class DataStatistics:
    def __init__(self, years : List[int] , months: List[int], spatial_resolution: int=1):
        """ Saves the data and the statistics for the given years and months.
        Compute the statistics with different level of resolution (week, day, month, season, year) over all the years and months given.
        If the statistics are already computed, it loads them, otherwise it computes them and saves them.

        Args:
            data (xr.DataArray): _description_
            years (List[int]): _description_
            months (List[int]): _description_
            resolution (Resolution): _description_
        """
        self.years = years
        self.months = months
        self.spatial_resolution = spatial_resolution

    def _get_stats(self, data: xr.DataArray, area : str,resolution: Resolution ) -> Dict[str, float]:
        path_mean, path_std  = f"data/{self.resolution.value}_{self.years}_{self.months}_average.nc", f"data/{self.resolution.value}_{self.years}_{self.months}_std.nc"
        path_min, path_max = f"data/{self.resolution.value}_{self.years}_{self.months}_min.nc", f"data/{self.resolution.value}_{self.years}_{self.months}_max.nc"
        
        if os.path.exists(path_mean) and os.path.exists(path_std) and os.path.exists(path_min) and os.path.exists(path_max):
            average = xr.open_dataarray(path_mean)
            std = xr.open_dataarray(path_std)
            min = xr.open_dataarray(path_min)
            max = xr.open_dataarray(path_max)
        else:
            average, std, min, max = self._compute_stats(data)
            self.save_stats(average, std, min, max)
        return {"mean" : average, "std" : std, "min": min, "max": max}
    
    def _compute_stats(self, data: xr.DataArray):
        # check we have the right year and months 
        data= data.sel(time = data.time.dt.year.isin(self.years))
        data = data.sel(time = data.time.dt.month.isin(self.months))
       
        average = data.groupby(f"time.{self.resolution.value}").mean(dim = "time")
        std = data.groupby(f"time.{self.resolution.value}").std(dim = "time")
        min = data.groupby(f"time.{self.resolution.value}").min(dim = "time")
        max = data.groupby(f"time.{self.resolution.value}").max(dim = "time")
        return average, std, min, max
    
    def save_stats(self, average, std, min,max):
        average.to_netcdf(f"data/{self.resolution.value}_{self.years}_{self.months}_average.nc")
        std.to_netcdf(f"data/{self.resolution.value}_{self.years}_{self.months}_std.nc")
        min.to_netcdf(f"data/{self.resolution.value}_{self.years}_{self.months}_min.nc")
        max.to_netcdf(f"data/{self.resolution.value}_{self.years}_{self.months}_max.nc")
    




        


        
        
        
