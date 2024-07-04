from typing import List, Dict
import xarray as xr
import os 
from utils.enums import Resolution
from utils.tools import AreaDataset 

class DataStatistics:
    def __init__(self, years : List[int], months : List[int], resolution: Resolution):
        self.resolution = resolution
        self.years = years
        self.months = months
    
    def _get_years_months_str(self):
        # Get the minimum and maximum years
        years_range = f"{min(self.years)}-{max(self.years)}"
        # Join the months into a string
        months_str = '-'.join(map(str, self.months))
        # Combine the years range and months string
        return f"{years_range}_{months_str}"
        
    def get_vars_stats_str(self, data_class : AreaDataset):
        vars = data_class.vars
        return "_".join(vars)

    def _get_stats(self, data_class : AreaDataset) -> Dict[str, float]:
        area = data_class.area
        spatial_resolution = data_class.spatial_resolution

        path_mean = f"statistics/{self.resolution.value}_{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_average.nc"
        path_std = f"statistics/{self.resolution.value}_{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_std.nc"
        path_min = f"statistics/{self.resolution.value}_{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_min.nc"
        path_max = f"statistics/{self.resolution.value}_{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_max.nc"

        if os.path.exists(path_mean) and os.path.exists(path_std) and os.path.exists(path_min) and os.path.exists(path_max):
            average = xr.open_dataset(path_mean)
            std = xr.open_dataset(path_std)
            min = xr.open_dataset(path_min)
            max = xr.open_dataset(path_max)
        else:
            average, std, min, max = self._compute_stats(data_class)
            self.save_stats(data_class, average, std, min, max)
        return {"mean" : average, "std" : std, "min": min, "max": max}
    
    def _compute_stats(self, data_class : AreaDataset):
        data = data_class.data
        # check we have the right year and months 
        data= data.sel(time = data.time.dt.year.isin(self.years))
        data = data.sel(time = data.time.dt.month.isin(self.months))
        print(data.groupby(f"time.{self.resolution.value}"))
       
        average = data.groupby(f"time.{self.resolution.value}").mean(dim = "time")
        std = data.groupby(f"time.{self.resolution.value}").std(dim = "time")
        min = data.groupby(f"time.{self.resolution.value}").min(dim = "time")
        max = data.groupby(f"time.{self.resolution.value}").max(dim = "time")
        return average, std, min, max
    
    def save_stats(self,data_class, average, std, min,max):
        area = data_class.area
        spatial_resolution = data_class.spatial_resolution
        if "statistics" not in os.listdir():
            os.mkdir("statistics")


        path_mean = f"statistics/{self.resolution.value}_{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_average.nc"
        path_std = f"statistics/{self.resolution.value}_{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_std.nc"
        path_min = f"statistics/{self.resolution.value}_{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_min.nc"
        path_max = f"statistics/{self.resolution.value}_{self._get_years_months_str()}_{self.get_vars_stats_str(data_class)}_{area}_{spatial_resolution}deg_max.nc"
        
        average.to_netcdf(path_mean)
        std.to_netcdf(path_std)
        min.to_netcdf(path_min)
        max.to_netcdf(path_max)


if __name__ == "__main__":
    years = [2005,2011]
    months = [6,7,8]
    area = "mediteranean"
    vars = ['tp']
    target = 'tp'
    spatial_resolution = 1

    data_path = "/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/PR/PR_era5_MED_1degr_19400101_20240229.nc"
    data = xr.open_dataset(data_path)

    data_class = AreaDataset(area, data, spatial_resolution, years, months, vars, target)
    stats = DataStatistics(Resolution.MONTHLY, years, months)
    print(stats._get_stats(data_class))