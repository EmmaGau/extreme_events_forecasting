from typing import List, Tuple
from torch.utils.data import Dataset
from extreme_events_forecasting.earthfomer_mediteranean.src.utils import enums
from enum import Enum
import xarray as xr
import numpy as np 
import torch 
from utils.temporal_aggregator import TemporalAggregator
import xesmf as xe
from utils.statistics import DataStatistics
from utils.scaler import DataScaler
from utils.enums import Resolution

# C'est quoi ces histoires de lost weights, clima as season group by data

# the dataset takes a data_dir mediteranean and data dir North Hemisphere
# oppens the data dir and reads the data with xarray, only take the 
# variables specified in the variables list
# then it remaps the data from mediteranean to North Hemisphere
# then it takes only the relevant months and years specified in the config
# uses the temporal aggregator to aggregate the input data 
# prepare the target data relative to the lead time + output resolution needed 
# then it scales the data using the scaler specified in the config
# then it returns the data both NH and MED


#(TODO) Climate as season, year float, month float

class DatasetEra(Dataset):
    def __init__(
        self,
        wandb_config : dict,
        data_dirs : str,
        temporal_aggregator : TemporalAggregator,
        scaler : DataScaler = None,
    ):
        self._initialize_config(wandb_config)
        self.stats_computer = DataStatistics(self.relevant_years, self.relevant_months,self.spatial_resolution)
        self.data_dirs = data_dirs
        self.scaler = scaler
        self.med_data, self.nh_data = self._load_and_prepare_data()
        self.temporal_aggregator_med, self.temporal_aggregator_nh = temporal_aggregator

        self.first_year = self.med_data.time.dt.year.min().item()
        self.last_year = self.med_data.time.dt.year.max().item()
        
    def _get_stats(self, data: xr.DataArray, resolution: Resolution):
        stats = self.stats_computer._get_stats(data, resolution)
        return stats
        
    def _initialize_config(self, wandb_config):
        """Initialize configuration settings."""
        ds_conf = wandb_config["dataset"]
        self.spatial_resolution = ds_conf["spatial_resolution"]
        self.target = ds_conf["target_variable"]
        self.variables_nh = ds_conf["variables_nh"]
        self.variables_med = ds_conf["variables_med"]
        self.land_sea_mask = ds_conf["land_sea_mask"]
        self.relevant_months = ds_conf["relevant_months"]
        self.relevant_years = ds_conf["relevant_years"]
    
    def change_spatial_resolution(self, data, spatial_resolution):
        regridded_data = data.coarsen(latitude=spatial_resolution, longitude=spatial_resolution, boundary="trim").mean()
        return regridded_data

    def _load_and_prepare_data(self):
        """Load data from directories for all variables and create big dataset that contains all variables for both regions
            and keep the relevant years/months."""
        med_data = xr.Dataset()
        nh_data = xr.Dataset()
        
        # Load Mediterranean data
        for var in self.variables_med:
            data = self._load_data(self.data_dirs['mediteranean'][var])
            med_data[var] = data

        # Load North Hemisphere data
        for var in self.variables_nh:
            data = self._load_data(self.data_dirs['north_hemisphere'][var])
            nh_data[var] = data
        
        # change resolution if necessary
        if self.spatial_resolution !=1:
            med_data = self.change_spatial_resolution(med_data, self.spatial_resolution)
            nh_data = self.change_spatial_resolution(nh_data, self.spatial_resolution)
        
        print("Data loaded")
        # Remap Mediterranean to North Hemisphere if necessary
        if nh_data is not None:
            med_data = self.remap_MED_to_NH(nh_data, med_data)
            print("Data remapped")
        return  med_data, nh_data
    
    def _load_data(self, dir_path):
        """Load data from a specified directory using xarray."""
        ds = xr.open_mfdataset(f"{dir_path}/*.nc", combine='by_coords')
        ds = self._filter_data_by_time(ds)
        return ds
    
    def remap_MED_to_NH(self, nh_data, med_data):
        """Remap Mediterranean data to North Hemisphere grid and pad with zeros."""
        # Empty array same dimensions and coordinates as nh_data
        print("nh_data",nh_data)
        fill_values = {var: 0 for var in nh_data.data_vars}
        remapped_data = xr.full_like(nh_data, fill_value=fill_values)
        print("remapped_data",remapped_data)

        # Find the overlap region and assign Mediterranean data to the remapped data
        med_lon_min, med_lon_max = med_data.min(dim = "longitude"), med_data.max(dim = "longitude")
        med_lat_min, med_lat_max = med_data.min(dim = "latitude"), med_data.max(dim = "latitude")

        # replace the values of the remapped data with the mediteranean data
        remapped_data = remapped_data.where(
            (remapped_data.longitude >= med_lon_min) & (remapped_data.longitude <= med_lon_max) &
            (remapped_data.latitude >= med_lat_min) & (remapped_data.latitude <= med_lat_max),
            other = med_data
        )

        return remapped_data

    def _filter_data_by_time(self, data):
        """Filter the data to include only the relevant months and years."""
        data = data.sel(time=data['time.year'].isin(self.relevant_years))
        data = data.sel(time=data['time.month'].isin(self.relevant_months))
        return data
    
    
    def __len__(self):
        len_med, len_nh = self.temporal_aggregator_med.compute_len_dataset(), self.temporal_aggregator_nh.compute_len_dataset()
        assert len_med == len_nh, "The length of the two datasets should be the same."
        return len_med
    

    def __getitem__(self, idx):
        # Aggregate the input data
        med_input_data, med_target_data, season_float, year_float = self.temporal_aggregator.aggregate(self.med_data, idx)
        nh_input_data, nh_target_data, _, _ = self.temporal_aggregator.aggregate(self.nh_data, idx)


        # Concatenate Mediterranean and North Hemisphere data along the variable dimension
        input_data = torch.cat([med_input_data, nh_input_data], dim=0)
        target_data = torch.cat([med_target_data], dim=0)

        return input_data, target_data, season_float, year_float


    


if __name__ == "__main__":
    # Générer des données d'exemple
    data = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
        (25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36),
    ]

    # Créer un Dataset
    dataset = DatasetEra(variables=[enums.T2M, enums.Precipitation, enums.SoilMoisture], data=data)

    # Afficher la taille du Dataset
    print(len(dataset))

    # Afficher un élément du Dataset
    print(dataset[0])