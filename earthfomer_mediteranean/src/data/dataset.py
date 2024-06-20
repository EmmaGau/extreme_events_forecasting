from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np 
import torch 
from utils.temporal_aggregator import TemporalAggregatorFactory
from utils.scaler import DataScaler
import wandb
from utils.tools import AreaDataset

class DatasetEra(Dataset):
    def __init__(
        self,
        wandb_config : dict,
        data_dirs : str,
        temporal_aggr_factory : TemporalAggregatorFactory,
    ):
        """The dataset takes a data_dir mediteranean and data dir North Hemisphere
            oppens the data dir and reads the data with xarray, only take the 
            variables specified in the variables list
            then it remaps the data from mediteranean to North Hemisphere
            then it takes only the relevant months and years specified in the config
            uses the temporal aggregator to aggregate the input data 
            prepare the target data relative to the lead time + output resolution needed 
            then it scales the data using the scaler specified in the config
            then it returns the data both NH and MED

        Args:
            wandb_config (dict): _description_
            data_dirs (str): _description_
            temporal_aggregator (TemporalAggregator): _description_
            scaler (DataScaler, optional): _description_. Defaults to None.
        """
        self._initialize_config(wandb_config)
        self.data_dirs = data_dirs
        self.aggregator_factory = temporal_aggr_factory

        self.med_data, self.nh_data = self._load_and_prepare_data()
        self.med_class = AreaDataset("mediteranean", self.med_data, self.spatial_resolution, self.relevant_years, self.relevant_months, self.variables_med, self.target)
        self.med_aggregator = self.aggregator_factory.create_aggregator(self.med_class)
        
        if self.nh_data is not None:
            self.nh_class = AreaDataset("north_hemisphere", self.nh_data, self.spatial_resolution, self.relevant_years, self.relevant_months, self.variables_nh, self.target)
            self.nh_aggregator = self.aggregator_factory.create_aggregator(self.nh_class)

        self.first_year = self.med_data.time.dt.year.min().item()
        self.last_year = self.med_data.time.dt.year.max().item()
        self.compute_len_dataset()
        
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

    def _load_and_prepare_data(self):
        """Load data from directories for all variables and create big dataset that contains all variables for both regions
            and keep the relevant years/months."""
        med_data = xr.Dataset() if len(self.variables_med) !=0 else None
        nh_data = xr.Dataset() if len(self.variables_nh) !=0 else None
        
        # Load Mediterranean data
        for var in self.variables_med:
            data = self._load_data(self.data_dirs['mediteranean'][var])
            med_data[var] = data[var]
        # Load North Hemisphere data
        for var in self.variables_nh:
            data = self._load_data(self.data_dirs['north_hemisphere'][var])
            nh_data[var] = data[var]
        
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
        ds = xr.open_dataset(dir_path)
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
    
    def change_spatial_resolution(self, data, spatial_resolution):
        regridded_data = data.coarsen(latitude=spatial_resolution, longitude=spatial_resolution, boundary="trim").mean()
        return regridded_data

    def _filter_data_by_time(self, data):
        """Filter the data to include only the relevant months and years."""
        data = data.sel(time=data['time.year'].isin(self.relevant_years))
        data = data.sel(time=data['time.month'].isin(self.relevant_months))
        return data
    
    def compute_len_dataset(self):
        len_med = self.med_aggregator.compute_len_dataset()
        if self.nh_data is not None:
            len_nh = self.nh_aggregator.compute_len_dataset()
            assert len_med == len_nh, "The length of the two datasets should be the same."
        self.len_dataset = len_med
    
    def __len__(self):
        return self.len_dataset
    
    def _transform_target(self, target_data):
        target_data.where(self.land_sea_mask == 1).mean(dim=['latitude', 'longitude'])
        target_data.where(self.land_sea_mask == 0).mean(dim=['latitude', 'longitude'])
        return target_data
    
    def _prepare_target(self, target_data):

        pass
        


    def __getitem__(self, idx):
        input_data = []
        # Aggregate the input data
        med_input_aggregated, med_target_aggregated, _,_ = self.med_aggregator.aggregate(idx)
        if self.nh_data is not None:
            nh_input_aggregated, nh_target_aggregated, _, _ = self.nh_aggregator.aggregate(idx)
        
       
        for var in self.variables_med:
            print(med_input_aggregated[var].values.shape)
            input_data.append(med_input_aggregated[var].values)


        return input_data


    
if __name__ == "__main__":
    data_dirs = {'mediteranean': {'tp':"/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/PR/PR_era5_MED_1degr_19400101_20240229.nc"},
                 'north_hemisphere': {}}

    wandb_config = {
    'dataset': {
        'variables_nh': [],
        'variables_med': ['tp'],
        'target_variable': 'tp',
        'relevant_years': [1950, 1980],
        'relevant_months': [10,11,12,1,2,3],
        'land_sea_mask': '/scistor/ivm/shn051/extreme_events_forecasting/primary_code/data/ERA5_land_sea_mask_1deg.nc',
        'spatial_resolution': 1
    },
    'scaler': {
        'mode': 'standardize'
    },
    'temporal_aggregator': {
        'stack_number_input': 3,
        'lead_time_number': 3,
        'resolution_input': 7,
        'resolution_output': 7,
        'scaling_years': [1950, 1980],
        'scaling_months': [10,11,12,1,2,3], 
    }
}

    # Initialize wandb
    wandb.init(project='linear_era', config=wandb_config)

    # Initialize dataset and dataloaders
    scaler = DataScaler(wandb_config['scaler'])
    temp_aggregator_factory = TemporalAggregatorFactory(wandb_config['temporal_aggregator'], scaler)
    

    train_dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory)
    print(train_dataset.__len__())

    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    sample = next(iter(dataloader))
    print(sample)
