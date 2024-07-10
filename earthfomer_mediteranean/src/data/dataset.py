from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np 
import torch 
from utils.temporal_aggregator import TemporalAggregatorFactory
from utils.scaler import DataScaler
import wandb
from utils.tools import AreaDataset
from utils.enums import StackType, Resolution
from utils.statistics import DataStatistics

class DatasetEra(Dataset):
    # TODO in statistics add if target is "pr" then do sum else do mean
    def __init__(
        self,
        wandb_config : dict,
        data_dirs : str,
        temporal_aggr_factory : TemporalAggregatorFactory,
        scaler: DataScaler 
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
        self.scaler = scaler 
        self.global_variables = self.variables_med + self.variables_nh

        self.resolution_input = self.aggregator_factory.resolution_input
        self.resolution_output = self.aggregator_factory.resolution_output
        self.data, self.target = self._load_and_prepare_data()

        self.aggregator = self.aggregator_factory.create_aggregator(self.data, self.target)

        
        self.first_year = self.data.time.dt.year.min().item()
        self.last_year = self.data.time.dt.year.max().item()

        self.land_sea_mask = self.get_binary_sea_mask()
    
    def _initialize_config(self, wandb_config):
        """Initialize configuration settings."""
        ds_conf = wandb_config["dataset"]
        self.spatial_resolution = ds_conf["spatial_resolution"]
        self.target_var = ds_conf["target_variable"]
        self.variables_nh = ds_conf["variables_nh"]
        self.variables_med = ds_conf["variables_med"]
        self.mask_path = ds_conf["land_sea_mask"]
        self.relevant_months = ds_conf["relevant_months"]
        self.scaling_years = self.expand_year_range(ds_conf["scaling_years"])
        self.relevant_years =  self.expand_year_range(ds_conf["relevant_years"])
        self.predict_sea_land= ds_conf["predict_sea_land"]

    def process_scaling(self, data_class: AreaDataset, time: int):        
        resolution = self.get_scaling_resolution(time)
        self.stat_computer = DataStatistics(self.scaling_years, self.relevant_months, resolution)
        self.statistics = self.stat_computer._get_stats(data_class)
        
        # Assign the appropriate time unit based on the resolution
        time_unit = self.get_time_unit(resolution)
        data_class.data = data_class.data.assign_coords({time_unit: getattr(data_class.data.time.dt, time_unit)})
        
        # Align statistics with the data based on the resolution
        statistics = {
            key: self.statistics[key].sel({f'{resolution.value}': data_class.data[time_unit]}) 
            for key in self.statistics
        }
        
        # Apply scaling
        scaled_data = self.scaler.scale(data_class.data, statistics)
        
        return scaled_data

    def get_scaling_resolution(self, time):
        if 1 <= time <= 5:
            return Resolution.DAILY
        elif 5 < time <= 14:
            return Resolution.WEEKLY
        elif 14 < time <= 60:
            return Resolution.MONTHLY
        elif 60 < time <= 90:
            return Resolution.SEASON
        else:
            raise ValueError("Resolution not supported")

    def get_time_unit(self, resolution: Resolution):
        if resolution == Resolution.DAILY:
            return 'day'
        elif resolution == Resolution.WEEKLY:
            return 'week'
        elif resolution == Resolution.MONTHLY:
            return 'month'
        elif resolution == Resolution.SEASON:
            return 'season'
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")
            
        
    def expand_year_range(self,year_range):
        if len(year_range) != 2:
            raise ValueError("Year range must be specified as [start_year, end_year]")
        start_year, end_year = year_range
        return list(range(start_year, end_year + 1))

    def check_dataset(self, reference_dataset, new_dataset, variable_name):
        if reference_dataset is None or len(reference_dataset.data_vars) == 0:
            return reference_dataset, new_dataset

        reference_dataset['time'] = reference_dataset.time.dt.floor('D')
        new_dataset['time'] = new_dataset.time.dt.floor('D')
        
        # Supprime les temps en double s'il y en a
        reference_dataset = reference_dataset.drop_duplicates(dim='time')
        new_dataset = new_dataset.drop_duplicates(dim='time')
        
        common_time = np.intersect1d(reference_dataset.time, new_dataset.time)
        if len(common_time) != len(reference_dataset.time) or len(common_time) != len(new_dataset.time):
            print(f"Attention : Décalage temporel pour {variable_name}. Ajustement aux temps communs...")
            reference_dataset = reference_dataset.sel(time=common_time)
            new_dataset = new_dataset.sel(time=common_time)
        
        # Vérifie et ajuste la latitude et la longitude
        for dim in ['latitude', 'longitude']:
            if dim in reference_dataset.dims and dim in new_dataset.dims:
                common_coords = np.intersect1d(reference_dataset[dim], new_dataset[dim])
                if len(common_coords) != len(reference_dataset[dim]) or len(common_coords) != len(new_dataset[dim]):
                    print(f"Attention : Décalage de {dim} pour {variable_name}. Ajustement aux coordonnées communes...")
                    reference_dataset = reference_dataset.sel({dim: common_coords})
                    new_dataset = new_dataset.sel({dim: common_coords})
        
        return reference_dataset, new_dataset


    def _load_and_prepare_data(self):
        """Load data from directories for all variables and create big dataset that contains all variables for both regions
            and keep the relevant years/months."""
        med_datasets = []
        nh_datasets = []

        for var in self.variables_med:
            med_data = self._load_data(self.data_dirs['mediteranean'][var])
            if med_datasets:
                med_datasets[0], med_data = self.check_dataset(med_datasets[0], med_data, var)
            med_datasets.append(med_data)
        
        med_data = xr.merge(med_datasets, compat='override', join='inner')

        for var in self.variables_nh:
            nh_data = self._load_data(self.data_dirs['north_hemisphere'][var])
            if nh_datasets:
                nh_datasets[0], nh_data = self.check_dataset(nh_datasets[0], nh_data, var)
            nh_datasets.append(nh_data)
        
        if nh_datasets:
            nh_data = xr.merge(nh_datasets, compat='override', join='inner')
        else:
            nh_data = None

        self.med_class = AreaDataset("mediteranean", med_data, self.spatial_resolution, self.relevant_years, self.relevant_months, self.variables_med, self.target_var)
        self.nh_class = AreaDataset("north_hemisphere", nh_data, self.spatial_resolution, self.relevant_years, self.relevant_months, self.variables_nh, self.target_var)
        target_class = AreaDataset("target", med_data[[self.target_var]], self.spatial_resolution, self.relevant_years, self.relevant_months, [self.target_var], self.target_var)
        
        # scale med-data and nh_data 

        med_data = self.process_scaling(self.med_class, self.resolution_input)
        nh_data = self.process_scaling(self.nh_class, self.resolution_input)


        # change resolution if necessary
        if self.spatial_resolution !=1:
            med_data = self.change_spatial_resolution(med_data, self.spatial_resolution)
            nh_data = self.change_spatial_resolution(nh_data, self.spatial_resolution)
        
        print("Data loaded")
        # Remap Mediterranean to North Hemisphere if necessary
        if nh_data is not None:
            med_data = self.remap_MED_to_NH(nh_data, med_data)
            print("Data remapped")
        # merge the 2 datasets 
        data = xr.merge([med_data, nh_data], compat='override', join='inner')

        # target 
        target = self.process_scaling(target_class, self.resolution_output)[self.target_var]

        return  data, target
    
    def _load_data(self, dir_path):
        """Load data from a specified directory using xarray."""
        ds = xr.open_dataset(dir_path, chunks = None)
        ds = self._filter_data_by_time(ds)
        return ds
    
    def get_binary_sea_mask(self):
        mask = xr.open_dataset(self.mask_path)
        threshold = 0.3
        mask["lsm"] = xr.apply_ufunc(
            lambda x: (x > threshold).astype(int),
            mask["lsm"],
            dask="allowed",  # Si vous utilisez dask, sinon retirez cet argument
            keep_attrs=True  # Conserver les attributs
        )
        return mask["lsm"]

    def remap_MED_to_NH(self, nh_data, med_data):
        """Remap Mediterranean data to North Hemisphere grid and pad with zeros."""
        # Find the overlap region
        med_lon_min, med_lon_max = med_data.longitude.min().item(), med_data.longitude.max().item()
        med_lat_min, med_lat_max = med_data.latitude.min().item(), med_data.latitude.max().item()

        # Create a mask for the Mediterranean region in the NH grid
        mask = (
            (nh_data.longitude >= med_lon_min) & (nh_data.longitude <= med_lon_max) &
            (nh_data.latitude >= med_lat_min) & (nh_data.latitude <= med_lat_max)
        )

        # Get a reference variable from NH data for shape
        nh_ref_var = [var for var in nh_data.data_vars if var != 'time_bnds'][0]
        nh_shape = nh_data[nh_ref_var]

        # Create a new dataset
        remapped_data = xr.Dataset()

        # Fill in the Mediterranean region
        for var in med_data.data_vars:
            if var == 'time_bnds':
                # Just copy time_bnds as is
                remapped_data[var] = med_data[var].copy()
            else:
                # Check the structure of the current variable
                if set(med_data[var].dims) == set(nh_shape.dims):
                    # If dimensions match, create a zero-filled DataArray and fill with med data
                    remapped_var = xr.zeros_like(nh_shape)
                    med_values = med_data[var].broadcast_like(mask)
                    remapped_var = remapped_var.where(~mask, med_values)
                else:
                    # If dimensions don't match, just copy the original data
                    remapped_var = med_data[var].copy()
                
                # Add the variable to the new dataset
                remapped_data[var] = remapped_var

        return remapped_data

    def change_spatial_resolution(self, data, spatial_resolution):
        regridded_data = data.coarsen(latitude=spatial_resolution, longitude=spatial_resolution, boundary="trim").mean()
        return regridded_data

    def _filter_data_by_time(self, data):
        """Filter the data to include only the relevant months and years."""
        data = data.sel(time=data['time.year'].isin(self.relevant_years))
        data = data.sel(time=data['time.month'].isin(self.relevant_months))
        return data
    
    def __len__(self):
        len = self.aggregator.compute_len_dataset()
        return len
    
    def _prepare_sea_land_target(self, target_data):
        sea_means = []
        land_means = []
        
        for time in target_data.time:
            sea_mean = target_data.sel(time=time).where(self.land_sea_mask == 0).mean(dim=['latitude', 'longitude'])
            land_mean = target_data.sel(time=time).where(self.land_sea_mask == 1).mean(dim=['latitude', 'longitude'])
            
            sea_means.append(sea_mean)
            land_means.append(land_mean)
        
        sea_data = xr.concat(sea_means, dim='time').rename('sea_mean')
        land_data = xr.concat(land_means, dim='time').rename('land_mean')
        
        # Combine the sea and land data into a single dataset
        target_data_combined = xr.Dataset({'sea_mean': sea_data, 'land_mean': land_data})
        target_tensor = torch.tensor((np.transpose(np.array([[target_data_combined['sea_mean'].values, target_data_combined['land_mean'].values]]), (2,1,0))))
        print(target_tensor.shape)
        
        return target_tensor
    
    def _prepare_target(self, target_data):
        if self.predict_sea_land:
            target_tensor = self._prepare_sea_land_target(target_data)
        else: 
            target_array = np.transpose(np.array([target_data.values]), (1,2,3,0))
            target_tensor = torch.tensor(target_array)
        return target_tensor
        
    def __getitem__(self, idx):
        input_list = []
        # Aggregate the input data
        input_aggregated, target_aggregated, season_float, year_float = self.aggregator.aggregate(idx)

       # input data preparation
        for var in self.global_variables:
            input_list.append(input_aggregated[var].values)
            
        input_data_np = np.transpose(np.array(input_list), (1,2,3,0))
        input_tensor = torch.tensor(input_data_np)  # size (batch_size, height, width, channels)

        # target preparation
        target_tensor = self._prepare_target(target_aggregated) # size (batch_size, height, width, channels)
        print("target_tensor", target_tensor.shape)
        print("input_tensor", input_tensor.shape)
        # replace nan values by 0
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0)

        print(input_tensor.shape)
        print(target_tensor.shape)
        
        return input_tensor, target_tensor


    
if __name__ == "__main__":
    data_dirs = {'mediteranean': {'tp':"/home/egauillard/data/PR_era5_MED_1degr_19400101_20240229_new.nc",
                                  't2m':"/home/egauillard/data/T2M_era5_MED_1degr_19400101-20240229.nc"},
                 
                 'north_hemisphere': {"stream": "/home/egauillard/data/STREAM250_era5_NHExt_1degr_19400101_20240229_new.nc",
                                      "sst": "/home/egauillard/data/SST_era5_NHExt_1degr_19400101-20240229_new.nc",
                                      "msl": "/home/egauillard/data/MSLP_era5_NHExt_1degr_19400101_20240229_new.nc",
                                      }}

    wandb_config = {
    'dataset': {
        'variables_nh': ["stream","sst", "msl"],
        'variables_med': ['tp', "t2m"],
        'target_variable': 'tp',
        'relevant_years': [1940,1990],
        'relevant_months': [10,11,12,1,2,3],
        'scaling_years': [1940,1990],
        'land_sea_mask': '/home/egauillard/data/ERA5_land_sea_mask_1deg.nc',
        'spatial_resolution': 1,
        'predict_sea_land': False,
    },
    'scaler': {
        'mode': 'standardize'
    },
    'temporal_aggregator': {
        'stack_number_input': 3,
        'lead_time_number': 3,
        'resolution_input': 7,
        'resolution_output': 7,
        'gap': 1,
    }
}

    # Initialize wandb
    wandb.init(project='linear_era', config=wandb_config)

    # Initialize dataset and dataloaders
    scaler = DataScaler(wandb_config['scaler'])
    temp_aggregator_factory = TemporalAggregatorFactory(wandb_config['temporal_aggregator'])
    

    train_dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory, scaler)
    print("len dataset", train_dataset.__len__())

    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    sample = next(iter(dataloader))
    

    

