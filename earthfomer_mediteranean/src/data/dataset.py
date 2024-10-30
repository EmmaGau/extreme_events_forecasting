from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import xclim
import wandb

from utils.temporal_aggregator import TemporalAggregatorFactory
from data.area_dataset import AreaDataset
from utils.enums import StackType, Resolution

class DatasetEra(Dataset):
    def __init__(
        self,
        wandb_config : dict,
        data_dirs : str,
        temporal_aggr_factory : TemporalAggregatorFactory,
        forecast_day : str = None):
        """A PyTorch Dataset for processing and combining climate data from multiple geographical regions.

        This dataset class loads, preprocesses, and combines climate data from the Mediterranean, Northern Hemisphere, and Tropical regions,
        using external classes for temporal aggregation and regional processing.
        It manages region alignment, merging, and provides samples for the DataLoader.

        Args:
            wandb_config (dict): Configuration dictionary containing dataset parameters:
            data_dirs (str):  Dictionary containing paths to data files:
                {
                    'mediteranean': {'var_name': 'path_to_file', ...},
                    'north_hemisphere': {'var_name': 'path_to_file', ...},
                    'tropics': {'var_name': 'path_to_file', ...}
                }
            temporal_aggregator_factory (TemporalAggregatorFactory):Factory for creating temporal aggregators
            scaler (DataScaler, optional): Specific forecast day to start from. Defaults to None.
        
        Notes:
            The workflow consists of:
            1. Loading data from multiple regions
            2. Checking temporal alignment across datasets
            3. Creating AreaDataset instances for each region
            4. Remapping Mediterranean and Tropical data to NH grid
            5. Merging all datasets
            6. Applying temporal aggregation
            7. Preparing input/target tensors for training
        """
        self._initialize_config(wandb_config)
        self.data_dirs = data_dirs
        self.aggregator_factory = temporal_aggr_factory
        self.global_variables = self.variables_med.copy() 

        if self.variables_nh is not None:
            self.global_variables.extend(self.variables_nh)
        
        if hasattr(self, 'variables_tropics') and self.variables_tropics is not None:
            self.global_variables.extend(self.variables_tropics)
        
        self.resolution_input = self.aggregator_factory.resolution_input
        self.resolution_output = self.aggregator_factory.resolution_output
        self.data, self.target = self._load_and_prepare_data()     

        # Only used for comparing with ECMWF to have overlapping days   
        if forecast_day is not None:
            self.select_for_specific_forecast_day(forecast_day)
        
        # need also the scaled climatolgy to compare with predicted values in the logging on wandb
        self.scaled_clim = self.get_scaled_climatology()

        # initialize the aggregator with the preprocessed data
        self.aggregator = self.aggregator_factory.create_aggregator(self.data, self.target, self.scaled_clim)

        self.land_sea_mask = self.get_binary_sea_mask()
        
    def _initialize_config(self, wandb_config):
        """Initialize configuration settings."""
        ds_conf = wandb_config["dataset"]
        self.spatial_resolution = ds_conf["spatial_resolution"]
        self.target_variables = ds_conf["target_variables"]
        self.variables_nh = ds_conf["variables_nh"]
        self.variables_med = ds_conf["variables_med"]
        self.mask_path = ds_conf["land_sea_mask"]
        self.relevant_months = ds_conf["relevant_months"]
        self.scaling_years = self.expand_year_range(ds_conf["scaling_years"])
        self.relevant_years =  self.expand_year_range(ds_conf["relevant_years"])
        self.predict_sea_land= ds_conf["predict_sea_land"]
        self.out_spatial_resolution = ds_conf.get("out_spatial_resolution", 1.0)  # Default to 1.0 if not specified
        self.sum_pr = ds_conf.get("sum_pr", False)
        self.variables_tropics = ds_conf.get("variables_tropics", None)
        self.coarse_t = ds_conf.get("coarse_t", True)
        self.coarse_s = ds_conf.get("coarse_s", False)
        self.coarse_t_target = ds_conf.get("coarse_t_target", True)
        self.coarse_s_target = ds_conf.get("coarse_s_target", False)
        self.scaling_mode = wandb_config["scaler"]["mode"]
        
        # Because name of parameters have changed through the project
        if "coarse" in wandb_config["scaler"].keys():
            self.coarse_t = wandb_config["scaler"]["coarse"]
            self.coarse_t_target = wandb_config["scaler"]["coarse"]
            print(f"Applying {'coarse' if self.coarse_t else 'fine-grained'} temporal scaling for both input and target data.")
            
    def select_for_specific_forecast_day(self, forecast_day):
        """This function selects only the data from a specific forecast day onward"""
        input_length = self.aggregator_factory.in_len*self.resolution_input + self.aggregator_factory.lead_time_gap

        # Adjust the forecast day by the output resolution to align with the right-anchored rolling mean.
        input_start_date = pd.Timestamp(forecast_day) + pd.Timedelta(days= self.resolution_output) - pd.Timedelta(days=input_length)
        end_date = self.data.time[-1].values
        
        self.data = self.data.sel(time=slice(input_start_date, end_date))
        self.target = self.target.sel(time=slice(input_start_date, end_date))

    def _load_data(self, dir_path):
        """Load data from a specified directory using xarray."""
        ds = xr.open_dataset(dir_path)
        return ds
            
    def expand_year_range(self, year_range):
        """Expand a year range into a list of years."""
        if len(year_range) != 2:
            raise ValueError("Year range must be specified as [start_year, end_year]")
        start_year, end_year = year_range
        return list(range(start_year, end_year + 1))

    def check_dataset(self, reference_dataset, new_dataset, variable_name):
        """Check if the new dataset is  temporaly and spatially aligned with the reference dataset and adjust if necessary."""
        if reference_dataset is None or len(reference_dataset.data_vars) == 0:
            return reference_dataset, new_dataset

        reference_dataset['time'] = reference_dataset.time.dt.floor('D')
        new_dataset['time'] = new_dataset.time.dt.floor('D')
        
        # Remove duplicate times
        reference_dataset = reference_dataset.drop_duplicates(dim='time')
        new_dataset = new_dataset.drop_duplicates(dim='time')

        #Align the time dimension
        common_time = np.intersect1d(reference_dataset.time, new_dataset.time)
        if len(common_time) != len(reference_dataset.time) or len(common_time) != len(new_dataset.time):
            print(f"Attention : Décalage temporel pour {variable_name}. Ajustement aux temps communs...")
            reference_dataset = reference_dataset.sel(time=common_time)
            new_dataset = new_dataset.sel(time=common_time)
        
        # Align the spatial dimensions
        for dim in ['latitude', 'longitude']:
            if dim in reference_dataset.dims and dim in new_dataset.dims:
                common_coords = np.intersect1d(reference_dataset[dim], new_dataset[dim])
                if len(common_coords) != len(reference_dataset[dim]) or len(common_coords) != len(new_dataset[dim]):
                    print(f"Attention : Décalage de {dim} pour {variable_name}. Ajustement aux coordonnées communes...")
                    reference_dataset = reference_dataset.sel({dim: common_coords})
                    new_dataset = new_dataset.sel({dim: common_coords})
        
        return reference_dataset, new_dataset

    def get_scaled_climatology(self):
        """ Get the scaled climatology for the target class to compute skill scores in the logging."""
        return self.target_class.scaled_climatology
        
    def _load_and_prepare_data(self):
        """Main function of the class : loads data from directories, preprocess, and merge datasets.
        Returns:
            tuple: A pair of xarray.Dataset objects:
                        - data: Combined and processed input data from all regions
                        - target: Processed target variables from Mediterranean region
        """
        med_datasets = []
        nh_datasets = []
        tropics_datasets = []

       # Load and align Mediterranean datasets
        for var in self.variables_med:
            # Align new data with existing datasets temporally and spatially
            med_data = self._load_data(self.data_dirs['mediteranean'][var])
            if med_datasets:
                med_datasets[0], med_data = self.check_dataset(med_datasets[0], med_data, var)
            med_datasets.append(med_data)
        
        med_data = xr.merge(med_datasets, compat='override', join='inner')

        # Load and align Northern Hemisphere datasets
        for var in self.variables_nh:
            nh_data = self._load_data(self.data_dirs['north_hemisphere'][var])
            if nh_datasets:
                nh_datasets[0], nh_data = self.check_dataset(nh_datasets[0], nh_data, var)
            nh_datasets.append(nh_data)
        
        if nh_datasets:
            nh_data = xr.merge(nh_datasets, compat='override', join='inner')
        else:
            nh_data = None
        
        # Load and align Tropical datasets if specified
        if hasattr(self, 'variables_tropics') and self.variables_tropics:
            for var in self.variables_tropics:
                tropics_data = self._load_data(self.data_dirs['tropics'][var])
                if tropics_datasets:
                    tropics_datasets[0], tropics_data = self.check_dataset(tropics_datasets[0], tropics_data, var)
                else:
                    tropics_data['time'] = tropics_data.time.dt.floor('D')
                tropics_datasets.append(tropics_data)
            
            if tropics_datasets:
                tropics_data = xr.merge(tropics_datasets, compat='override', join='inner')
            else:
                tropics_data = None
        else:
            tropics_data = None

        # Create AreaDataset instances for each region
        self.med_class = self._create_area_dataset("mediteranean", med_data, self.variables_med)
        self.target_class = self._create_area_dataset("target", med_data[self.target_variables], self.target_variables, is_target=True)
        self.nh_class = self._create_area_dataset("north_hemisphere", nh_data, self.variables_nh) if nh_data is not None else None
        self.tropics_class = self._create_area_dataset("tropics", tropics_data, self.variables_tropics) if tropics_data is not None else None
        
        # Extract scaled data from each region
        target = self.target_class.scaled_data 
        med_data = self.med_class.scaled_data
        nh_data = self.nh_class.scaled_data if nh_data is not None else None
        tropics_data = self.tropics_class.scaled_data if tropics_data is not None else None

        # Combine all regional data into a single dataset
        data = self._remap_and_merge_data(med_data, nh_data, tropics_data)

        return data, target

    def _create_area_dataset(self, area, data, variables, is_target=False):
        """ Create an AreaDataset instance for a specific region or for target
        
            Args:
                area (str): Region identifier ('mediteranean', 'north_hemisphere', 'tropics', or 'target')
                data (xarray.Dataset): Raw data for the region
                variables (List[str]): Variables to process for this region
                is_target (bool, optional): Whether this is target data. Defaults to False.

            Returns:
                AreaDataset: Processed dataset for the region, or None if data is None
        """
        if data is None:
            return None
        return AreaDataset(
            area=area,
            data=data,
            spatial_resolution=self.out_spatial_resolution if is_target else self.spatial_resolution,
            temporal_resolution=self.resolution_output if is_target else self.resolution_input,
            years=self.relevant_years,
            months=self.relevant_months,
            vars=variables,
            target=self.target_variables,
            sum_pr = self.sum_pr,
            is_target = is_target,
            coarse_t = self.coarse_t_target if is_target else self.coarse_t,
            coarse_s =  self.coarse_s_target if is_target else self.coarse_s,
            scaling_years = self.scaling_years,
            scaling_mode = self.scaling_mode)

    def _remap_and_merge_data(self, med_data, nh_data, tropics_data):
        """Remap and merge data from different regions to a common Northern Hemisphere grid.

            Args:
                med_data (xarray.Dataset): Mediterranean region data
                nh_data (xarray.Dataset): Northern Hemisphere data
                tropics_data (xarray.Dataset): Tropical region data

            Returns:
                xarray.Dataset: Combined dataset on the NH grid with all variables

            Note:
                The method ensures temporal alignment across all datasets and
                remaps Mediterranean and Tropical data to the NH grid structure.
        """
        # Find common timestamps across all available datasets
        common_times = med_data.time
        if nh_data is not None:
            common_times = np.intersect1d(common_times, nh_data.time)
        if tropics_data is not None:
            common_times = np.intersect1d(common_times, tropics_data.time)

        # Align all datasets to common timestamps
        med_data = med_data.sel(time=common_times)
        if nh_data is not None:
            nh_data = nh_data.sel(time=common_times)
        if tropics_data is not None:
            tropics_data = tropics_data.sel(time=common_times)

        # Initialize with NH data if available, otherwise use Mediterranean data
        data_list = [nh_data]  if nh_data is not None else [med_data]

        # Remap Mediterranean data to NH grid if NH data is available
        if nh_data is not None:
            remapped_med = self.remap_to_NH(nh_data, med_data)
            data_list.append(remapped_med)
            print("Mediterranean data remapped to NH")

        # Remap Tropical data to NH grid if available
        if tropics_data is not None:
            remapped_tropics = self.remap_to_NH(nh_data, tropics_data)
            data_list.append(remapped_tropics)
            print("Tropical data remapped to NH")

        # Merge all datasets
        data = xr.merge(data_list, compat='override', join='inner')
        return data

    def remap_to_NH(self, nh_data, data_to_remap):
        """
        Remap data to North Hemisphere grid and pad with zeros.
        
        Args:
        nh_data (xarray.Dataset): The reference Northern Hemisphere dataset.
        data_to_remap (xarray.Dataset): The dataset to be remapped.
        
        Returns:
        xarray.Dataset: The remapped data on the NH grid.
        """
        # Find the overlap region
        lon_min, lon_max = data_to_remap.longitude.min().item(), data_to_remap.longitude.max().item()
        lat_min, lat_max = data_to_remap.latitude.min().item(), data_to_remap.latitude.max().item()

        # Create a mask for the region in the NH grid
        mask = (
            (nh_data.longitude >= lon_min) & (nh_data.longitude <= lon_max) &
            (nh_data.latitude >= lat_min) & (nh_data.latitude <= lat_max)
        )

        # Get a reference variable from NH data for shape
        nh_ref_var = [var for var in nh_data.data_vars if var != 'time_bnds'][0]
        nh_shape = nh_data[nh_ref_var]

        # Create a new dataset
        remapped_data = xr.Dataset()

        # Fill in the region
        for var in data_to_remap.data_vars:
            if var == 'time_bnds':
                # Just copy time_bnds as is
                remapped_data[var] = data_to_remap[var].copy()
            else:
                # Check the structure of the current variable
                if set(data_to_remap[var].dims) == set(nh_shape.dims):
                    # Ensure that the time dimension is aligned
                    common_times = np.intersect1d(data_to_remap.time, nh_data.time)
                    data_slice = data_to_remap[var].sel(time=common_times)
                    nh_slice = nh_data[nh_ref_var].sel(time=common_times)

                    # Interpolate data to NH grid
                    interpolated = data_slice.interp(
                        latitude=nh_slice.latitude,
                        longitude=nh_slice.longitude,
                        method='nearest'
                    )
                    
                    # Create a zero-filled DataArray and fill with interpolated data
                    remapped_var = xr.zeros_like(nh_slice)
                    remapped_var = remapped_var.where(~mask, interpolated)
                else:
                    # If dimensions don't match, just copy the original data
                    remapped_var = data_to_remap[var].copy()
                
                # Add the variable to the new dataset
                remapped_data[var] = remapped_var

        return remapped_data
            
    def __len__(self):
        # With the temporal aggregation, the length of the dataset is computed differently
        len = self.aggregator.compute_len_dataset()
        return len
    
    def get_binary_sea_mask(self):
        # TODO: not used for now, make it work
        """Generate a binary land-sea mask from the source mask file."""
        mask = xr.open_dataset(self.mask_path, chunks = None)
        threshold = 0.3
        mask["lsm"] = xr.apply_ufunc(
            lambda x: (x > threshold).astype(int),
            mask["lsm"],
            dask="allowed",  # Si vous utilisez dask, sinon retirez cet argument
            keep_attrs=True  # Preserve metadata attributes
        )
        return mask["lsm"]

    def _prepare_sea_land_target(self, target_data):
        # TODO: not used for now, make it work
        """Calculate separate sea and land means for target variables."""
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
        
        return target_tensor
    
    def _prepare_target(self, target_data):
        """Prepare target data for training by converting to appropriate tensor format."""
        if self.predict_sea_land:
            target_tensor = self._prepare_sea_land_target(target_data)
        else: 
            target_list = []
            for var in self.target_variables:
                target_list.append(target_data[var].values)
            
            target_data_np = np.transpose(np.array(target_list), (1,2,3,0))
            target_tensor = torch.tensor(target_data_np)  # size (time, height, width, channels)
        return target_tensor
        
    def __getitem__(self, idx):
        """Implement PyTorch Dataset interface for data loading.

            This method prepares a single batch of data including input features,
            target variables, and associated metadata.

            Args:
                idx (int): Index of the sample to retrieve

            Returns:
                tuple: Contains the following elements:
                    - input_tensor (torch.Tensor): Input data of shape (time, height, width, channels)
                    - target_tensor (torch.Tensor): Target data, shape depends on prediction mode
                    - season_float (float): Season indicator for temporal context
                    - year_float (float): Year indicator for temporal context
                    - clim_tensor (torch.Tensor): Climatology data for reference
                    - input_time_tensor (torch.Tensor): Time indices for input data
                    - target_time_tensor (torch.Tensor): Time indices for target data
        """
        input_list = []
        clim_list = []
        # Get aggregated data from temporal aggregator
        input_aggregated, target_aggregated, clim_target_data, season_float, year_float, \
        input_time_indexes, target_time_indexes = self.aggregator.aggregate(idx)
        
        # Prepare input data tensors
        for var in self.global_variables:
            input_list.append(input_aggregated[var].values)
        for var in self.target_variables:
            clim_list.append(clim_target_data[var].values)
        
        # Convert lists to tensors with appropriate shapes
        clim_tensor = torch.tensor(np.transpose(np.array(clim_list), (1,2,3,0)))
        input_data_np = np.transpose(np.array(input_list), (1,2,3,0))
        input_tensor = torch.tensor(input_data_np)
        
        # Prepare target tensor
        target_tensor = self._prepare_target(target_aggregated)
        
        # Handle NaN values
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0)
        
        # Convert time indices to tensors
        input_time_tensor = torch.tensor(input_time_indexes)
        target_time_tensor = torch.tensor(target_time_indexes)

        return (input_tensor, target_tensor, season_float, year_float, 
                clim_tensor, input_time_tensor, target_time_tensor)


    
if __name__ == "__main__":
    # debug the dataset
    data_dirs = {'mediteranean': {'tp':"/home/egauillard/data/PR_era5_MED_1degr_19400101_20240229_new.nc",
                                  't2m':"/home/egauillard/data/T2M_era5_MED_1degr_19400101-20240229.nc"},
                 
                 'north_hemisphere': {"stream": "/home/egauillard/data/STREAM250_era5_NHExt_1degr_19400101_20240229_new.nc",
                                      "sst": "/home/egauillard/data/SST_era5_NHExt_1degr_19400101-20240229_new.nc",
                                      "msl": "/home/egauillard/data/MSLP_era5_NHExt_1degr_19400101_20240229_new.nc",
                                      },
                 'tropics': {"ttr": "/home/egauillard/data/OLR_era5_tropics_1degr_19400101_20240229.nc"}}

    wandb_config = {
    'dataset': {
        'variables_tropics': ["ttr"],
        'variables_nh': ["stream", "msl", "sst"],
        'variables_med': ['tp', "t2m"],
        'target_variables': ['tp'],
        'relevant_years': [1940,1941],
        'relevant_months': [10,11,12,1,2,3],
        'scaling_years': [1940,2005],
        'land_sea_mask': '/home/egauillard/data/ERA5_land_sea_mask_1deg.nc',
        'spatial_resolution': 1,
        'predict_sea_land': False,
        'out_spatial_resolution': 1,
        'sum_pr': True,
        "coarse_t":False,
        "coarse_s": False,
        "coarse_t_target": False,
        "coarse_s_target": False
    },
    'scaler': {
        'mode': 'standardize',
    },
    'temporal_aggregator': {
        'in_len': 3,
        'out_len': 3,
        'resolution_input': 7,
        'resolution_output': 7,
        'gap': 1,
        'lead_time_gap': 0
    }
}

    # Initialize dataset and dataloaders
    temp_aggregator_factory = TemporalAggregatorFactory(wandb_config['temporal_aggregator'])
    
    train_dataset = DatasetEra(wandb_config, data_dirs, temp_aggregator_factory)
    print("len dataset", train_dataset.__len__())

    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    sample = next(iter(dataloader))
    print(sample[0].shape)

    

    

