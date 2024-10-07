from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np 
import torch 
from utils.temporal_aggregator import TemporalAggregatorFactory
from data.area_dataset import AreaDataset
from utils.enums import StackType, Resolution
import wandb 
import xclim

class DatasetEra(Dataset):
    def __init__(
        self,
        wandb_config : dict,
        data_dirs : str,
        temporal_aggr_factory : TemporalAggregatorFactory):
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
        self.global_variables = self.variables_med.copy()  # Commencez avec les variables méditerranéennes

        if self.variables_nh is not None:
            self.global_variables.extend(self.variables_nh)
        
        if hasattr(self, 'variables_tropics') and self.variables_tropics is not None:
            self.global_variables.extend(self.variables_tropics)
        
        self.resolution_input = self.aggregator_factory.resolution_input
        self.resolution_output = self.aggregator_factory.resolution_output
        self.data, self.target = self._load_and_prepare_data()
        self.scaled_clim = self.get_scaled_climatology()

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
        
        if "coarse" in wandb_config["scaler"].keys():
            self.coarse_t = wandb_config["scaler"]["coarse"]
            self.coarse_t_target = wandb_config["scaler"]["coarse"]
            print(f"Applying {'coarse' if self.coarse_t else 'fine-grained'} temporal scaling for both input and target data.")
            
    
    def _load_data(self, dir_path):
        """Load data from a specified directory using xarray."""
        ds = xr.open_dataset(dir_path)
        return ds
            
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

    def get_scaled_climatology(self):
        return self.target_class.scaled_climatology
        
    def _load_and_prepare_data(self):
        """Load data from directories for all variables and create big dataset that contains all variables for both regions
            and keep the relevant years/months."""
        med_datasets = []
        nh_datasets = []
        tropics_datasets = []

        # first check that dataset are aligned temporaly and spatially, and merge datasets
        for var in self.variables_med:
            med_data = self._load_data(self.data_dirs['mediteranean'][var])
            if med_datasets:
                med_datasets[0], med_data = self.check_dataset(med_datasets[0], med_data, var)
                # si y a des nan, print something et replace theme by 0 
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

        # Créer les classes AreaDataset
        self.med_class = self._create_area_dataset("mediteranean", med_data, self.variables_med)
        self.nh_class = self._create_area_dataset("north_hemisphere", nh_data, self.variables_nh) if nh_data is not None else None
        self.tropics_class = self._create_area_dataset("tropics", tropics_data, self.variables_tropics) if tropics_data is not None else None
        self.target_class = self._create_area_dataset("target", med_data[self.target_variables], self.target_variables, is_target=True)

        target = self.target_class.scaled_data 
        med_data = self.med_class.scaled_data
        nh_data = self.nh_class.scaled_data if nh_data is not None else None
        tropics_data = self.tropics_class.scaled_data if tropics_data is not None else None

        # Remapping et fusion des données
        data = self._remap_and_merge_data(med_data, nh_data, tropics_data)

        return data, target

    def _create_area_dataset(self, area, data, variables, is_target=False):
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
        # Trouver les temps communs à tous les jeux de données
        common_times = med_data.time
        if nh_data is not None:
            common_times = np.intersect1d(common_times, nh_data.time)
        if tropics_data is not None:
            common_times = np.intersect1d(common_times, tropics_data.time)

        # Sélectionner seulement les temps communs pour chaque jeu de données
        med_data = med_data.sel(time=common_times)
        if nh_data is not None:
            nh_data = nh_data.sel(time=common_times)
        if tropics_data is not None:
            tropics_data = tropics_data.sel(time=common_times)

        data_list = [nh_data]  if nh_data is not None else [med_data]

        if nh_data is not None:
            remapped_med = self.remap_to_NH(nh_data, med_data)
            data_list.append(remapped_med)
            print("Mediterranean data remapped to NH")
        
        if tropics_data is not None:
            remapped_tropics = self.remap_to_NH(nh_data, tropics_data)
            data_list.append(remapped_tropics)
            print("Tropical data remapped to NH")
        
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
        len = self.aggregator.compute_len_dataset()
        return len
    
    def get_binary_sea_mask(self):
        mask = xr.open_dataset(self.mask_path, chunks = None)
        threshold = 0.3
        mask["lsm"] = xr.apply_ufunc(
            lambda x: (x > threshold).astype(int),
            mask["lsm"],
            dask="allowed",  # Si vous utilisez dask, sinon retirez cet argument
            keep_attrs=True  # Conserver les attributs
        )
        return mask["lsm"]

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
        
        return target_tensor
    
    def _prepare_target(self, target_data):
        if self.predict_sea_land:
            target_tensor = self._prepare_sea_land_target(target_data)
        else: 
            target_list = []
            for var in self.target_variables:
                target_list.append(target_data[var].values)
            
            target_data_np = np.transpose(np.array(target_list), (1,2,3,0))
            target_tensor = torch.tensor(target_data_np)  # size (batch_size, height, width, channels)
        return target_tensor
        
    def __getitem__(self, idx):
        input_list = []
        clim_list = []
        # Aggregate the input data
        input_aggregated, target_aggregated, clim_target_data, season_float, year_float , input_time_indexes, target_time_indexes = self.aggregator.aggregate(idx)
       # input data preparation
        for var in self.global_variables:
            input_list.append(input_aggregated[var].values)
        for var in self.target_variables:
            clim_list.append(clim_target_data[var].values)
        
        clim_tensor = torch.tensor(np.transpose(np.array(clim_list), (1,2,3,0)))
            
        input_data_np = np.transpose(np.array(input_list), (1,2,3,0))
        input_tensor = torch.tensor(input_data_np)  # size (batch_size, height, width, channels)

        # target preparation
        target_tensor = self._prepare_target(target_aggregated) # size (batch_size, height, width, channels)
        # replace nan values by 0
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0)
        input_time_tensor = torch.tensor(input_time_indexes)
        target_time_tensor = torch.tensor(target_time_indexes)

        return input_tensor, target_tensor, season_float, year_float, clim_tensor, input_time_tensor, target_time_tensor


    
if __name__ == "__main__":
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
        'out_spatial_resolution': 10,
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

    train_dataset.compute_climatology()

    

    

