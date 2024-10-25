
import pandas as pd
import numpy as np
from enum import Enum
import xarray as xr
import torch 
from utils.enums import StackType, Resolution
from typing import List

# à la place faire une rolling mean
# en gros l'idée c'est d'avoir en parametre le type de stack = [1,7,1:4,30]
# ca veut dire on lui donne le dernier j, la rolling mean sur les 7 d'avant, les 14 d'avant, les 30 d'avant
# donc pour créer les données aggrgés on va stacker toutes ces moyennes et s'assurer qu'il nous reste assez de de timepoints
# pour creer la target 
# ensuite cela doit preparer la target en fonction du lead time et donner les données aggrégées selon la resolution de l'ouput

# faut faire attention si on saute d'une année à l'autre toujours regarder si on a assez de données pour créer la target

MAPPING_SEASON= {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}

class TemporalAggregator:
    def __init__(self, dataset: xr.Dataset, target: xr.Dataset, clim: xr.Dataset, in_len: int, out_len: int, resolution_input: int, resolution_output: int, gap: int = 1, lead_time_gap: int = 0):
        self.name = "TemporalAggregator"
        # dataset parameters 
        self.dataset = dataset
        self.target = target
        self.clim = clim

        # aggregation parameters
        self.in_len = in_len
        self.out_len = out_len
        self.resolution_input = resolution_input
        self.resolution_output = resolution_output
        self.gap = gap
        self.lead_time_gap = lead_time_gap

        # group data by wet season
        self.wet_season_data = self._group_by_wet_season(self.dataset)
        self.wet_season_target = self._group_by_wet_season(self.target)

        # Print statements to check grouping
        print(f"Wet seasons in dataset: {list(self.wet_season_data.groups.keys())}")
        print(f"Wet seasons in target: {list(self.wet_season_target.groups.keys())}")

        self._index_mapping = self._create_index_mapping()
        print(f"Index mapping created with {len(self._index_mapping)} entries.")

        self.date_encoder = {}
        self.date_decoder = {}
        self._encode_all_dates()

    def _encode_all_dates(self):
        all_dates = set(self.dataset.time.values) | set(self.target.time.values)
        print("All dates combined:", sorted(all_dates))
        for i, date in enumerate(sorted(all_dates)):
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            self.date_encoder[date_str] = i
            self.date_decoder[i] = date_str
        print("Date encoder:", self.date_encoder)
        print("Date decoder:", self.date_decoder)

    def _encode_date(self, date):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        encoded_date = self.date_encoder[date_str]
        print(f"Encoded date {date_str} as {encoded_date}")
        return encoded_date

    def _decode_date(self, encoded_date):
        decoded_date = self.date_decoder[encoded_date]
        print(f"Decoded date {encoded_date} as {decoded_date}")
        return decoded_date
        
    def _group_by_wet_season(self, data):
        self.start_month = 9
        data['wet_season_year'] = data.time.dt.year * (data.time.dt.month >= self.start_month) + (data.time.dt.year - 1) * (data.time.dt.month < self.start_month)
        grouped_data = data.groupby('wet_season_year')
        return grouped_data

    def _get_clim_data(self, time_indexes):
        days_of_year = [pd.Timestamp(date).dayofyear for date in time_indexes]
        print(f"Days of year for climate data: {days_of_year}")
        clim_data = self.clim.sel(dayofyear=days_of_year)
        print(f"Retrieved climate data for {len(days_of_year)} days.")
        return clim_data

    def _create_index_mapping(self):
        index_mapping = {}
        total_samples = 0
        
        for year, wet_season in self.wet_season_data:
            width_input = self.in_len * self.resolution_input
            width_output = self.out_len * self.resolution_output
            total_width = width_input + self.lead_time_gap + width_output
            season_length = len(wet_season.time.values)

            print(f"Processing wet season year {year} with {season_length} days.")
            print(f"Total width required: {total_width} (input: {width_input}, lead_time_gap: {self.lead_time_gap}, output: {width_output})")
            
            if season_length >= total_width:
                num_samples = ((season_length - total_width) // self.gap) + 1
                print(f"  -> Generating {num_samples} samples for this season.")
                
                for local_idx in range(num_samples):
                    index_mapping[total_samples + local_idx] = {
                        'wet_season_year': year,
                        'local_idx': local_idx * self.gap
                    }

                total_samples += num_samples
            else:
                print(f"  -> Not enough data for this season. Required: {total_width}, Available: {season_length}.")
        
        print(f"Total samples across all seasons: {total_samples}")
        return index_mapping

    def aggregate(self, idx: int):
        # Récupérer la saison humide et l'indice local à partir du mapping
        mapping = self._index_mapping[idx]
        year = mapping['wet_season_year']
        local_idx = mapping['local_idx']
        print(f"\nAggregating for idx {idx}: year {year}, local_idx {local_idx}")

        wet_season = self.wet_season_data[year]
        wet_season_target = self.wet_season_target[year]

        width_input = self.in_len * self.resolution_input
        width_output = self.out_len * self.resolution_output

        # Récupérer les données d'entrée et de sortie
        input_time_indexes = wet_season.time.values[local_idx:local_idx + width_input:self.resolution_input]
        target_start_idx = local_idx + width_input + self.lead_time_gap
        target_time_indexes = wet_season_target.time.values[target_start_idx:target_start_idx + width_output:self.resolution_output]
        print(f"  -> Input time indexes: {input_time_indexes}")
        print(f"  -> Target time indexes: {target_time_indexes}")
        print(f"  -> Lead time gap: {self.lead_time_gap} days")

         # Vérifier que le lead_time_gap est correctement appliqué
        expected_gap = self.lead_time_gap + self.resolution_input
        actual_gap = (pd.Timestamp(target_time_indexes[0]) - pd.Timestamp(input_time_indexes[-1])).days
        print(f"  -> Actual gap between input and target: {actual_gap} days")
        print(f"  -> Expected gap: {expected_gap} days (lead_time_gap + resolution_input)")
        
        if actual_gap != expected_gap:
            print(f"  -> Warning: Actual gap ({actual_gap}) differs from expected gap ({expected_gap})")
        else:
            print(f"  -> Gap verification successful: Actual gap matches expected gap")

        input_data = wet_season.sel(time=input_time_indexes)
        target_data = wet_season_target.sel(time=target_time_indexes)
        clim_input_data = self._get_clim_data(input_time_indexes)
        clim_target_data = self._get_clim_data(target_time_indexes)

        # Calculer les floats pour la saison et l'année
        first_time_value = wet_season.time.dt.season.values[local_idx]
        season_float = MAPPING_SEASON[str(first_time_value)] / 4
        first_year = 1940
        last_year = 2024
        year_float = (year - first_year) / (last_year - first_year)
        print(f"  -> Season float: {season_float}, Year float: {year_float}")

        # Encoder les indices temporels
        encoded_input_time_indexes = [self._encode_date(date) for date in input_time_indexes]
        encoded_target_time_indexes = [self._encode_date(date) for date in target_time_indexes]

        return input_data, target_data, clim_target_data, season_float, year_float, encoded_input_time_indexes, encoded_target_time_indexes

    def compute_len_dataset(self):
        dataset_length = len(self._index_mapping)
        print(f"Computed dataset length: {dataset_length}")
        return dataset_length
    
    def decode_time_indexes(self, encoded_indexes_array):
        decoded_indexes = [[self._decode_date(int(idx)) for idx in encoded_indexes_array[i]] for i in range(len(encoded_indexes_array))]
        print(f"Decoded time indexes: {decoded_indexes}")
        return decoded_indexes

class TemporalAggregatorFactory:
    def __init__(self, config):
        self.in_len = config['in_len']
        self.out_len = config['out_len']
        self.resolution_input = config['resolution_input']
        self.resolution_output = config['resolution_output']
        self.gap = config['gap']
        self.lead_time_gap = config.get('lead_time_gap', 0)
        print(f"Initialized TemporalAggregatorFactory with lead_time_gap: {self.lead_time_gap}")

    def create_aggregator(self, dataset : xr.Dataset, target: xr.Dataset, clim: xr.Dataset):
        
        return TemporalAggregator(
            dataset=dataset,
            target = target,
            clim=clim,
            in_len=self.in_len,
            out_len=self.out_len,
            resolution_input=self.resolution_input,
            resolution_output=self.resolution_output,
            gap = self.gap,
            lead_time_gap = self.lead_time_gap
        )

if __name__ == "__main__":
    # Example usage
    config = {
        'in_len': 6,
        'out_len': 3,
        'resolution_input': 7,
        'resolution_output': 7,
        'gap': 1,
        'lead_time_gap': 14
    }
    
    # Create dummy datasets for testing
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
    dataset = xr.Dataset({'var': ('time', np.random.rand(len(dates)))}, coords={'time': dates})
    target = xr.Dataset({'var': ('time', np.random.rand(len(dates)))}, coords={'time': dates})
    clim = xr.Dataset({'var': ('dayofyear', np.random.rand(366))}, coords={'dayofyear': range(1, 367)})

    factory = TemporalAggregatorFactory(config)
    aggregator = factory.create_aggregator(dataset, target, clim)
    
    # Test the aggregator
    for i in range(5):
        print(f"\nTesting aggregation for index {i}")
        aggregator.aggregate(i)

    print("\nComputing dataset length:")
    aggregator.compute_len_dataset()