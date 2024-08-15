
import pandas as pd
import numpy as np
from enum import Enum
import xarray as xr
import torch 
from utils.statistics import DataScaler
from utils.enums import StackType, Resolution
from utils.statistics import DataStatistics
from typing import List
from utils.tools import AreaDataset
from typing import List



MAPPING_SEASON= {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


import xarray as xr
import pandas as pd
import numpy as np

MAPPING_SEASON = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}

class HierarchicalAggregator:
    def __init__(self, datasets: list[xr.Dataset], target: xr.Dataset, in_len: list[int], out_len: int, resolution_input: list[int], resolution_output: int, gap: int = 1):
        self.name = "HierarchicalAggregator"
        self.datasets = datasets
        self.target = target
        self.in_len = in_len
        self.out_len = out_len
        self.resolution_input = resolution_input
        self.resolution_output = resolution_output
        self.gap = gap

        # Align dates across all datasets
        self._align_dates()

        # Group data by wet season
        self.wet_season_data = self._group_by_wet_season(self.datasets)
        self.wet_season_target = self._group_by_wet_season([self.target])

        # Initialize temporal index
        self._current_temporal_idx = 0
        self.current_wet_season_year = list(self.wet_season_data.groups.keys())[0]
        self._temporal_idx_maping = {}
        self.date_encoder = {}
        self.date_decoder = {}
        self._encode_all_dates()

    def _align_dates(self):
        common_dates = set(self.datasets[0].time.values)
        for ds in self.datasets[1:]:
            common_dates &= set(ds.time.values)
        common_dates &= set(self.target.time.values)
        common_dates = sorted(list(common_dates))

        self.datasets = [ds.sel(time=common_dates) for ds in self.datasets]
        self.target = self.target.sel(time=common_dates)

    def _encode_all_dates(self):
        all_dates = set(self.datasets[0].time.values) | set(self.target.time.values)
        for i, date in enumerate(sorted(all_dates)):
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            self.date_encoder[date_str] = i
            self.date_decoder[i] = date_str

    def _encode_date(self, date):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        return self.date_encoder[date_str]

    def _decode_date(self, encoded_date):
        return self.date_decoder[encoded_date]

    def _group_by_wet_season(self, datasets):
        start_month = 10
        grouped_data = []
        for data in datasets:
            data['wet_season_year'] = data.time.dt.year * (data.time.dt.month >= start_month) + (data.time.dt.year - 1) * (data.time.dt.month < start_month)
            grouped_data.append(data.groupby('wet_season_year'))
        return grouped_data

    def _compute_number_samples_in_season(self, wet_season_data: xr.DataArray):
        width_input = sum([in_len * res for in_len, res in zip(self.in_len, self.resolution_input)])
        width_output = self.out_len * self.resolution_output
        total_width = width_input + width_output
        total_length = len(wet_season_data.time.values)
        return ((total_length - total_width) // self.gap) + 1

    def compute_len_dataset(self):
        length = 0
        for wet_season in self.wet_season_data[0].groups.values():
            length += self._compute_number_samples_in_season(wet_season)
        return length

    def aggregate(self, idx: int):
        input_data = []
        target_data = []
        wet_season = [group[self.current_wet_season_year] for group in self.wet_season_data]
        wet_season_target = self.wet_season_target[0][self.current_wet_season_year]

        width_input = [in_len * res for in_len, res in zip(self.in_len, self.resolution_input)]
        width_output = self.out_len * self.resolution_output

        if self._current_temporal_idx + sum(width_input) + width_output >= len(wet_season[0].time.values):
            self.current_wet_season_year += 1
            self._current_temporal_idx = 0
            wet_season = [group[self.current_wet_season_year] for group in self.wet_season_data]
            wet_season_target = self.wet_season_target[0][self.current_wet_season_year]

        first_time_value = wet_season[0].time.dt.season.values[self._current_temporal_idx]

        # Select input data
        for ds, w_input in zip(wet_season, width_input):
            input_time_indexes = ds.time.values[self._current_temporal_idx:self._current_temporal_idx + w_input:self.resolution_input]
            input_data.append(ds.sel(time=input_time_indexes))

        # Stack input data
        input_data = xr.concat(input_data, dim='time')

        # Select target data
        # je dirai le max plutot que la sum
        start_idx_output = self._current_temporal_idx + sum(width_input)
        target_time_indexes = wet_season_target.time.values[start_idx_output:start_idx_output + width_output:self.resolution_output]
        target_data = wet_season_target.sel(time=target_time_indexes)

        # Update temporal index for the next iteration
        self._current_temporal_idx += self.gap
        self._temporal_idx_maping[idx] = self._current_temporal_idx

        # Encode temporal indexes
        encoded_input_time_indexes = [self._encode_date(date) for date in input_time_indexes]
        encoded_target_time_indexes = [self._encode_date(date) for date in target_time_indexes]

        return input_data, target_data, self.season_float, self.year_float, encoded_input_time_indexes, encoded_target_time_indexes

    def decode_time_indexes(self, encoded_indexes_array):
        return [[self._decode_date(int(idx)) for idx in encoded_indexes_array[i]] for i in range(len(encoded_indexes_array))]


class TemporalAggregatorFactory:
    def __init__(self, config):
        self.in_len = config['in_len']
        self.out_len = config['out_len']
        self.resolution_input = config['resolution_input']
        self.resolution_output = config['resolution_output']
        self.gap = config['gap']

    def create_aggregator(self, datasets: list[xr.Dataset], target: xr.Dataset) -> HierarchicalAggregator:
        return HierarchicalAggregator(
            datasets=datasets,
            target=target,
            in_len=self.in_len,
            out_len=self.out_len,
            resolution_input=self.resolution_input,
            resolution_output=self.resolution_output,
            gap=self.gap
        )

if __name__ == "__main__":
    # Example configuration and instantiation
    config = {
        'in_len': [1, 1, 1, 3],
        'out_len': 1,
        'resolution_input': [30, 14, 7, 1],
        'resolution_output': 1,
        'gap': 1
    }

    # Replace these with actual datasets
    datasets = [xr.Dataset(), xr.Dataset(), xr.Dataset(), xr.Dataset()]
    target = xr.Dataset()

    factory = TemporalAggregatorFactory(config)
    aggregator = factory.create_aggregator(datasets, target)

    # Example usage
    for idx in range(aggregator.compute_len_dataset()):
        input_data, target_data, season_float, year_float, encoded_input_time_indexes, encoded_target_time_indexes = aggregator.aggregate(idx)
        print(f"Input Data: {input_data}")
        print(f"Target Data: {target_data}")
        print(f"Season Float: {season_float}")
        print(f"Year Float: {year_float}")
        print(f"Encoded Input Time Indexes: {encoded_input_time_indexes}")
        print(f"Encoded Target Time Indexes: {encoded_target_time_indexes}")
