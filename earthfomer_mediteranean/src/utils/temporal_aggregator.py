
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

# à la place faire une rolling mean
# en gros l'idée c'est d'avoir en parametre le type de stack = [1,7,14,30]
# ca veut dire on lui donne le dernier j, la rolling mean sur les 7 d'avant, les 14 d'avant, les 30 d'avant
# donc pour créer les données aggrgés on va stacker toutes ces moyennes et s'assurer qu'il nous reste assez de de timepoints
# pour creer la target 
# ensuite cela doit preparer la target en fonction du lead time et donner les données aggrégées selon la resolution de l'ouput

# faut faire attention si on saute d'une année à l'autre toujours regarder si on a assez de données pour créer la target

MAPPING_SEASON= {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}

class TemporalAggregator:
    def __init__(self, dataset: xr.Dataset, target: xr.Dataset,  in_len : int,out_len : int, resolution_input : int, resolution_output: int, gap : int =1):
        self.name = "TemporalAggregator"
        # dataset parameters 
        self.dataset = dataset
        self.target = target

        # aggregation parameters
        self.in_len = in_len
        self.out_len = out_len
        self.resolution_input = resolution_input
        self.resolution_output = resolution_output
        self.gap = gap

        # group data by wet season
        self.wet_season_data = self._group_by_wet_season(self.dataset)
        self.wet_season_target = self._group_by_wet_season(self.target)

        # initialize temporal index
        self._current_temporal_idx = 0
        self.current_wet_season_year = list(self.wet_season_data.groups.keys())[0]
        self._temporal_idx_maping = {}
        self.date_encoder = {}
        self.date_decoder = {}
        self._encode_all_dates()

    def _encode_all_dates(self):
        all_dates = set(self.dataset.time.values) | set(self.target.time.values)
        for i, date in enumerate(sorted(all_dates)):
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            self.date_encoder[date_str] = i
            self.date_decoder[i] = date_str

    def _encode_date(self, date):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        return self.date_encoder[date_str]

    def _decode_date(self, encoded_date):
        return self.date_decoder[encoded_date]
        
    def _group_by_wet_season(self, data):
        self.start_month = 10
        data['wet_season_year']= data.time.dt.year*(data.time.dt.month>=self.start_month) + (data.time.dt.year-1)*(data.time.dt.month<self.start_month)
        grouped_data = data.groupby('wet_season_year')
        return grouped_data
    
    def _compute_number_samples_in_season(self, wet_season_data: xr.DataArray):
        width_input = self.in_len*self.resolution_input
        width_output = self.out_len*self.resolution_output
        total_width = width_input + width_output
        total_lenght = len(wet_season_data.time.values)
        return ((total_lenght - total_width)//self.gap) +1
    
    def compute_len_dataset(self):
        length = 0
        for _, wet_season in self.wet_season_data:
            length += self._compute_number_samples_in_season(wet_season)
        return length

    def aggregate(self, idx: int):
        input_data = []
        target_data = []
        wet_season = self.wet_season_data[self.current_wet_season_year]
        wet_season_target = self.wet_season_target[self.current_wet_season_year]

        width_input = self.in_len * self.resolution_input
        width_output = self.out_len * self.resolution_output

        if self._current_temporal_idx + width_input + width_output >= len(wet_season.time.values):
            self.current_wet_season_year += 1
            self._current_temporal_idx = 0
            wet_season = self.wet_season_data[self.current_wet_season_year]
            wet_season_target = self.wet_season_target[self.current_wet_season_year]

        first_time_value = wet_season.time.dt.season.values[self._current_temporal_idx]
        self.season_float = MAPPING_SEASON[str(first_time_value)]/4

        first_year = 1940
        last_year = 2024
        self.year_float = (self.current_wet_season_year - first_year) / (last_year - first_year)

        # Sélectionner les données d'entrée
        input_time_indexes = wet_season.time.values[self._current_temporal_idx:self._current_temporal_idx + width_input:self.resolution_input]
        input_data = wet_season.sel(time=input_time_indexes)

        # Sélectionner les données de sortie
        start_idx_output = self._current_temporal_idx + width_input
        target_time_indexes = wet_season_target.time.values[start_idx_output:start_idx_output + width_output:self.resolution_output]
        target_data = wet_season_target.sel(time=target_time_indexes)

        # Mettre à jour l'index temporel pour la prochaine itération
        self._current_temporal_idx += self.gap
        self._temporal_idx_maping[idx] = self._current_temporal_idx

        # Encoder les indices temporels
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

    def create_aggregator(self, dataset : xr.Dataset, target: xr.Dataset) -> TemporalAggregator:
        
        return TemporalAggregator(
            dataset=dataset,
            target = target,
            in_len=self.in_len,
            out_len=self.out_len,
            resolution_input=self.resolution_input,
            resolution_output=self.resolution_output,
            gap = self.gap
        )

if __name__ == "__main__":
    pass