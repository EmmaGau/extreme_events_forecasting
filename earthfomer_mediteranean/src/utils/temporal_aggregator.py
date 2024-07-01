
import pandas as pd
import numpy as np
from enum import Enum
import xarray as xr
import torch 
from utils.scaler import DataScaler
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
    def __init__(self, dataset: AreaDataset, stack_number_input : int,lead_time_number : int, resolution_input : int, resolution_output: int, scaler: DataScaler, scaling_years: List[int], scaling_months: List[int], gap : int =0):
        self.name = "TemporalAggregator"
        # dataset parameters 
        self.dataset = dataset

        # aggregation parameters
        self.stack_number_input = stack_number_input
        self.lead_time_number = lead_time_number
        self.resolution_input = resolution_input
        self.resolution_output = resolution_output
        self.gap = gap

        # load data and scaler
        self.scaler = scaler
        self.scaling_resolution = self.get_scaling_resolution()
        self.stat_computer = DataStatistics(scaling_years, scaling_months, self.scaling_resolution)
        self.statistics =  self.stat_computer._get_stats(self.dataset)
        
        # group data by wet season
        self.wet_season_data = self._group_by_wet_season(self.dataset.data)

        # initialize temporal index
        self._current_temporal_idx = 0
        self.current_wet_season_year = list(self.wet_season_data.groups.keys())[0]
        self._temporal_idx_maping = {}

    def get_scaling_resolution(self):
        if 1<= self.resolution_input <= 5:
            return Resolution.DAILY
        elif 5< self.resolution_input <= 14:
            return Resolution.WEEKLY
        elif 14< self.resolution_input <= 60:
            return Resolution.MONTHLY
        elif 60< self.resolution_input <= 90:
            return Resolution.SEASON
        else:
            print("Resolution not supported")
    
    def _group_by_wet_season(self, data):
        self.start_month = min(self.dataset.months)
        data['wet_season_year']= data.time.dt.year*(data.time.dt.month>=self.start_month) + (data.time.dt.year-1)*(data.time.dt.month<self.start_month)
        grouped_data = data.groupby('wet_season_year')
        return grouped_data
    
    def _compute_number_samples_in_season(self, wet_season_data: xr.DataArray):
        width_input = self.stack_number_input*self.resolution_input
        width_output = self.lead_time_number*self.resolution_output
        return len(wet_season_data.time.values)//(width_input + width_output)
    
    def compute_len_dataset(self):
        length = 0
        for _, wet_season in self.wet_season_data:
            length += self._compute_number_samples_in_season(wet_season)
        return length
    
    def find_resolution_idx (self,resolution: Resolution, data : xr.DataArray, idx: int):
        if resolution.value == "day":
            return data.time.dt.day.values[idx]
        elif resolution.value == "week":
            return data.time.dt.week.values[idx]
        elif resolution.value == "month":
            return data.time.dt.month.values[idx]
        elif resolution.value == "season":
            return data.time.dt.season.values[idx]
    
    def scale(self, wet_season: xr.DataArray,data: xr.DataArray, idx : int):
        self.stat_idx = self.find_resolution_idx(self.scaling_resolution, wet_season, idx)
        related_stats = {key: value.sel(**{f"{self.scaling_resolution.value}": self.stat_idx}, method="nearest") for key, value in self.statistics.items()}
        return self.scaler.scale(data, related_stats)

    def aggregate(self, idx: int):
        input_data = []
        target_data = []
        wet_season = self.wet_season_data[self.current_wet_season_year]

        width_input = self.stack_number_input*self.resolution_input
        width_output = self.lead_time_number*self.resolution_output
        
        if self._current_temporal_idx + width_input + width_output >= len(wet_season.time.values):
            self.current_wet_season_year +=1
            self._current_temporal_idx = 0
            wet_season = self.wet_season_data[self.current_wet_season_year]
        
        first_time_value = wet_season.time.dt.season.values[self._current_temporal_idx]
        self.season_float = MAPPING_SEASON[str(first_time_value)]

        first_year = list(self.wet_season_data.groups.keys())[0]
        last_year = list(self.wet_season_data.groups.keys())[-1]
        self.year_float = self.current_wet_season_year - first_year/ (last_year - first_year)
        # compute the window of temporal indexes we will use to create the input data

        width_input = self.stack_number_input*self.resolution_input
        input_time_indexes = wet_season.time.values[self._current_temporal_idx:self._current_temporal_idx + width_input -1]
        input_window = wet_season.sel(time=input_time_indexes)

        # compute the indexes of interest for the target wet_season
        start_idx_output = self._current_temporal_idx + width_input 
        target_time_indexes = wet_season.time.values[start_idx_output : start_idx_output + self.resolution_output*self.lead_time_number]
        output_window = wet_season.sel(time = target_time_indexes)

        # stack mean for 
        for i in range(0,self.stack_number_input):
            mean_input = input_window.sel(time = input_time_indexes[i*self.resolution_input:(i+1)*self.resolution_input]).mean(dim = "time") # data array 1 value per cell of the grid
            scaled_input = self.scale(wet_season,mean_input, self._current_temporal_idx + i*self.resolution_input)
            input_data.append(scaled_input)
        for i in range(0,self.lead_time_number):
            mean_output = output_window.sel(time = target_time_indexes[i*self.resolution_output:(i+1)*self.resolution_output]).mean(dim = "time")
            scaled_output = self.scale(wet_season,mean_output, start_idx_output + i*self.resolution_output)
            target_data.append(scaled_output)

        input_data = xr.concat(input_data, dim = "time")
        target_data = xr.concat(target_data, dim = "time")

        # update temporal index for next iteration
        # self._current_temporal_idx = start_idx_output + self.resolution_output*self.lead_time_number +1
        self._current_temporal_idx += self.gap
        self._temporal_idx_maping[idx] = self._current_temporal_idx

        return input_data, target_data, self.season_float, self.year_float
    
class TemporalAggregatorFactory:
    def __init__(self, config, scaler):
        self.stack_number_input = config['stack_number_input']
        self.lead_time_number = config['lead_time_number']
        self.resolution_input = config['resolution_input']
        self.resolution_output = config['resolution_output']
        self.scaling_years = config['scaling_years']
        self.scaling_months = config['scaling_months']
        self.scaler = scaler

    def create_aggregator(self, data_class : AreaDataset) -> TemporalAggregator:
        
        return TemporalAggregator(
            dataset = data_class,
            stack_number_input=self.stack_number_input,
            lead_time_number=self.lead_time_number,
            resolution_input=self.resolution_input,
            resolution_output=self.resolution_output,
            scaling_years=self.scaling_years,
            scaling_months=self.scaling_months,
            scaler=self.scaler
        )

if __name__ == "__main__":
    pass