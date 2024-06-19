
import pandas as pd
import numpy as np
from enum import Enum
import xarray as xr
import torch 
from utils.scaler import DataScaler
from utils.enums import StackType, Resolution

# à la place faire une rolling mean
# en gros l'idée c'est d'avoir en parametre le type de stack = [1,7,14,30]
# ca veut dire on lui donne le dernier j, la rolling mean sur les 7 d'avant, les 14 d'avant, les 30 d'avant
# donc pour créer les données aggrgés on va stacker toutes ces moyennes et s'assurer qu'il nous reste assez de de timepoints
# pour creer la target 
# ensuite cela doit preparer la target en fonction du lead time et donner les données aggrégées selon la resolution de l'ouput

# faut faire attention si on saute d'une année à l'autre toujours regarder si on a assez de données pour créer la target

MAPPING_SEASON= {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}

    
class TemporalAggregator:
    # Also the target need to be coarsen after scaling we need to prepare that

    def __init__(self, dataset,area, stack_number_input : int,lead_time_number : int, resolution_input : int, resolution_output: int, scaler: DataScaler):
        self.name = "TemporalAggregator"
        self.area = area
        # aggregation parameters
        self.stack_number_input = stack_number_input
        self.lead_time_number = lead_time_number
        self.resolution_input = resolution_input
        self.resolution_output = resolution_output
        # load data and scaler
        self.dataset = dataset
        self.scaler = scaler
        self.statistics = self.dataset.get_statistics(Resolution.WEEKLY)[area]
        
        # group data by wet season
        self.wet_season_data = self._group_by_wet_season(self.dataset)


        # initialize temporal index
        self._current_temporal_idx = 0
        self.current_wet_season_year = self.wet_season_data.groups.keys()[0]
        self._temporal_idx_maping = {}
    
    def _group_by_wet_season(self, data):
        self.start_month = min(self.relevant_months)
        data['wet_season_year']= data.time.dt.year*(data.time.dt.month>=self.start_month) + (data.time.dt.year-1)*(data.time.dt.month<self.start_month)
        grouped_data = data.groupby('wet_season_year')
        return grouped_data
    
    def _compute_number_samples_in_season(self, wet_season_data: xr.DataArray):
        width_input = self.stack_number_input*self.resolution_input
        width_output = self.lead_time_number*self.resolution_output
        return len(wet_season_data.time.values)//(width_input + width_output)
    
    def compute_len_dataset(self):
        length = 0
        for _, wet_season_data in self.wet_season_data:
            length += self._compute_number_samples_in_season(wet_season_data)
        return length
    
    def scale(self, data: xr.DataArray):
        week_idx = data.time.dt.week.values[0]
        week_stats = {key: value.where(value.time.dt.week == week_idx) for key, value in self.statistics.items() }
        scaled_output = self.scaler.scale(data, week_stats)
        return scaled_output
    
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
        
        self.season_float = MAPPING_SEASON[str(wet_season[self._current_temporal_idx].time.dt.season.values)]
        self.year_float = self.current_wet_season_year - self.dataset.first_year/ (self.dataset.last_year - self.dataset.first_year)
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
            scaled_input = self.scale(mean_input)
            input_data.append(scaled_input)
        for i in range(0,self.lead_time_number):
            mean_output = output_window.sel(time = target_time_indexes[i*self.resolution_output:(i+1)*self.resolution_output]).mean(dim = "time")
            scaled_output = self.scale(mean_output)
            target_data.append(scaled_output)

        input_data = np.stack(input_data, axis = 0)
        target_data = np.stack(target_data, axis = 0)

        # update temporal index for next iteration
        self._current_temporal_idx = start_idx_output + self.resolution_output*self.lead_time_number +1
        self._temporal_idx_maping[idx] = self._current_temporal_idx

        return torch.Tensor(input_data), torch.Tensor(target_data), torch.Tensor([self.year_float]), torch.Tensor([self.season_float])
    


class TemporalAggregatorFactory:
    def __init__(self, stack_number_input: int, lead_time_number: int, resolution_input: int, resolution_output: int, scaler: DataScaler):
        self.stack_number_input = stack_number_input
        self.lead_time_number = lead_time_number
        self.resolution_input = resolution_input
        self.resolution_output = resolution_output
        self.scaler = scaler

    def create_aggregator(self, dataset: xr.Dataset, area: str) -> TemporalAggregator:
        return TemporalAggregator(
            dataset=dataset,
            area=area,
            stack_number_input=self.stack_number_input,
            lead_time_number=self.lead_time_number,
            resolution_input=self.resolution_input,
            resolution_output=self.resolution_output,
            scaler=self.scaler
        )



# class HierarchicalAggregator:
#     def __init__(self,stack_type_input: list[StackType], lead_time_output: int, resolution_output: int, scaler : callable):
#         self.name = "HierarchicalAggregator"
#         self.stack_type_input = stack_type_input
#         self.lead_time_output = lead_time_output
#         self.resolution_output = resolution_output
#         self._temporal_idx = 0
#         self.scaler = scaler
#         self._temporal_idx_maping = {}
#         self._current_temporal_idx = 0
    
#     def compute_len_dataset(self, data: xr.DataArray):
#         width_input = max([stack.value for stack in self.stack_type_input])
#         return len(data.time.values) - width_input - self.lead_time_output - self.resolution_output
    
#     def aggregate(self, data: xr.DataArray, idx: int):
#         input_data = []
#         target_data = []

#         # compute the window of temporal indexes we will use to create the input data
#         width_input = max([stack.value for stack in self.stack_type_input])
#         input_time_indexes = data.time.values[self._current_temporal_idx:self._current_temporal_idx + width_input]
#         input_window = data.sel(time=input_time_indexes)

#         # compute the indexes of interest for the target data
#         start_idx_output = self._current_temporal_idx + width_input + 1
#         target_time_indexes = data.time.values[start_idx_output : start_idx_output + self.lead_time_output, self.resolution_output]
        
#         # compute the aggregated data for the input and stack them
#         rolling_mean = {stack.value : input_window.rolling(time = stack.value, center = False ).mean() for stack in self.stack_type_input}
#         input_data = [rolling_mean[stack.value].values for stack in self.stack_type_input]
#         input_data = np.stack(input_data, axis = 0)

#         # prepare target data 
#         target_seq = data.sel(time = target_time_indexes).values
#         target_data.append(target_seq)

#         # update temporal index for next iteration
#         self._current_temporal_idx = start_idx_output + self.lead_time_output +1
#         self._temporal_idx_maping[idx] = self._current_temporal_idx

#         return torch.Tensor(input_data), torch.Tensor(target_data)

if __name__ == "__main__":
    stack_type_input = [StackType.DAILY, StackType.WEEKLY, StackType.BIWEEKLY, StackType.MONTHLY]