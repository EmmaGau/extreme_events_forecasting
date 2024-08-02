


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