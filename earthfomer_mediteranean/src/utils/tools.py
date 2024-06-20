import xarray as xr
import os
from typing import List

class AreaDataset:
    def __init__(self, area: str, data: xr.Dataset, spatial_resolution: int, years : List[int], months: List[int], vars : List[str], target : str):
        self.area = area
        self.data = data
        self.spatial_resolution = spatial_resolution
        self.years = years
        self.months = months
        self.vars = vars
        self.target = target