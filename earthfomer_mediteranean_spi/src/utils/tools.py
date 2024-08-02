import xarray as xr
import os
from typing import List
import xarray as xr
from typing import List, Dict
import numpy as np

class AreaDataset:
    def __init__(self, area: str, data: xr.Dataset, temporal_resolution: Dict[str, int], spatial_resolution: int, years: List[int], months: List[int], vars: List[str], target: str, sum_pr : bool = False, is_target: bool = False):
        self.area = area
        self.data = data
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.years = years
        self.months = months
        self.vars = vars
        self.target = target
        self.sum_pr = sum_pr
        self.is_target = is_target

        self._preprocess()

    def _preprocess(self):
        # Vérifier que nous avons les bonnes années
        self.data = self.data.sel(time=self.data.time.dt.year.isin(self.years))
        
        # Changer la résolution temporelle pour chaque variable
        self.change_temporal_resolution(self.temporal_resolution)
        
        # Changer la résolution spatiale
        self.change_spatial_resolution(self.spatial_resolution)
        
    def change_temporal_resolution(self, temporal_resolution: int):
        # Calculer le nombre de dates à supprimer au début
        dates_to_remove = temporal_resolution - 1

        # Obtenir toutes les dates
        all_dates = self.data.time.values

        # Identifier les dates à conserver
        dates_to_keep = all_dates[dates_to_remove:]

        # Créer un nouveau dataset
        new_data = xr.Dataset()

        for var in self.vars:
            if var != "time_bnds":
                if var == 'spi':
                    # Pour 'spi', ne pas changer la résolution temporelle mais enlever les NaN
                    new_data[var] = self.data[var].dropna(dim='time')
                elif np.issubdtype(self.data[var].dtype, np.number):
                    # Appliquer le rolling non centré
                    rolled = self.data[var].rolling(time=temporal_resolution, center=False).mean()
                    # Sélectionner seulement les dates à conserver
                    new_data[var] = rolled.sel(time=dates_to_keep)
                else:
                    new_data[var] = self.data[var].sel(time=dates_to_keep)

        # Sélectionner les mois pertinents
        new_data = new_data.sel(time=new_data.time.dt.month.isin(self.months))

        # Mettre à jour self.data
        self.data = new_data

        print(f"Nombre de pas de temps après traitement : {len(new_data.time)}")

    def change_spatial_resolution(self,spatial_resolution: int):
        if spatial_resolution != 1:
            regridded_data = self.data.coarsen(latitude=spatial_resolution, longitude=spatial_resolution, boundary="trim").mean()
            self.data = regridded_data
