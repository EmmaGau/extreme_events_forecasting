import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from matplotlib import font_manager

# Configuration de la police
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Charger les datasets et préparer les données
normal_input = xr.open_dataset('/home/egauillard/extreme_events_forecasting/notebooks/test/input_unscaled.nc')
coarse_input = xr.open_dataset('/home/egauillard/extreme_events_forecasting/notebooks/test/coarse_dataset.nc')
fine_input = xr.open_dataset('/home/egauillard/extreme_events_forecasting/notebooks/test/fine_dataset.nc')
coarse_t_fine_s = xr.open_dataset('/home/egauillard/extreme_events_forecasting/notebooks/test/coarse_t_fine_s_dataset.nc')

normal_input["tp"] = 1000 * normal_input["tp"]
datasets = [normal_input, coarse_input, coarse_t_fine_s, fine_input]
dataset_labels = ["Original Data", "Coarse Scaling", "Spatial Fine Scaling", "Spatial & Temporal Fine Scaling"]

variables = ['stream', 'msl', 'sst', 'tp', 't2m', 'ttr']

# Configuration du plot
n_datasets = len(datasets)
n_vars = len(variables)

# Augmenter la taille de la figure proportionnellement
fig = plt.figure(figsize=(8*n_datasets, 5*n_vars))

# Define the extents and units for each variable
var_info = {
    "t2m": {"extent": [-10, 39, 31, 45], "units": "°K", "region": "Mediterranean", "full_name": "2m temperature"},
    "tp": {"extent": [-10, 39, 31, 45], "units": "mm", "region": "Mediterranean", "full_name": "Total precipitation"},
    "ttr": {"extent": [-180, 180, -20, 30], "units": "W/m²", "region": "Tropics", "full_name": "Top thermal radiation"},
    "stream": {"extent": [-180, 180, -20, 90], "units": "m²/s", "region": "Northern Hemisphere +", "full_name": "Stream function"},
    "sst": {"extent": [-180, 180, -20, 90], "units": "°K", "region": "Northern Hemisphere +", "full_name": "Sea surface temperature"},
    "msl": {"extent": [-180, 180, -20, 90], "units": "Pa", "region": "Northern Hemisphere +", "full_name": "Mean sea level pressure"},
}

cmap = plt.cm.RdBu_r
# Create grid for subplots with minimal spacing
grid = fig.add_gridspec(n_vars, n_datasets, hspace=0.03, wspace=0.05)

for row, var in enumerate(variables):
    for col, (ds, label) in enumerate(zip(datasets, dataset_labels)):
        ax = fig.add_subplot(grid[row, col], projection=ccrs.PlateCarree())
        
        data = ds[var].isel(time=0)
        
        if label == "Original Data":
            data = data.sel(latitude=slice(var_info[var]["extent"][2], var_info[var]["extent"][3]), 
                            longitude=slice(var_info[var]["extent"][0], var_info[var]["extent"][1]))
            vmin, vmax = data.min(), data.max()
        else:
            max_abs_val = max(abs(data.min()), abs(data.max()))
            vmin, vmax = -max_abs_val, max_abs_val
        
        im = ax.pcolormesh(data.longitude, data.latitude, data, 
                           transform=ccrs.PlateCarree(), 
                           cmap=cmap,
                           vmin=vmin, vmax=vmax)
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        
        # Ajouter les lignes de latitude et longitude avec alpha réduit
        ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, alpha=0.3)
        
        ax.set_extent(var_info[var]["extent"], crs=ccrs.PlateCarree())
        
        # Adjust colorbar settings for smaller size but larger font
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=20, shrink=0.8)
        if label == "Original Data":
            cbar.set_label(f'{var_info[var]["full_name"]} ({var_info[var]["units"]})', fontsize=20)
        else:
            cbar.set_label(f'{var_info[var]["full_name"]} standardized', fontsize=20)
        cbar.ax.tick_params(labelsize=10)
        
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        if row == 0:
            ax.set_title(label, fontsize=24, pad=30, fontweight='bold')
        if col == 0:
            ax.text(-0.1, 0.15, f'{var_info[var]["full_name"]}\n{var_info[var]["region"]}', va='center', ha='right', rotation='vertical', 
                    transform=ax.transAxes, fontsize=19)

plt.savefig('input_data_visualization_regions_improved.png', dpi=300, bbox_inches='tight')
plt.close()