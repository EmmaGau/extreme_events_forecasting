import xclim 
import xarray as xr
from xclim.core import units

def spi_transformation(pr_path : str, cal_start: str, cal_end: str,  window: int = 14):
    pr_data = xr.open_dataset(pr_path)["tp"]
    pr_data['time'] = pr_data.time.dt.floor('D')
    print(pr_data.attrs['units'])
    pr_data.attrs['units'] = 'm/d'

    # sort the time index
    pr_data = pr_data.sortby('time')

    # mettre tous les jours 
    spi_data = xclim.indices.standardized_precipitation_index(pr_data, freq='D', window= window, dist='gamma', method='ML', cal_start=cal_start, cal_end=cal_end)
    new_path = pr_path.replace("PR", f"SPI_{window}")
    # save 
    spi_data.to_netcdf(new_path)
    return pr_data

if __name__ == '__main__':
    path = '/home/egauillard/data/PR_era5_MED_1degr_19400101_20240229_new.nc'
    spi_transformation(path, '1940-01-01', '1990-12-31', 14)