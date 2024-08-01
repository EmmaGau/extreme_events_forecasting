import xclim 
import xarray as xr
from xclim.core import units
import argparse 
import numpy as np

def spi_transformation(pr_path : str, cal_start_y: str, cal_end_y: str,  window: int = 14):
    cal_start = cal_start_y + "-01-01"
    cal_end = cal_end_y + "-12-31"
    pr_data = xr.open_dataset(pr_path)["tp"]
    pr_data['time'] = pr_data.time.dt.floor('D')
    print(pr_data.attrs['units'])
    pr_data.attrs['units'] = 'm/d'

    # sort the time index
    pr_data = pr_data.sortby('time')

    # mettre tous les jours 
    spi_data = xclim.indices.standardized_precipitation_index(pr_data, freq='D', window= window, dist='gamma', method='ML', cal_start=cal_start, cal_end=cal_end)
    print(spi_data)
    new_path = pr_path.replace("PR", f"SPI_{window}_cal_{cal_start_y}_{cal_end_y}")
    # save 
    spi_data.to_netcdf(new_path)
    return pr_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SPI transformation')
    parser.add_argument('--start_year', type=str, help='start year for the calibration period')
    parser.add_argument('--end_year', type=str, help='end year for the calibration')
    parser.add_argument('--window', type=int, help='window for the SPI calculation')
    args = parser.parse_args()

    window = args.window

    path = '/home/egauillard/data/PR_era5_MED_1degr_19400101_20240229_new.nc'
    spi_transformation(path, args.start_year, args.end_year, window)