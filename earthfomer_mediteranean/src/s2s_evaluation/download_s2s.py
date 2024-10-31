import climetlab as cml
import xarray as xr

# Helper function to generate the same dates for multiple years
def generate_dates(years):
    base_dates = [
        # Janvier
        "0102", "0109", "0116", "0123", "0130",
        # Février
        "0206", "0213", "0220", "0227",
        # Mars
        "0305", "0312", "0319", "0326",
        # Octobre
        "1001", "1008", "1015", "1022", "1029",
        # Novembre
        "1105", "1112", "1119", "1126",
        # Décembre
        "1203"
    ]
    dates = []
    for year in years:
        dates += [int(f"{year}{date}") for date in base_dates]
    return dates

dates_2020 = generate_dates([2020])

s2s_tp_input_hindcast = cml.load_dataset("s2s-ai-challenge-training-input",
                 origin='ecmwf',
                date= dates_2020,
                format='netcdf',
                parameter='tp').to_xarray()
# only select year that is from 2016
s2s_tp_input_hindcast = s2s_tp_input_hindcast.sel(forecast_time=s2s_tp_input_hindcast.forecast_time.dt.year >= 2016)["tp"]

s2s_tp_input_2020 =  cml.load_dataset("s2s-ai-challenge-test-input",
                                origin='ecmwf',
                                date=dates_2020,
                                format='netcdf',
                                parameter='tp').to_xarray()["tp"]

# fusion all the data 
# Concaténer les DataArrays 'tp'
tp_concatenated = xr.concat([s2s_tp_input_hindcast, s2s_tp_input_2020], dim='forecast_time')

# Créer un nouveau Dataset avec la variable 'tp' concaténée
s2s_tp_input = xr.Dataset({'tp': tp_concatenated})


# save the array
s2s_tp_input.to_netcdf('s2s_tp_input.nc')