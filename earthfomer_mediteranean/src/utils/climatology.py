from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
import xarray as xr
import os
from typing import List, Dict
from enums import Resolution
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression
import pandas as pd 

"""Using arima in a monthly model is challenging, because it waits for continuous values . And splitting into 
month disrupts the continuity. So, we will use linear regression for monthly model.


    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import LinearRegression
import pandas as pd 

class ClimatologyMonthlyLinearReg:
    def __init__(self, data: xr.DataArray, var: str, split_years: dict):
        self.data = data
        self.split_years = split_years
        self.var = var
        self.models = {}
        self.train_years = split_years["train"]
        self.test_years = split_years["test"]
        self.month_means = {month: self.get_average_by_month(month) for month in range(1, 13)}
        self.month_mapping = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                         7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        self._train_models()

    def get_average_by_month(self, month: int):
        month_data = self.data[self.var].where(self.data['time.month'] == month, drop=True)
        mean_data = month_data.groupby("time.year").mean(dim=['time', 'latitude', 'longitude'])
        return mean_data

    def _train_models(self):
        for month in range(1, 13):
            month_mean = self.month_means[month]
            self.train_years = [year for year in self.train_years if year in month_mean.year]
            train_data = month_mean.sel(year=self.train_years)
            x_train = np.array(train_data.year).reshape(-1, 1)
            y_train = np.array(train_data)
            model = LinearRegression()
            model.fit(x_train, y_train)
            self.models[month] = model

    def predict(self, year: int, month: int):
        if month in self.models:
            model = self.models[month]
            prediction = model.predict(np.array([[year]]))
            return prediction[0]
        else:
            raise ValueError("Model for month {} is not trained.".format(month))

    def evaluate(self):
        results = {}
        combined_data = []
        for month in range(1, 13):
            month_means = self.month_means[month]
            test_data = month_means.sel(year=self.test_years)
            x_test = np.array(test_data.year).reshape(-1, 1)
            y_test = np.array(test_data)

            if month in self.models:
                model = self.models[month]
                y_pred = model.predict(x_test)
                mse = np.mean((y_test - y_pred) ** 2)
                results[month] = {'y_test': y_test, 'y_pred': y_pred, 'mse': mse}
                
                for i, year in enumerate(self.test_years):
                    combined_data.append({'year': year, 'month': month, 'y_test': y_test[i], 'y_pred': y_pred[i]})
            else:
                results[month] = {'y_test': None, 'y_pred': None, 'mse': None}
        
        combined_df = pd.DataFrame(combined_data)
        combined_df['date'] = pd.to_datetime(combined_df[['year', 'month']].assign(day=1))
        combined_df = combined_df.sort_values('date')
        
        average_mse = np.mean([result['mse'] for result in results.values() if result['mse'] is not None])
        return results, average_mse, combined_df
    
    def plot_yearly_results(self):
        results, _, combined_df = self.evaluate()
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        x_test = combined_df['date']
        y_test = combined_df['y_test']
        y_pred = combined_df['y_pred']
        
        ax.plot(x_test, y_test, label='y_test', marker='o', color = 'purple')
        ax.plot(x_test, y_pred, label='y_pred', marker='x', color = 'orange')
        ax.set_title('Results on test set: ground truth vs predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel(self.var)
        ax.legend()
        plt.show()


    def plot_test_results(self):
        results, _ , _= self.evaluate()
        
        fig, axs = plt.subplots(4, 3, figsize=(15, 20))
        axs = axs.flatten()
        
        for month in range(1, 13):
            result = results[month]
            if result['y_test'] is not None:
                x_test = self.month_means[month].sel(year=self.test_years).year
                y_test = result['y_test']
                y_pred = result['y_pred']
                
                ax = axs[month - 1]
                ax.plot(x_test, y_test, label='y_test', marker='o', color = 'purple')
                ax.plot(x_test, y_pred, label='y_pred', marker='x', color = 'orange')
                ax.set_title(f'Month {self.month_mapping[month]}')
                ax.set_xlabel('Year')
                ax.set_ylabel(self.var)
                ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle('Results on test set : ground truth vs predicted for every month', fontsize=16)
        plt.show()

class ClimatologyARIMA:
    def __init__(self, data: xr.DataArray, var: str, split_years: dict):
        self.data = data
        self.split_years = split_years
        self.var = var
        self.train_years = split_years["train"]
        self.test_years = split_years["test"]
        self.model = None
        self.train_data = None
        self.test_data = None
        
        self._prepare_data()
        self._train_model()

    def _prepare_data(self):
        df_list = []
        for year in self.data['time.year'].values:
            for month in range(1, 13):
                month_data = self.data[self.var].sel(time=str(year)).where(self.data['time.month'] == month, drop=True)
                mean_data = month_data.mean(dim=['time', 'latitude', 'longitude']).values
                df_list.append({'year': year, 'month': month, self.var: mean_data})
        
        data = pd.DataFrame(df_list)
        data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
        data.set_index('date', inplace=True)

        self.train_data = data[data['year'].isin(self.train_years)][self.var]
        self.test_data = data[data['year'].isin(self.test_years)][self.var]

    def _train_model(self):
        self.model = ARIMA(self.train_data, order=(5, 1, 0,4))  # (p, d, q) order can be adjusted
        self.model = self.model.fit()

    def predict(self, steps: int):
        forecast = self.model.get_forecast(steps=steps)
        return forecast.predicted_mean

    def evaluate(self):
        steps = len(self.test_data)
        predictions = self.predict(steps)
        y_test = self.test_data.values
        y_pred = predictions.values
        
        mse = np.mean((y_test - y_pred) ** 2)
        results = {'y_test': y_test, 'y_pred': y_pred, 'mse': mse}
        
        return results

    def plot_test_results(self):
        results = self.evaluate()
        
        test_dates = self.test_data.index
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, y_test, label='y_test', marker='o')
        plt.plot(test_dates, y_pred, label='y_pred', marker='x')
        plt.title('Test Results: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel(self.var)
        plt.legend()
        plt.show()




if __name__ == "__main__":
    data = xr.open_dataarray("/scistor/ivm/data_catalogue/reanalysis/ERA5_0.25/PR/PR_era5_MED_1degr_19400101_20240229.nc")
    split_years = {"train": list(range(1940, 2010)), "test": list(range(2010, 2022))}
    climatology = ClimatologyMonthlyLinearReg(data, "tp", split_years)
    print(climatology.evaluate())
    print(climatology.predict(2021, 1))
    print(climatology.predict(2021, 12))