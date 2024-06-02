import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from skforecast.model_selection import backtesting_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg

import plotly
import plotly.graph_objects as go


from astral import LocationInfo
from astral.sun import sun



from custom_utils import *

data_path = "./../data/"
file_name = "data.csv"
data = pd.read_csv(data_path + file_name)
data = first_preprocess(data, data_path)
data.head()

exo_columns = ['forecast_temperature', 'forecast_feelslike', 'forecast_weathertype',
       'forecast_windspeed', 'forecast_uvindex',
       'forecast_precipitationprobability', 'forecast_visibility',
       'week_of_year', 'daylight_hours', 'is_daylight', 'sine_hour',
       'cos_hour', 'sine_month', 'cos_month', 'sine_day_of_week',
       'cos_day_of_week', 'sine_forecast_winddirection',
       'cos_forecast_winddirection', 'sine_sunrise_hour', 'cos_sunrise_hour',
       'sine_sunset_hour', 'cos_sunset_hour']
endo_columns = ['weekend', 'bank_holiday', 'day_of_month', 'year', "others"]
target_column = ["y"]
column_name = exo_columns + endo_columns  + target_column
test_date = '2023-04-01'

# transform the data
others_data = data.copy()
# "hot_water", "sockets", "lighting"
others_data["others"] = others_data["hot_water"] + others_data["sockets"] + others_data["lighting"]

others_data["y"] = others_data["others"].shift(1)
others_data = others_data.dropna(axis=0)

others_data.head()

others_data_scaler, others_transformed_data = create_std_scaler(others_data, column_name[:-1])
others_target_scaler, others_transformed_target_data = create_std_scaler(others_data, column_name[-1])

others_transformed_data = pd.merge(others_transformed_data, others_transformed_target_data,
                               left_index=True,
                               right_index=True)

# # Training and testing data for comms and services
train_data = others_transformed_data[others_transformed_data.index < test_date][column_name]
test_data = others_transformed_data[others_transformed_data.index >= test_date][column_name]
train_data.shape, test_data.shape


params = {
    'n_estimators': 100,
    'criterion':  'absolute_error',
    'max_depth': 5,
    'max_features': "sqrt",
    'random_state': 994564,
    'verbose': 0
}
forecaster = ForecasterAutoreg(regressor = RandomForestRegressor(**params),
                               lags = 24)

metric, predictions = backtesting_forecaster(
    forecaster         = forecaster,
    y                  = train_data['y'],
    exog               = train_data[exo_columns + endo_columns],
    steps              = 24,
    metric             = 'mean_absolute_error',
    initial_train_size = train_data[train_data.index < '2023-01-01'].shape[0],
    refit              = True,
    n_jobs             = 'auto',
    verbose            = False,
    show_progress      = True
)

print(f"Backtest error: {metric:.2f}")
predictions.head()

print(metric)

validation = train_data[train_data.index >= "2023-01-01"]
validation = validation[["y"]]
validation = pd.merge(validation, predictions, left_index=True, right_index=True)

validation.head()

validation[["pred", "y"]] = others_target_scaler.inverse_transform(validation[["pred", "y"]])
validation.head()



cal_metrics(validation["y"].to_numpy(), validation["pred"].to_numpy())

lags_grid = [24, (4, 8, 12, 16, 24)]
max_features = ["sqrt", "log2"]

def search_space(trial):
  _search_space = {
      "n_estimators": trial.suggest_int("n_estimators", 150, 250, step=50),
      "max_depth": trial.suggest_int("max_depth", 3, 10, step=1),
      "lags": trial.suggest_categorical("lags", lags_grid),
      "criterion": "absolute_error",
      "max_features": trial.suggest_categorical("max_features", max_features),
      "warm_start": True
  }
  return _search_space

from skforecast.model_selection import bayesian_search_forecaster

results_search, frozen_trial = bayesian_search_forecaster(
    forecaster=forecaster,
    y=train_data["y"],
    steps=24,
    metric='mean_absolute_error',
    search_space=search_space,
    initial_train_size=train_data[train_data.index < "2023-01-01"].shape[0],
    refit=False,
    n_trials=20,
    return_best=True,
    n_jobs="auto",
    verbose=False,
    show_progress=True
)

results_search.sort_values(by=["mean_absolute_error"], ascending=False).head()

results_search.to_csv("results_search_others.csv", index=False)




