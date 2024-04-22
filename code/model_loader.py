from typing import Dict
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

from pycaret.regression import *
import pickle

# Defaults
warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.figsize"] = [16, 9]
np.random.seed(45)



class CombinedModel:
    def __init__(self):
        self.data_scaler = {}
        self.t_window = 5
        self.artifacts = {}
        self.field_of_interest = ["bld_engcons", "car_chargers"]
        self.forecast_variables = [
            "forecast_temperature", "forecast_feelslike", "forecast_weathertype",
            "forecast_windspeed", "forecast_uvindex", "forecast_precipitationprobability",
        ]

    def transform2d(self, data) -> dict:
        """
        A function to transform data into 2D data for classical machine learning models

        Args:
            data: pd.DataFrame: data to be transformed
            
        Return:
            data: dict: containing data for raw data, x_train_aob, y_train_aob, x_train_car, y_train_car

        """
        # create the target variables by shifting the data by 1
        # NOTE: do not shuffle the data before creating the target calumn as the sequence matters
        data["target_aob"] = data[self.field_of_interest[0]].shift(1)
        data["target_car"] = data[self.field_of_interest[1]].shift(1)
        self.target_cols = ["target_car", "target_aob"]
        data = data.dropna(subset=self.target_cols, axis=0)

        # scale the values of total load for numerical stability and efficiency
        # cols = self.forecast_variables + ["target_car"]
        # self.data_scaler["car"] = MinMaxScaler()
        # self.data_scaler["car"] = self.data_scaler["car"].fit(data[cols])
        # data[cols] = self.data_scaler["car"].transform(data[cols])

        # cols = self.forecast_variables + ["target_aob"]
        # self.data_scaler["aob"] = MinMaxScaler()
        # self.data_scaler["aob"] = self.data_scaler["aob"].fit(data[cols])
        # data[cols] = self.data_scaler["aob"].transform(data[cols])
        
        final_data = {"data": data}

        # need to separate the data because two models are supposed to be made for 
        # forecasting
        # split the data into train and test data
        # NOTE: it is okay to shuffle the data as each sample / row is treated equally for 
        # 2 dimensional data and does not disturb the data


        data_car = data.drop(columns=["bld_engcons", "target_aob"], axis=1)
        train_data, test_data = train_test_split(
            data_car, test_size=0.3, random_state=45, shuffle=True
        )
        validation_data, test_data = train_test_split(
            test_data, 
            test_size=0.5,
            shuffle=True
        )
        final_data["car"] = {
            "data": data,
            "train": train_data,
            "test": test_data,
            "val": validation_data
        }

        data_aob = data.drop(columns=["car_chargers", "target_car"], axis=1)
        train_data, test_data = train_test_split(
            data_aob, test_size=0.3, random_state=45, shuffle=True
        )
        validation_data, test_data = train_test_split(
            test_data, 
            test_size=0.5
        )
        final_data["aob"] = {
            "data": data,
            "train": train_data,
            "test": test_data,
            "val": validation_data
        }
        
        return final_data
    
    def transform3d(self, data) -> dict:
        pass

    def plot_predictions(self, prediction: pd.DataFrame, actual: pd.DataFrame, val: str):
        """
        A function to plot the prediction vs actual data

        Args:
            prediction: pd.DataFrame: values of prediction
            actual: pd.DataFrame: values of actual data
            val: str: name of the target value
        
        Returns:
            None
        
        Raises:
            None

        """
        plot_data = pd.concat([prediction, actual], axis=1, ignore_index=False)
        plot_data.plot(style=".-")
        plt.legend(loc="center left", bbox_to_anchor=(1, 1))
        plt.savefig(f"assets/{val}.png")
        # plt.show()
        plot_data[["prediction"]].plot(style=".-")
        plt.legend(loc="center left", bbox_to_anchor=(1, 1))
        plt.savefig(f"assets/{val}_prediction.png")
        # plt.show()
        plot_data[["actual"]].plot(style=".-")
        plt.legend(loc="center left", bbox_to_anchor=(1, 1))
        plt.savefig(f"assets/{val}_actual.png")
        # plt.show()
    
    def train(self, data:dict) -> dict:
        """
        A function to train the data model

        Args:
            data: dict
        
        Return:
            models: dict
        
        Raises:
            None

        """
        models = {}
        numeric_features1 = list(set(data["car"]["train"].columns.to_list()) - set(["target_car"]))
        # train for car
        pycaret_s1 = setup(
            data = data["car"]["train"],
            test_data = data["car"]["val"],
            target="target_car",
            fold_strategy="timeseries",
            numeric_features = numeric_features1,
            fold=4,
            transform_target=False,
            session_id = 123,
            data_split_shuffle=False,
            fold_shuffle=False
        )
        best1 = pycaret_s1.compare_models(sort="MAE")
        compare_models_csv1 = pycaret_s1.pull()
        models["car"] = {"best": best1, "compare": compare_models_csv1, "setup": pycaret_s1, 
                         "target": "target_car", "features": numeric_features1}
        pycaret_s1.save_model(best1, "assets/car")

        numeric_features2 = list(set(data["aob"]["train"].columns.to_list()) - set(["target_aob"]))
        # train for car
        pycaret_s2 = setup(
            data = data["aob"]["train"],
            test_data = data["aob"]["val"],
            target="target_aob",
            fold_strategy="timeseries",
            numeric_features = numeric_features2,
            fold=4,
            transform_target=False,
            session_id = 123,
            data_split_shuffle=False,
            fold_shuffle=False
        )
        best2 = pycaret_s2.compare_models(sort="MAE")
        compare_models_csv2 = pycaret_s2.pull()
        models["aob"] = {"best": best2, "compare": compare_models_csv2, "setup": pycaret_s2, 
                         "target": "target_aob", "features": numeric_features2}
        pycaret_s2.save_model(best2, "assets/aob")

        for val in ["aob", "car"]:
            predictions = models[val]["setup"].predict_model(models[val]["best"], 
                                                             data=data[val]["test"][models[val]["features"]])
            predictions = predictions.rename(columns={"prediction_label": models[val]["target"]})
            # cols = self.forecast_variables + [models[val]["target"]]
            # predictions[cols] = self.data_scaler[val].inverse_transform(predictions[cols])
            test_data = data[val]["test"]
            # test_data[cols] = self.data_scaler[val].inverse_transform(test_data[cols])
            predictions = predictions[[models[val]["target"]]].rename(columns={models[val]["target"]: "prediction"})
            # predictions["prediction"] = np.abs(round(predictions["prediction"]))
            actual = test_data[[models[val]["target"]]].rename(columns={models[val]["target"]: "actual"})
            self.plot_predictions(predictions, actual, val)
            models[val]["compare"]["model_src"] = val

        csv_data = pd.concat([models["car"]["compare"], models["aob"]["compare"]], axis=0, ignore_index=True)
        csv_data.to_csv("assets/model_comparison.csv", index=False)
        
        return models

    def run(self, data: dict) -> dict:
        """
        A function to run the pipeline for training the data

        Args:
            data: dict: dictionary of dataset for building and car
        
        Raises:
            None
        
        Return:
            models: dict
        """
        data = self.transform2d(data)
        models = self.train(data)
        return models


class SeparateModel:
    def __init__(self):
        self.data_scaler = MinMaxScaler()

    def transform(self, data):
        pass

    def train(self, data):
        pass

    def test(self, model, data):
        pass

    def run(self, data):
        pass


class WeekendModel:
    def __init__(self):
        pass

    def transform(self, data):
        pass

    def train(self, data):
        pass

    def test(self, model, data):
        pass

    def run(self, data):
        pass


class ModelLoader:
    def __init__(self, config: Dict):
        self.config = config["model"]
    
    def run(self, data: pd.DataFrame) -> dict:
        if self.config["combined_model"]:
            data1 = data.drop(
                columns=self.config["combined_model_drop_cols"], axis=1)
            model_exec = CombinedModel()
            model_exec.run(data1)
        elif self.config["separate_model"]:
            data1 = data.drop(
                columns=self.config["separate_model_drop_cols"], axis=1)
            model_exec = SeparateModel()
            model_exec.run(data1)
        elif self.config["weekend_model"]:
            pass

        return model_exec.artifacts