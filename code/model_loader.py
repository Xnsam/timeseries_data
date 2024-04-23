from typing import Dict
import warnings
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
import matplotlib.pyplot as plt

from pycaret.regression import *

# Defaults
warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.figsize"] = [16, 9]
np.random.seed(45)



class CombinedModel:
    def __init__(self):
        self.asset_path = "assets/combined/"
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

    def plot_predictions(self, prediction: pd.DataFrame, actual: pd.DataFrame, val: str, save_path: str=None):
        """
        A function to plot the prediction vs actual data

        Args:
            prediction: pd.DataFrame: values of prediction
            actual: pd.DataFrame: values of actual data
            val: str: name of the target value
            save_path: str: name of the save path
        
        Returns:
            None
        
        Raises:
            None

        """
        if save_path is None:
            save_path = self.asset_path

        plot_data = pd.concat([prediction, actual], axis=1, ignore_index=False)
        data = [plot_data, plot_data[["prediction"]], plot_data[["actual"]]]
        path_names = [f"{save_path}/{val}/prediction_v_actual.png",
                       f"{save_path}/{val}/prediction.png", f"{save_path}/{val}/actual.png"]
        
        for _data, _path_name in zip(data, path_names):
            plt.clf()
            _data.plot(style=".-")
            plt.legend(loc="center left", bbox_to_anchor=(1, 1))
            plt.xticks(rotation=45)
            plt.savefig(_path_name)
        
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
        target_vals = {"aob": "target_aob", "car": "target_car"}
        for val in ["aob", "car"]:
            numeric_features1 = list(set(data[val]["train"].columns.to_list()) - set([target_vals[val]]))
            # train for car
            pycaret_s1 = setup(
                data = data[val]["train"],
                test_data = data[val]["val"],
                target=target_vals[val],
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
            models[val] = {"best": best1, "compare": compare_models_csv1, "setup": pycaret_s1, 
                            "target": target_vals[val], "features": numeric_features1}
            pycaret_s1.save_model(best1, f"assets/{val}")

            predictions = models[val]["setup"].predict_model(models[val]["best"],
                                                             data=data[val]["test"][models[val]["features"]])
            predictions = predictions.rename(columns={"prediction_label": models[val]["target"]})
            test_data = data[val]["test"]
            predictions = predictions[[models[val]["target"]]].rename(columns={models[val]["target"]: "prediction"})
            actual = test_data[[models[val]["target"]]].rename(columns={models[val]["target"]: "actual"})
            self.plot_predictions(predictions, actual, val)
            models[val]["compare"]["model_src"] = val

        csv_data = pd.concat([models["car"]["compare"], models["aob"]["compare"]], axis=0, ignore_index=True)
        csv_data.to_csv(f"{self.asset_path}/model_comparison.csv", index=False)
        
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
        print("running combined model")
        data = self.transform2d(data)
        models = self.train(data)
        return models, data


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




class FeatSelect:
    def __init__(self, model_type: str):
        self.assets_path = f"assets/{model_type}/fs"
        if model_type == "combined":
            self.model = CombinedModel()
        elif model_type == "separate": 
            self.model = SeparateModel()
        else:
            self.model = WeekendModel()
    
    def train(self, data: dict) -> dict:
        """
        A function to train and run the feature selection pipeline

        Args:
            data: pd.DataFrame: data to be used for
        
        Returns:
            models: dict: all models after feature selection
        
        Raises:
            None
        """
        models = {}
        target_vals = {"aob": "target_aob", "car": "target_car"}
        for val in ["aob", "car"]:
            save_path = f"{self.assets_path}/{val}"
            os.makedirs(save_path, exist_ok=True)
            numeric_features = list(set(data[val]["train"].columns.to_list()) - set([target_vals[val]]))
            pycaret_s = setup(
                data=data[val]["train"],
                test_data = data[val]["val"],
                target=target_vals[val],
                fold_strategy="timeseries",
                numeric_features= numeric_features,
                fold=4,
                transform_target=False,
                session_id=123,
                data_split_shuffle=False,
                fold_shuffle=False,
                normalize=True,
                normalize_method="robust",
                polynomial_features=True,
                feature_selection=True,
                feature_selection_method="classic",
                remove_multicollinearity=True,
                low_variance_threshold=0.1
            )
            best = pycaret_s.compare_models(sort="MAE")
            compare_models = pycaret_s.pull()
            selector = pycaret_s.get_config("pipeline").named_steps["feature_selection"].transformer
            models[val] = {
                "best": best, "compare": compare_models, "setup": pycaret_s,
                "target": target_vals[val], "features": numeric_features,
                "selected_features": selector.get_feature_names_out().tolist()
            }
            pycaret_s.save_model(best, f"{save_path}/{val}")
            with open(f"{save_path}/selected_features.txt", "w") as f:
                f.write(",".join(models[val]["selected_features"]))
            test_data = data[val]["test"]
            predictions = pycaret_s.predict_model(best, data=test_data[numeric_features])
            predictions = predictions.rename(columns={"prediction_label": target_vals[val]})
            predictions = predictions[[target_vals[val]]].rename(columns={models[val]["target"]: "prediction"})
            actual = test_data[[target_vals[val]]].rename(columns={target_vals[val]: "actual"})
            self.model.plot_predictions(predictions, actual, val, save_path)
            models[val]["compare"]["model_src"] = val
        
        csv_data = pd.concat(
            [models["car"]["compare"], models["aob"]["compare"]], axis=0, ignore_index=True
        )
        csv_data.to_csv(f"{save_path}/model_comparison.csv", index=False)
        return models


    def run(self, data: pd.DataFrame) -> dict:
        """
        A function to run the pipeline

        Args:
            data: pd.DataFrame: data to be used for
        
        Returns:
            models: dict: all models after feature selection
        
        Raises:
            None
        """
        print("running feature selection model")
        # transform the data
        transformed_data = self.model.transform2d(data)
        models = self.train(transformed_data)
        return models, transformed_data


class ModelLoader:
    def __init__(self, config: Dict):
        self.config = config["model"]
    
    def run(self, data: pd.DataFrame) -> dict:
        if self.config["combined_model"]:
            data = data.drop(columns=self.config["combined_model_drop_cols"], axis=1)
            model_exec = CombinedModel()
        elif self.config["separate_model"]:
            data = data.drop(columns=self.config["separate_model_drop_cols"], axis=1)
            model_exec = SeparateModel()
        elif self.config["weekend_model"]:
            pass
        elif self.config["fs_select_combined"]:
            model_exec = FeatSelect(model_type="combined")
        elif self.config["fs_select_weekend"]:
            model_exec = FeatSelect(model_type="weekend")
        elif self.config["fs_select_separate"]:
            model_exec = FeatSelect(model_type="separate")
        
        self.asset_path = model_exec.asset_path
        return model_exec.run(data)
    

        