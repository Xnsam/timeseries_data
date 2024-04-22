from typing import Dict
import pandas as pd
import numpy as np



class FeatureTransformations:

    @staticmethod
    def bool_transform(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        A function to convert bool values to 1's and 0's

        Args:
            data: pd.DataFrame: data frame to be used
            col_name: str: column to be transformed
        
        Returns:
            data: pd.DataFrame: dataset with the transformed column
        
        Raises:
            None
        """
        data[col_name] = data[col_name].astype(int)
        return data
    
    @staticmethod
    def cyclic_transform(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        A function to convert periodic features like days, months etc to a cyclic encoding for
        better representation in ml model building

        Args:
            data: pd.DataFrame: data frame to be used
            col_name: str: column to be transformed
        
        Returns:
            data: pd.DataFrame: dataset with the transformed column
        
        Raises:
            None
        """
        denom_map = {
            "hour": 24, "month": 12, "day_of_week": 7, 
        }

        data[f"sine_{col_name}"] =  np.sin(2 * np.pi * data[col_name] / denom_map[col_name])
        data[f"cos_{col_name}"] = np.cos(2 * np.pi * data[col_name] / denom_map[col_name])
        data = data.drop(columns=[col_name], axis=1)
        return data
    
    @staticmethod
    def ohe_transform(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        A function to convert string variables in the one hot encoding dummy variables

        Args:
            data: pd.DataFrame: data to be worked on
            col_name: str: name of the column to be transformed
        
        Returns:
            data: pd.DataFrame: dataset with the transformed column
        
        Raises:
            None
        """
        tmp_data = pd.get_dummies(data[col_name]).astype(int)
        data = pd.concat([data, tmp_data], axis=1, ignore_index=False)
        data = data.drop(columns=[col_name], axis=1)
        return data

class ImputeData:

    @staticmethod
    def interpolation(data: pd.DataFrame) -> pd.DataFrame:
        """
        A function to do linear interpolation to impute the data for the columns with NA values

        Args:
            data: pd.DataFrame: data to be transformed
        
        Returns:
            data: pd.DataFrame: transformed dataset

        Raises:
            None
        """
        # fetch the list of columns that has missing values in them 
        missing_columns = pd.DataFrame(data.isna().sum().reset_index())
        missing_columns = missing_columns[missing_columns[0] > 0]
        missing_columns = missing_columns["index"].tolist()

        # impute the data
        for column in missing_columns:
            data[column] = data[column].interpolate(method="linear", limit_direction="both")

        return data
        

class FEEx:
    def __init__(self, config: Dict):
        self.config = config["feat_eng"]
        self.fxn_map = {
            "bool": FeatureTransformations.bool_transform, 
            "cyclic": FeatureTransformations.cyclic_transform,
            "ohe": FeatureTransformations.ohe_transform
        }
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        A function to apply the feature transformations
        """
        # perform imputation
        if self.config["impute_missing"]:
            data = ImputeData.interpolation(data)

        # perform feature transformations and encodings
        for enc_type in self.config["fxn_col_map"]:
            for col_name in self.config["fxn_col_map"][enc_type]:
                data = self.fxn_map[enc_type](data, col_name)
        
        data = data.fillna(0)
        
        return data