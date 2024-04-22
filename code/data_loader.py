import pandas as pd
from typing import Dict

class DataLoader:
    """
    A class for loading the data
    """
    def __init__(self, config: Dict):
        self.folder_path = "data/"
        self.config = config["data"]
    
    def load_data(self) -> pd.DataFrame:
        """
        A function to load the data
        """
        # load the data 
        data = pd.read_csv(self.folder_path + self.config["file_name"])

        # modify the column nanes
        new_col_names_map =  {i: i.lower().replace(" ", "_") for i in data.columns}
        data = data.rename(columns=new_col_names_map)

        # fix the time stamp
        # data["time"] = pd.to_datetime(data["time"], format="%Y-%m-%d %H:%M:%S")
        data = data.set_index("time")

        # cut short the data to only load the data from the CSV
        if self.config["subset_data"]:
            data = data[data.index >= "2021-01-01"]
        
        # subset columns to remove the columns that are identified as not of interest
        if self.config["exclude_columns"]:
            data = data.drop(columns=self.config["exclude_columns"], axis=1)

        return data