import yaml

from data_loader import DataLoader
from feature_engineering import FEEx
from model_loader import ModelLoader
from xai_model import XAIModel


class PipelineExecutor:
    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path))
        self.data_loader = DataLoader(self.config)
        self.model_loader = ModelLoader(self.config)
        self.xai_model = XAIModel(self.config)
        self.feex = FEEx(self.config)
    
    def run_model_pipeline(self) -> None:
        """
        A Function to run the model pipeline
        """
        # load the data
        data = self.data_loader.load_data()
        # implement feature engineering selection
        data = self.feex.apply(data)
        # run the models
        models, data = self.model_loader.run(data)
        # run XAI
        self.xai_model.run(models, data)
        
    def run_fs_pipeline(self) -> None:
        """
        A function to run the feature selection pipeline
        """
        pass

    def run(self) -> None:
        """
        Main function to handle different model pipelines
        """
        if self.config["run"] == "model":
            self.run_model_pipeline()
        elif self.config["run"] == "feature_selection":
            self.run_fs_pipeline()



main_data = PipelineExecutor("config.yaml")
main_data.run()
