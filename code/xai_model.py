from typing import Dict
import warnings

import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.figsize"] = [40, 12]
np.random.seed(45)

class XAIModel:
    def __init__(self, config: Dict):
        self.config = config
    
    def run(self, models: dict, data: dict, asset_path: str) -> None:
        """
        A function run the explainable model

        Args:
            models: dict: model artifacts
            data: dict: data artifacts
            asset_path: str: path of the asset to save
        
        Returns:
            None
        
        Raises:
            None
        """
        for target in ["car", "aob"]:
            explainer = shap.Explainer(models[target]["best"].predict, data[target]["val"][models[target]["features"]])
            shap_values = explainer(data[target]["test"][models[target]["features"]])
            shap.plots.waterfall(shap_values[45], max_display=14, show=False)
            plt.gcf().set_size_inches(16, 9)
            plt.savefig(f"{asset_path}/{target}/waterfall.png")
            shap.plots.bar(shap_values.abs.max(0), show=False)
            plt.gcf().set_size_inches(16, 9)
            plt.savefig(f"{asset_path}/{target}/bar.png")
            shap.plots.beeswarm(shap_values, show=False)
            plt.gcf().set_size_inches(16, 9)
            plt.savefig(f"{asset_path}/{target}/beeswarm.png")
            shap.plots.heatmap(shap_values, show=False)
            plt.gcf().set_size_inches(16, 9)
            plt.savefig(f"{asset_path}/{target}/heatmap.png")
            plt.clf()
            

