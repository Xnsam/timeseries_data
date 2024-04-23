from typing import Dict
import warnings

import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.figsize"] = [16, 9]
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
            # plot pdp plots
            # shap.partial_dependence_plot(
            #     models[target]["target"],
            #     models[target]["best"].predict,
            #     data[target]["test"][models[target]["features"]],
            #     ice=False,
            #     model_expected_value =True,
            #     feature_expected_value=True,
            #     show=False
            # )
            # plt.savefig(f"assets/pdp_{target}.png")
            breakpoint()
            plt.clf()
            explainer = shap.Explainer(
                models[target]["best"].predict, 
                data[target]["val"][models[target]["features"]]
            )
            shap_values = explainer(data[target]["test"][models[target]["features"]])
            shap.plots.waterfall(shap_values[45], max_display=14, show=False)
            plt.savefig(f"{asset_path}/{target}/waterfall.png")
            plt.clf()
            shap.plots.bar(shap_values.abs.max(0), show=False)
            plt.savefig(f"{asset_path}/{target}/bar.png")
            

