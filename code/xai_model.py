from typing import Dict

import shap
import matplotlib.pyplot as plt


class XAIModel:
    def __init__(self, config: Dict):
        self.config = config
    
    def run(self, models: dict, data: dict) -> None:
        """
        A function run the explainable model

        Args:
            models: dict: model artifacts
            data: dict: data artifacts
        
        Returns:
            None
        
        Raises:
            None
        """
        breakpoint()
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

            explainer = shap.Explainer(
                models[target]["best"].predict, 
                data[target]["val"][models[target]["features"]]
            )
            shap_values = explainer(data[target]["test"][models[target]["features"]])
            shap.plots.waterfall(shap_values[45], max_display=14, show=True)

            shap.plots.bar(shap_values.abs.max(0), show=False)
            

