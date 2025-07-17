import shap
import pandas as pd
from typing import Any

def explain_tabular(model, X_background: pd.DataFrame, X_input: pd.DataFrame) -> Any:
    """
    Generates SHAP explanations for tabular models (e.g., LightGBM, XGBoost, Sklearn).
    """
    explainer = shap.Explainer(model, X_background)
    shap_values = explainer(X_input)
    return shap_values

def plot_tabular_explanation(shap_values, X_input):
    shap.initjs()
    shap.summary_plot(shap_values, X_input)

if __name__ == "__main__":
    # Example:
    # model = ...
    # X_background = pd.read_csv(...)
    # X_input = pd.read_csv(...)
    # shap_values = explain_tabular(model, X_background, X_input)
    # plot_tabular_explanation(shap_values, X_input)
    print("Tabular explainability module.")