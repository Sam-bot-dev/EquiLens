import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(df, target_column):
    """
    Train a simple model for explainability
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, X


def shap_explain(model, X):
    """
    Generate SHAP values
    """

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    return shap_values


def get_feature_importance(shap_values, X):
    """
    Get average feature importance (fixed for classification)
    """

    values = shap_values.values

    # 👇 THIS IS THE FIX
    if len(values.shape) == 3:
        # shape = (samples, features, classes)
        values = values[:, :, 1]  # take class 1 (approved)

    importance = abs(values).mean(axis=0)

    feature_importance = dict(zip(X.columns, importance))

    # Convert to float (clean output)
    feature_importance = {k: float(v) for k, v in feature_importance.items()}

    # Sort
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    return feature_importance