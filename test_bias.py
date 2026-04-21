import pandas as pd
from modules.bias_detector import calculate_bias
from modules.fairness_metrics import fairness_summary
from modules.explainability import train_model, shap_explain, get_feature_importance

# Sample dataset
data = {
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
    "approved": [1, 0, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# Bias + fairness
result = calculate_bias(df, "approved", "gender")
fairness = fairness_summary(df, "approved", "gender")

# Encode
df_encoded = df.copy()
df_encoded["gender"] = df_encoded["gender"].map({"M": 1, "F": 0})

# Model + SHAP
model, X = train_model(df_encoded, "approved")
shap_values = shap_explain(model, X)
importance = get_feature_importance(shap_values, X)

print(result)
print(fairness)
print("Feature Importance:", importance)