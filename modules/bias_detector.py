import pandas as pd

def calculate_bias(df, target_column, sensitive_column):
    results = {}

    # Validate input
    if target_column not in df.columns or sensitive_column not in df.columns:
        return {"error": "Invalid columns"}

    if df[target_column].isnull().any():
        return {"error": "Target column contains null values"}

    groups = df[sensitive_column].unique()
    group_rates = {}

    for group in groups:
        group_data = df[df[sensitive_column] == group]

        if len(group_data) == 0:
            continue

        approval_rate = group_data[target_column].mean()
        group_rates[str(group)] = float(approval_rate)

    if not group_rates:
        return {"error": "No valid groups found"}

    max_rate = max(group_rates.values())
    min_rate = min(group_rates.values())

    bias_score = float(round(max_rate - min_rate, 3))

    # Fairness classification
    if bias_score < 0.1:
        fairness = "Fair"
    elif bias_score < 0.2:
        fairness = "Moderate Bias"
    else:
        fairness = "High Bias"

    results["group_rates"] = group_rates
    results["bias_score"] = bias_score
    results["fairness"] = fairness

    return results