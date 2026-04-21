import pandas as pd

def calculate_bias(df, target_column, sensitive_column):
    """
    Calculates bias based on approval rates across groups

    Args:
        df (DataFrame): dataset
        target_column (str): prediction column (0/1)
        sensitive_column (str): column to check bias (e.g., gender)

    Returns:
        dict: bias metrics
    """

    results = {}

    groups = df[sensitive_column].unique()

    group_rates = {}

    for group in groups:
        group_data = df[df[sensitive_column] == group]

        # approval rate (mean of target)
        approval_rate = group_data[target_column].mean()
        group_rates[group] = approval_rate

    # Find max difference
    max_rate = max(group_rates.values())
    min_rate = min(group_rates.values())

    bias_score = max_rate - min_rate

    # results["group_rates"] = group_rates
    # results["bias_score"] = round(bias_score, 3)
    results["group_rates"] = {k: float(v) for k, v in group_rates.items()}
    results["bias_score"] = float(round(bias_score, 3))

    # Simple fairness label
    if bias_score < 0.1:
        results["fairness"] = "Fair"
    elif bias_score < 0.2:
        results["fairness"] = "Moderate Bias"
    else:
        results["fairness"] = "High Bias"

    return results