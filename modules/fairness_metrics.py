import pandas as pd

def demographic_parity(df, target_column, sensitive_column):
    """
    Calculates demographic parity difference
    """

    groups = df[sensitive_column].unique()
    rates = {}

    for group in groups:
        group_data = df[df[sensitive_column] == group]
        rates[group] = group_data[target_column].mean()

    max_rate = max(rates.values())
    min_rate = min(rates.values())

    dp_diff = max_rate - min_rate

    return round(dp_diff, 3), rates


def disparate_impact(df, target_column, sensitive_column):
    """
    Calculates disparate impact ratio
    """

    groups = df[sensitive_column].unique()
    rates = {}

    for group in groups:
        group_data = df[df[sensitive_column] == group]
        rates[group] = group_data[target_column].mean()

    max_rate = max(rates.values())
    min_rate = min(rates.values())

    if max_rate == 0:
        return 0, rates

    di_ratio = min_rate / max_rate

    return round(di_ratio, 3), rates


def fairness_summary(df, target_column, sensitive_column):
    """
    Combines all fairness metrics
    """

    dp, dp_rates = demographic_parity(df, target_column, sensitive_column)
    di, di_rates = disparate_impact(df, target_column, sensitive_column)

    summary = {
    "demographic_parity_difference": float(dp),
    "disparate_impact_ratio": float(di),
    "group_rates": {k: float(v) for k, v in dp_rates.items()}
    }

    # Interpretation
    if dp < 0.1 and di > 0.8:
        summary["fairness"] = "Fair"
    elif dp < 0.2 and di > 0.6:
        summary["fairness"] = "Moderate Bias"
    else:
        summary["fairness"] = "High Bias"

    return summary