import pandas as pd

def compute_group_rates(df, target_column, sensitive_column):
    rates = {}

    for group in df[sensitive_column].unique():
        group_data = df[df[sensitive_column] == group]

        if len(group_data) == 0:
            continue

        rates[group] = group_data[target_column].mean()

    return {str(k): float(v) for k, v in rates.items()}


def fairness_summary(df, target_column, sensitive_column):
    
    if target_column not in df.columns or sensitive_column not in df.columns:
        return {"error": "Invalid column names"}

    rates = compute_group_rates(df, target_column, sensitive_column)

    if not rates:
        return {"error": "No valid groups"}

    values = list(rates.values())

    dp = round(max(values) - min(values), 3)

    if max(values) == 0:
        di = 0
    else:
        di = round(min(values) / max(values), 3)

    summary = {
        "demographic_parity_difference": float(dp),
        "disparate_impact_ratio": float(di),
        "group_rates": rates
    }

    # Interpretation
    if dp < 0.1 and di > 0.8:
        summary["fairness"] = "Fair"
    elif dp < 0.2 and di > 0.6:
        summary["fairness"] = "Moderate Bias"
    else:
        summary["fairness"] = "High Bias"

    return summary