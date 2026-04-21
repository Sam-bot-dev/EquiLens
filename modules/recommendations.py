def generate_recommendations(bias_score, fairness_label, feature_importance=None):
    """
    Generate actionable recommendations to reduce bias

    Args:
        bias_score (float)
        fairness_label (str)
        feature_importance (dict, optional)

    Returns:
        list of recommendations
    """

    recommendations = []

    # 🟢 If model is fair
    if fairness_label == "Fair":
        recommendations.append("✅ Model is fair. No major action needed.")
        return recommendations

    # 🔴 High Bias Warning
    if bias_score > 0.2:
        recommendations.append("⚠️ High bias detected. Immediate action recommended.")

    # 📊 Data-level fixes
    recommendations.append("🔄 Rebalance dataset (ensure equal representation of all groups)")
    recommendations.append("📉 Reduce skew in training data distribution")

    # 🧹 Feature-level fixes
    recommendations.append("🧹 Remove or limit sensitive attributes (e.g., gender, race)")
    recommendations.append("🔍 Check for proxy variables that indirectly encode sensitive attributes")

    # ⚖️ Model-level fixes
    recommendations.append("⚖️ Apply reweighting or fairness-aware algorithms")
    recommendations.append("🧠 Train with fairness constraints (e.g., equal opportunity)")

    # 📈 Monitoring
    recommendations.append("📊 Continuously monitor model predictions for bias in production")

    # 🔥 BONUS: Feature-based insight (VERY IMPRESSIVE)
    if feature_importance:
        top_feature = list(feature_importance.keys())[0]
        recommendations.append(f"🚨 Feature '{top_feature}' has high influence — review its impact on fairness")

    return recommendations