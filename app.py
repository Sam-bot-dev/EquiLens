from flask import Flask, render_template, request, jsonify
import pandas as pd

from modules.bias_detector import calculate_bias
from modules.fairness_metrics import fairness_summary
from modules.explainability import train_model, shap_explain, get_feature_importance
from modules.recommendations import generate_recommendations

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

def analyze_groups(group_rates):
    worst_group = min(group_rates, key=group_rates.get)
    best_group = max(group_rates, key=group_rates.get)

    return worst_group, best_group

def generate_summary(bias_score, fairness_label, worst_group, dp, di):
    return (
        f"The model shows {fairness_label} (Bias Score: {bias_score}). "
        f"The most affected group is '{worst_group}'. "
        f"Demographic Parity Difference is {dp:.3f} and Disparate Impact Ratio is {di:.3f}."
    )

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 1. Read file
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file)

        # 2. Get inputs
        target_column = request.form.get("target", "approved")
        sensitive_column = request.form.get("sensitive", "gender")

        # 2a. Validate dataset is not empty
        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        # 2b. Validate columns exist
        if target_column not in df.columns:
            return jsonify({"error": f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"}), 400

        if sensitive_column not in df.columns:
            return jsonify({"error": f"Sensitive column '{sensitive_column}' not found. Available columns: {list(df.columns)}"}), 400

        # 2c. Validate target column is binary
        if df[target_column].nunique() > 2:
            return jsonify({"error": "Target column must be binary (0 and 1 values only)"}), 400

        # 3. Bias detection
        bias_result = calculate_bias(df, target_column, sensitive_column)
        if "error" in bias_result:
            return jsonify({"error": bias_result["error"]}), 400

        # 4. Fairness metrics
        fairness_result = fairness_summary(df, target_column, sensitive_column)
        if "error" in fairness_result:
            return jsonify({"error": fairness_result["error"]}), 400

        # 5. Prepare data (encoding)
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

        # 6. Train model
        model, X = train_model(df_encoded, target_column)

        # 7. Explainability (SHAP) with fail-safe
        X_sample = X.sample(min(100, len(X)))
        try:
            shap_values = shap_explain(model, X_sample)
            importance = get_feature_importance(shap_values, X_sample)
        except Exception:
            importance = {}

        # 7a. Limit feature importance to top 8 features
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])

        # 8. Group analysis
        group_rates = fairness_result.get("group_rates", {})
        worst_group, best_group = analyze_groups(group_rates) if group_rates else ("Unknown", "Unknown")

        # 9. Generate summary
        dp = float(fairness_result.get("demographic_parity_difference", 0))
        di = float(fairness_result.get("disparate_impact_ratio", 0))
        summary_text = generate_summary(
            bias_result["bias_score"],
            bias_result["fairness"],
            worst_group,
            dp,
            di
        )

        # 10. Recommendations
        recommendations = generate_recommendations(
            bias_result["bias_score"],
            bias_result["fairness"],
            importance
        )

        # Clean numbers for JSON serialization
        importance = {k: float(v) for k, v in importance.items()}
        group_rates = {k: float(v) for k, v in group_rates.items()}
        bias_score = float(bias_result["bias_score"])

        # 11. Return final JSON
        return jsonify({
            "summary": summary_text,
            "bias": {
                "bias_score": bias_score,
                "fairness": bias_result["fairness"]
            },
            "fairness": {
                "demographic_parity_difference": dp,
                "disparate_impact_ratio": di,
                "group_rates": group_rates
            },
            "feature_importance": importance,
            "insights": {
                "most_affected_group": worst_group,
                "least_affected_group": best_group
            },
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)