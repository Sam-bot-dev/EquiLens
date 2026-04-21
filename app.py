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


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']

        if not file:
            return jsonify({"error": "No file uploaded"})

        df = pd.read_csv(file)

        target_column = request.form.get("target", "approved")
        sensitive_column = request.form.get("sensitive", "gender")

        # Bias Detection
        bias_result = calculate_bias(df, target_column, sensitive_column)

        if "error" in bias_result:
            return jsonify({"error": bias_result["error"]})

        # Fairness Metrics
        fairness_result = fairness_summary(df, target_column, sensitive_column)

        # Encode ALL categorical columns
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes

        # Explainability (SAFE MODE)
        model, X = train_model(df_encoded, target_column)

        X_sample = X.sample(min(100, len(X)))
        shap_values = shap_explain(model, X_sample)

        importance = get_feature_importance(shap_values, X_sample)
        importance = {k: round(v, 3) for k, v in importance.items()}

        # Recommendations
        recommendations = generate_recommendations(
            bias_result["bias_score"],
            bias_result["fairness"],
            importance
        )

        return jsonify({
            "summary": f"⚠️ The model shows {bias_result['fairness']} (Bias Score: {bias_result['bias_score']})",
            "bias": bias_result,
            "fairness": fairness_result,
            "feature_importance": importance,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)})
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)