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
        df = pd.read_csv(file)

        target_column = request.form.get("target", "approved")
        sensitive_column = request.form.get("sensitive", "gender")

        # Bias Detection
        bias_result = calculate_bias(df, target_column, sensitive_column)

        # Fairness Metrics
        fairness_result = fairness_summary(df, target_column, sensitive_column)

        # Encode categorical (simple)
        df_encoded = df.copy()
        df_encoded[sensitive_column] = df_encoded[sensitive_column].astype('category').cat.codes

        # Explainability
        model, X = train_model(df_encoded, target_column)
        shap_values = shap_explain(model, X)
        importance = get_feature_importance(shap_values, X)
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


if __name__ == '__main__':
    app.run(debug=True)