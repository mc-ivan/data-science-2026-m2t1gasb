from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
from flask_cors import CORS

# Inicialização da aplicação Flask
app = Flask(__name__)

# Auto reload de templates
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Habilita CORS
CORS(app)

# Diretório base
BASE_DIR = os.getcwd()

# CARREGAMENTO DO MODELO
try:
    # Carrega modelo treinado
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    
    # Carrega scaler utilizado no pré-processamento
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

except Exception as e:
    model = None
    scaler = None
    print(f"Erro ao carregar modelo: {e}")


# ROTA PRINCIPAL (FRONTEND)
@app.route("/")
def home():
    # Renderiza página HTML
    return render_template("index.html")


# HEALTH CHECK (MONITORAMENTO)
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

# ENDPOINT DE PREDIÇÃO
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recebe dados JSON da requisição
        data = request.get_json()

        # Validação básica
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features'"}), 400

        # Conversão para lista numérica
        try:
            features = [float(x) for x in data["features"]]
        except:
            return jsonify({"error": "Features must be numeric"}), 400

        # Validação do tamanho (dataset tem 9 features)
        if len(features) != 9:
            return jsonify({"error": "Expected 9 features"}), 400

        # Converte para array numpy
        features = np.array(features).reshape(1, -1)

        # Aplica normalização
        features = scaler.transform(features)

        # Realiza predição
        prediction = int(model.predict(features)[0])

        # Obtém probabilidades das classes
        probabilities = model.predict_proba(features)[0]

        proba_healthy = float(probabilities[0])
        proba_cancer = float(probabilities[1])

        # Retorna resultado
        return jsonify({
            "prediction": prediction,
            "label": "Câncer" if prediction == 1 else "Saudável",
            "proba_healthy": proba_healthy,
            "proba_cancer": proba_cancer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# ENDPOINT PARA MÉTRICAS DEL MODELO
@app.route("/metrics")
def metrics():
    try:
        from sklearn.metrics import confusion_matrix, roc_curve, auc

        X_test = joblib.load(os.path.join(BASE_DIR, "X_test.pkl"))
        y_test = joblib.load(os.path.join(BASE_DIR, "y_test.pkl"))
        df = joblib.load(os.path.join(BASE_DIR, "df.pkl"))
        results_df = joblib.load(os.path.join(BASE_DIR, "results_df.pkl"))

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        return jsonify({
            "confusion_matrix": cm,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc,
            "results": results_df.to_dict(),
            "age": df["Age"].tolist(),
            "bmi": df["BMI"].tolist(),
            "glucose": df["Glucose"].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# EXECUÇÃO LOCAL
if __name__ == "__main__":
    # IMPORTANTE: necessário para Docker
    app.run(host="0.0.0.0", port=5000, debug=True)
