from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os
from flask_cors import CORS

# Inicializa app
app = Flask(__name__)

# Configura CORS (permitir todo en dev)
CORS(app, resources={r"/*": {"origins": "*"}})

# Carrega modelo e scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    model = None
    scaler = None
    print(f"Erro ao carregar modelo: {e}")


# Home (Frontend)
@app.route("/", methods=["GET"])
def home():
    try:
        # return send_file(os.path.join(os.getcwd(), "index.html"))
        return send_from_directory(".", "index.html")
        return send_file("index.html", cache_timeout=0)
    except Exception:
        return jsonify({"message": "Cancer Classification API running"})


# Health Check (PRO)
@app.route("/health", methods=["GET"])
def health():
    try:
        if model is None or scaler is None:
            raise Exception("Model or scaler not loaded")

        # Teste de inferência
        dummy = np.zeros((1, 30))
        dummy = scaler.transform(dummy)
        model.predict(dummy)

        return jsonify({
            "status": "ok",
            "model": "loaded",
            "inference": "working"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "details": str(e)
        }), 500


# Predict Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validação básica
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features'"}), 400

        # Validação numérica
        try:
            features = [float(x) for x in data["features"]]
        except:
            return jsonify({"error": "Features must be numeric"}), 400

        # Validação de tamanho
        if len(features) != 30:
            return jsonify({"error": "Expected 30 features"}), 400

        # Conversão para numpy
        features = np.array(features).reshape(1, -1)

        # Normalização
        features = scaler.transform(features)

        # Predição
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()

        return jsonify({
            "prediction": int(prediction),
            "label": "Benigno" if prediction == 1 else "Maligno",
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Execução local (não usado em produção com Gunicorn)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)