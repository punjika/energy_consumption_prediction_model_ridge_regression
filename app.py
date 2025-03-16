from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("xgboost_energy_model.pkl")  # Update path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get input data as JSON
        features = np.array(data["features"]).reshape(1, -1)  # Convert to numpy array
        prediction = model.predict(features)
        return jsonify({"Predicted Energy Consumption": float(prediction[0])})  # Convert to float for JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 400

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides the PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=True)
  # Ensure this runs before sending a request
