from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models and preprocessing tools
with open("naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/")
def home():
    return "Government Scheme Predictor is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_features = [[
            data["Age"],
            data["Gender"],
            data["Income"],
            data["Employment"],
            data["Education"],
            data["Category"],
            data["Disability"],
            data["Marital_Status"],
            data["Area"]
        ]]

        # Preprocess the input
        X = preprocessor.transform(input_features)

        # Predict with Naive Bayes and SVM
        nb_pred = nb_model.predict(X)
        svm_pred = svm_model.predict(X)

        # Decode predictions
        nb_scheme = label_encoder.inverse_transform(nb_pred)[0]
        svm_scheme = label_encoder.inverse_transform(svm_pred)[0]

        return jsonify({
            "naive_bayes_prediction": nb_scheme,
            "svm_prediction": svm_scheme
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

