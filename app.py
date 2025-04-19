import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the models and preprocessing tools
with open("naive_bayes.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("preprocessor_nb.pkl", "rb") as f:
    preprocessor_nb = pickle.load(f)

with open("preprocessor_svm.pkl", "rb") as f:
    preprocessor_svm = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


@app.route("/")
def home():
    return "Government Scheme Predictor API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert input to dataframe-like format for preprocessing
        input_df = [data]

        # Preprocess for each model
        processed_nb = preprocessor_nb.transform(input_df)
        processed_svm = preprocessor_svm.transform(input_df)

        # Predict
        nb_pred = nb_model.predict(processed_nb)[0]
        svm_pred = svm_model.predict(processed_svm)[0]

        # Decode labels
        nb_scheme = label_encoder.inverse_transform([nb_pred])[0]
        svm_scheme = label_encoder.inverse_transform([svm_pred])[0]

        return jsonify({
            "naive_bayes_prediction": nb_scheme,
            "svm_prediction": svm_scheme
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)


