from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

# Load models and encoders
with open("naive_bayes .pkl", "rb") as f:
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
    with open('index.html', 'r') as f:
        return render_template_string(f.read())  # Serve the HTML form

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'post':
        data = request.form
        df = pd.DataFrame([{
            'Category': data['category'],
            'Education': data['education'],
            'Employment': data['employment'],
            'Marital status': data['marital status'],
            'Area': data['area']
    }])

    # Preprocess
    input_nb = preprocessor_nb.transform(df)
    input_svm = preprocessor_svm.transform(df)

    # Predictions
    nb_pred = label_encoder.inverse_transform(nb_model.predict(input_nb))[0]
    svm_pred = label_encoder.inverse_transform(svm_model.predict(input_svm))[0]

    return jsonify({
        "naive_bayes": nb_pred,
        "svm": svm_pred
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



