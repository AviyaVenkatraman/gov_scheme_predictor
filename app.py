from flask import Flask, request, jsonify, send_file
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load models and encoders
with open('naive_bayes.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('preprocessor_nb.pkl', 'rb') as f:
    preprocessor_nb = pickle.load(f)

with open('preprocessor_svm.pkl', 'rb') as f:
    preprocessor_svm = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])

    input_nb = preprocessor_nb.transform(input_df)
    input_svm = preprocessor_svm.transform(input_df)

    pred_nb = label_encoder.inverse_transform(nb_model.predict(input_nb))[0]
    pred_svm = label_encoder.inverse_transform(svm_model.predict(input_svm))[0]

    return jsonify({
        'Naive Bayes Prediction': pred_nb,
        'SVM Prediction': pred_svm
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)


