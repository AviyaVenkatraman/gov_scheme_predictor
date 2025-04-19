# Government Scheme Predictor

This web application predicts eligible government schemes for users based on their input using machine learning models (Naive Bayes and SVM).

Features

- Predicts schemes based on age, gender, education, income, etc.
- Trained using Naive Bayes and Support Vector Machine (SVM)
- Uses a balanced dataset with categorized schemes (Education, Startup, Research Development)

Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy

Files

- `app.py` — Flask backend that loads models and serves predictions
- `naive_bayes_model.pkl` — Trained Naive Bayes model
- `svm_model.pkl` — Trained SVM model
- `preprocessor.pkl` — Preprocessing pipeline (used before prediction)
- `label_encoder.pkl` — Label encoder for scheme names
- `requirements.txt` — All dependencies

   

