import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)  # ✅ Correct use of __name__

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load and preprocess the dataset
dataset = pd.read_csv('diabetes.csv')
dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    else:
        pred = "You don't have Diabetes."

    return render_template('index.html', prediction_text=pred)

# ✅ Correct __name__ check
if __name__ == "__main__":
    app.run(debug=True)
