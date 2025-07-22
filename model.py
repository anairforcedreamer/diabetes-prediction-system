#!/usr/bin/env python
# coding: utf-8

# # Description:                                                                                                                   
# The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
# 
# Attributes:
# 1. Glucose Level
# 2. BMI
# 3. Blood pressure
# 4. Pregnancies
# 5. Skin thickness
# 6. Insulin
# 7. Diabetes pedigree function
# 8. Age
# 9. Outcome

# # Step 0: Import libraries and Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
dataset = pd.read_csv('diabetes.csv')

# Preprocessing
print("Preprocessing data...")
dataset_X = dataset.iloc[:, [1, 4, 5, 7]].values
dataset_Y = dataset.iloc[:, 8].values

sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)

X = pd.DataFrame(dataset_scaled)
Y = dataset_Y

# Split the data
print("Splitting dataset into training and testing...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])

# Train the model
print("Training the model...")
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, Y_train)

# Evaluate the model
accuracy = svc.score(X_test, Y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Save the model
print("Saving model to model.pkl...")
pickle.dump(svc, open('model.pkl', 'wb'))

# Load the model and test a prediction
model = pickle.load(open('model.pkl', 'rb'))

# Example test prediction (optional)
sample = np.array([[86, 66, 26.6, 31]])
sample_scaled = sc.transform(sample)
prediction = model.predict(sample_scaled)

print(f"Prediction for sample input {sample.tolist()[0]}: {prediction[0]}")
print("✅ Model trained and saved successfully.")
