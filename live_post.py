"""
Script Name: live_post.py
Purpose: Script to send a POST request to the FastAPI endpoints
for training and prediction.
Author: Zidane
Date: 21-08-2024
"""

import requests

# Define the API endpoints
train_url = "https://end-end-ml-deployment-fast-api-1.onrender.com/train/"
predict_url = "https://end-end-ml-deployment-fast-api-1.onrender.com/predict/"

# Define the payload with the field names for prediction
payload = [{
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}]

train_payload = [{
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
    "salary": ">50K",  # Include the label for training
}]

# Uncomment this section if you need to train the model
# response = requests.post(train_url, json=train_payload)
# print(f"Training Status Code: {response.status_code}")
# try:
#     print("Training Response JSON:", response.json())
# except requests.exceptions.JSONDecodeError:
#     print("Training Response Content:", response.text)

# Make the POST request for prediction
response = requests.post(predict_url, json=payload)

# Print the status code
print(f"Prediction Status Code: {response.status_code}")

# Try to print the response JSON, or print the raw response if JSON
# decoding fails
try:
    print("Prediction Response JSON:", response.json())
except requests.exceptions.JSONDecodeError:
    print("Prediction Response Content:", response.text)
