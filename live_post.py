"""
Script Name: live_post.py
Purpose:
Author:Zidane
Date:18-08-2024
"""
import requests

# Define the correct API endpoint
url = "http://127.0.0.1:8000/predict/"

# Define the payload with correct field names
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
    "native-country": "United-States"
}]

# Make the POST request
response = requests.post(url, json=payload)

# Print the status code
print(f"Status Code: {response.status_code}")

# Try to print the response JSON, or print the raw response if JSON decoding fails
try:
    print("Response JSON:", response.json())
except requests.exceptions.JSONDecodeError:
    print("Response Content:", response.text)
