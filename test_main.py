"""
Script Name: test_main.py
Purpose:
Author: Zidane
Date: 21-08-2024
"""

from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch


client = TestClient(app)


def test_read_root():
    """
    Test the root endpoint to ensure it returns the welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Prediction API"
    }


def test_predict_positive():
    """
    Test an example where the predicted income is greater than 50K.
    """
    response = client.post(
        "/predict/",
        json=[
            {
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
            }
        ],
    )
    assert response.status_code == 200
    assert response.json() == {"predictions": [">50K"]}


def test_predict_positive():
    with patch("main.encoder", create=True) as mock_encoder:
        mock_encoder.return_value = "mocked_encoder"
        response = client.post(
            "/predict/",
            json=[
                {
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
                }
            ],
        )
        assert response.status_code == 200
        assert response.json() == {"predictions": [">50K"]}

def test_predict_negative():
    """
    Test an example where the predicted income is less than or equal to 50K.
    """
    response = client.post(
        "/predict/",
        json=[{
            "age": 50,
            "workclass": "Self-emp-not-inc",
            "fnlwgt": 83311,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 13,
            "native-country": "United-States",
        }],
    )
    assert response.status_code == 200
    assert response.json() == {"predictions": ["<=50K"]}


def test_predict_malformed_payload():
    """
    Test that the endpoint throws the right
    exception when passed a malformed payload.
    """
    # A payload with missing required fields
    malformed_payload = [{
        "age": 50,
        # Missing 'workclass'
        "fnlwgt": 83311,
        "education": "Bachelors",
        "education-num": 13,
        # Missing 'marital-status'
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 13,
        "native-country": "United-States",
    }]

    response = client.post("/predict/", json=malformed_payload)
    # Assuming your API returns a 422 Unprocessable Entity for malformed input
    assert response.status_code == 422
    # Optional: Check for specific error details in the response
    assert "detail" in response.json()
