from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API"}

def test_predict_positive():
    response = client.post("/predict/", json=[{
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
    }])
    assert response.status_code == 200
    assert response.json() == {"predictions": [">50K"]}

def test_predict_negative():
    response = client.post("/predict/", json=[{
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
        "native-country": "United-States"
    }])
    assert response.status_code == 200
    assert response.json() == {"predictions": ["<=50K"]}
