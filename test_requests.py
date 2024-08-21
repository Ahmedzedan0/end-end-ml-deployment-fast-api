import requests

base_url = "https://end-end-ml-deployment-fast-api.onrender.com"

def test_root():
    response = requests.get(f"{base_url}/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API"}

def test_predict_positive():
    data = [{
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlwgt": 287927,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }]
    response = requests.post(f"{base_url}/predict/", json=data)
    assert response.status_code == 200
    
    # Strip the leading/trailing whitespace from the predictions before assertion
    predictions = [pred.strip() for pred in response.json()['predictions']]
    assert predictions == [">50K"]

def test_predict_negative():
    data = [{
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
    response = requests.post(f"{base_url}/predict/", json=data)
    assert response.status_code == 200
    
    # Strip the leading/trailing whitespace from the predictions before assertion
    predictions = [pred.strip() for pred in response.json()['predictions']]
    assert predictions == ["<=50K"]

def test_predict_invalid():
    data = {}
    response = requests.post(f"{base_url}/predict/", json=data)
    assert response.status_code == 422
