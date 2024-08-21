"""
Script Name: test_main.py
Purpose: Tests for the Census Income Prediction API.
Author: Zidane
Date: 21-08-2024
"""
from fastapi.testclient import TestClient
import os
import sys
import json

root_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root_dir)

from main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Census Income Prediction API"}


def test_predict_invalid():
    data = {}
    response = client.post("/predict", json=json.dumps(data))
    assert response.status_code == 422
