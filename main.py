"""
Script Name: main.py
Purpose: FastAPI application for predicting census income based on demographic data.
Author: Zidane
Date: 14-08-2024
"""

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import numpy as np

# Add the current directory to sys.path for module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from starter.ml.data import process_data
    from starter.ml.model import inference

app = FastAPI()

class CensusData(BaseModel):
    """
    Pydantic model for representing census data used in predictions.

    Attributes:
        age (int): The age of the individual.
        workclass (str): The workclass of the individual.
        fnlwgt (int): Final weight, a calculated field.
        education (str): The education level of the individual.
        education_num (int): The number of years of education, alias to 'education-num'.
        marital_status (str): The marital status of the individual.
        occupation (str): The occupation of the individual.
        relationship (str): The relationship status of the individual.
        race (str): The race of the individual.
        sex (str): The sex of the individual.
        capital_gain (int): The capital gain of the individual.
        capital_loss (int): The capital loss of the individual.
        hours_per_week (int): The number of hours the individual works per week.
        native_country (str): The native country of the individual.
    """
    age: int
    workclass: str = Field(..., alias="workclass")
    fnlwgt: int
    education: str = Field(..., alias="education")
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(..., alias="occupation")
    relationship: str = Field(..., alias="relationship")
    race: str = Field(..., alias="race")
    sex: str = Field(..., alias="sex")
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        """
        Configuration class for the CensusData model, providing an example schema.
        """
        schema_extra = {
            "example": {
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
            }
        }

# Load the model, encoder, and label binarizer
model = joblib.load(os.path.join(os.path.dirname(__file__), "model/random_forest_model.joblib"))
encoder = joblib.load(os.path.join(os.path.dirname(__file__), "model/encoder.joblib"))
lb = joblib.load(os.path.join(os.path.dirname(__file__), "model/lb.joblib"))

@app.get("/")
def read_root():
    """
    Root endpoint that returns a welcome message.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Census Income Prediction API"}

@app.post("/predict/")
def predict(data: List[CensusData]):
    """
    Prediction endpoint that takes in census data and returns income predictions.

    Args:
        data (List[CensusData]): A list of census data records.

    Returns:
        dict: A dictionary containing the list of predictions.
    """
    # Convert input data to DataFrame
    input_data = [d.dict(by_alias=True) for d in data]
    df = pd.DataFrame(input_data)

    # Process the data
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

    # Make predictions
    preds = inference(model, X)
    predictions = lb.inverse_transform(preds)

    return {"predictions": predictions.tolist()}
