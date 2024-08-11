import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np

if __name__ == "__main__":
    from starter.ml.data import process_data
    from starter.ml.model import inference

app = FastAPI()

class CensusData(BaseModel):
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
    return {"message": "Welcome to the Census Income Prediction API"}

@app.post("/predict/")
def predict(data: List[CensusData]):
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
