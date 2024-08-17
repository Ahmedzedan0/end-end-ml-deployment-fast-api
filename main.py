"""
Script Name: main.py
Purpose: FastAPI application for predicting census income based on demographic data.
Author: Zidane
Date: 17-08-2024
"""
import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted, NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to sys.path for module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required functions from the custom modules
from starter.ml.data import process_data
from starter.ml.model import ModelTrainer

app = FastAPI()

class CensusData(BaseModel):
    """
    Pydantic model for representing census data used in predictions.
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
        json_schema_extra = {
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
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), "model/random_forest_model.joblib"))
    encoder = joblib.load(os.path.join(os.path.dirname(__file__), "model/encoder.joblib"))
    lb = joblib.load(os.path.join(os.path.dirname(__file__), "model/lb.joblib"))
    logger.info("Model, encoder, and label binarizer loaded successfully.")
    
    # Check if the model is fitted
    try:
        check_is_fitted(model)
        logger.info("Model is fitted and ready for predictions.")
    except NotFittedError:
        logger.error("The model is not fitted. Ensure the model is properly trained and saved.")
        raise HTTPException(status_code=500, detail="Loaded model is not fitted.")

except Exception as e:
    logger.error(f"Failed to load model or preprocessing files: {e}")
    raise HTTPException(status_code=500, detail="Model or preprocessing file loading failed.")

@app.get("/")
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Census Income Prediction API"}

# Instantiate the ModelTrainer class
model_trainer = ModelTrainer()

@app.post("/predict/")
def predict(data: List[CensusData]):
    try:
        input_data = [d.dict(by_alias=True) for d in data]
        df = pd.DataFrame(input_data)

        logger.info(f"Input data received for prediction: {df.head()}")

        cat_features = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ]
        X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
        
        logger.info(f"Processed data: {X}")

        if X is None or X.shape[0] == 0:
            logger.error("Processed data is empty or invalid.")
            raise HTTPException(status_code=400, detail="Processed data is empty or invalid.")

        # Use the predict method from ModelTrainer
        preds = model_trainer.predict(X)
        
        logger.info(f"Raw predictions: {preds}")

        if preds is None or preds.size == 0:
            logger.error("No predictions were made.")
            raise HTTPException(status_code=500, detail="Prediction failed, no output produced.")

        predictions = lb.inverse_transform(preds)

        logger.info(f"Prediction successful: {predictions.tolist()}")

        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
