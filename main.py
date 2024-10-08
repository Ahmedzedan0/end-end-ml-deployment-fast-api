"""
Script Name: main.py
Purpose: FastAPI application for predicting census income based
on demographic data.
Author: Zidane
Date: 21-08-2024
"""

from ml.model import ModelTrainer
from ml.data import process_data
import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pickle
import pandas as pd
import traceback
from sklearn.utils.validation import check_is_fitted, NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to sys.path for module resolution
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "starter"))

app = FastAPI()

# Declare global variables
model = None
encoder = None
lb = None
model_trainer = None

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
                "native-country": "United-States",
            }
        }

@app.on_event("startup")
async def startup_event():
    global model, encoder, lb, model_trainer
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model/random_forest_model.pkl")
        encoder_path = os.path.join(os.path.dirname(__file__), "model/encoder.pkl")
        lb_path = os.path.join(os.path.dirname(__file__), "model/lb.pkl")

        # Check if all necessary files exist
        if not os.path.exists(model_path) or not os.path.exists(encoder_path) or not os.path.exists(lb_path):
            logger.error("Model, encoder, or label binarizer files are missing.")
            raise HTTPException(status_code=500, detail="Necessary files are missing.")

        # Load model
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
            logger.info("Model loaded successfully.")

        # Load encoder and validate it
        with open(encoder_path, "rb") as encoder_file:
            encoder = pickle.load(encoder_file)
            logger.info("Encoder loaded successfully.")
            
            # Test encoder
            try:
                test_data = [['Private']]  # Test with a valid category
                encoder.transform(test_data)
                logger.info("Encoder test passed.")
            except Exception as e:
                logger.error(f"Encoder test failed: {e}")
                raise HTTPException(status_code=500, detail="Encoder test failed.")

        # Load label binarizer
        with open(lb_path, "rb") as lb_file:
            lb = pickle.load(lb_file)
            logger.info("Label binarizer loaded successfully.")

        # Initialize ModelTrainer and validate model
        model_trainer = ModelTrainer()
        model_trainer.model = model
        try:
            check_is_fitted(model_trainer.model)
            logger.info("Model is fitted and ready for predictions.")
        except NotFittedError:
            logger.error("The model is not fitted.")
            raise HTTPException(status_code=500, detail="Loaded model is not fitted.")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise HTTPException(status_code=500, detail="Application startup failed.")

@app.get("/")
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Census Income Prediction API"}

@app.post("/train/")
def train_model(data: List[CensusData]) -> dict:
    try:
        input_data = [d.dict(by_alias=True) for d in data]
        df = pd.DataFrame(input_data)

        logger.info(f"Input data received for training: {df.head()}")

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        label = "salary"

        # Verify that the label column exists and has data
        if label not in df.columns:
            logger.error(f"The label column '{label}' is not found in the DataFrame.")
            raise ValueError(f"Label column '{label}' is missing from the input data.")

        logger.info(f"Label column '{label}' data: {df[label].head()}")

        X_train, y_train, _, _ = process_data(
            df,
            categorical_features=cat_features,
            label=label,
            training=True,
            encoder=None,
            lb=None,
        )

        # Train the model
        model_trainer.train(X_train, y_train)

        return {"message": "Model trained successfully."}
    except Exception:
        logger.error(f"Error during training: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Training failed.")

@app.post("/predict/")
def predict(data: List[CensusData]) -> dict:
    try:
        if encoder is None:
            logger.error("Encoder is not loaded.")
            raise HTTPException(status_code=500, detail="Model encoder is not available.")

        input_data = [d.dict(by_alias=True) for d in data]
        df = pd.DataFrame(input_data)

        logger.info(f"Input data received for prediction: {df.head()}")

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        X, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        logger.info(f"Processed data: {X}")

        if X is None or X.shape[0] == 0:
            logger.error("Processed data is empty or invalid.")
            raise HTTPException(status_code=400, detail="Processed data is empty or invalid.")

        preds = model_trainer.predict(X)

        logger.info(f"Raw predictions: {preds}")

        if preds is None or preds.size == 0:
            logger.error("No predictions were made.")
            raise HTTPException(status_code=500, detail="Prediction failed, no output produced.")

        predictions = lb.inverse_transform(preds)

        logger.info(f"Prediction successful: {predictions.tolist()}")

        return {"predictions": predictions.tolist()}
    except Exception:
        logger.error(f"Error during prediction: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Prediction failed.")
