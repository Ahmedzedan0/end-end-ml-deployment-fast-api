"""
Script Name: train_model.py
Purpose: This script is used to train a machine learning model using the census dataset,
save the trained model along with the encoder and label binarizer, evaluate the model's performance, 
and compute slice performance metrics.
Author: Zidane
Date: 17-08-2024
"""

import os
import sys
import logging
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.model import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(current_dir, '..'))

        from ml.data import process_data

        logger.info("Loading dataset...")
        data_path = os.path.join(current_dir, "data/census.csv")
        data = pd.read_csv(data_path)

        logger.info(f"Dataset loaded with shape: {data.shape}")

        # Define categorical features
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

        data.columns = data.columns.str.strip()

        logger.info("Splitting data into train and test sets...")
        train, test = train_test_split(data, test_size=0.20)
        logger.info(f"Training set shape: {train.shape}, Test set shape: {test.shape}")

        logger.info("Processing training data...")
        X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )
        logger.info(f"Processed training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")

        logger.info("Processing test data...")
        X_test, y_test, _, _ = process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        logger.info(f"Processed test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

        logger.info("Initializing and training the model...")
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        logger.info("Model training completed.")

        model_dir = os.path.abspath(os.path.join(current_dir, "../model"))
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "random_forest_model.pkl")
        encoder_path = os.path.join(model_dir, "encoder.pkl")
        lb_path = os.path.join(model_dir, "lb.pkl")

        logger.info(f"Saving model to {model_path}")
        with open(model_path, 'wb') as model_file:
            pickle.dump(trainer.model, model_file)

        logger.info(f"Saving encoder to {encoder_path}")
        with open(encoder_path, 'wb') as encoder_file:
            pickle.dump(encoder, encoder_file)

        logger.info(f"Saving label binarizer to {lb_path}")
        with open(lb_path, 'wb') as lb_file:
            pickle.dump(lb, lb_file)

        logger.info("Model, encoder, and label binarizer saved successfully.")

        # Track the files with DVC
        logger.info("Tracking the saved files with DVC...")
        os.system(f'dvc add {model_path}')
        os.system(f'dvc add {encoder_path}')
        os.system(f'dvc add {lb_path}')
        logger.info("Files tracked with DVC successfully.")

        # Validate the saved model
        try:
            with open(model_path, 'rb') as model_file:
                loaded_model = pickle.load(model_file)
            logger.info("Model loaded successfully from %s for validation.", model_path)
        except Exception as e:
            logger.error("Failed to load the saved model for validation: %s", str(e))
            sys.exit(1)

        # Evaluate the model
        preds = loaded_model.predict(X_test)
        precision, recall, fbeta = trainer.compute_metrics(y_test, preds)
        logger.info(f"Evaluation Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting the training script...")
    main()
    logger.info("Training script completed.")
