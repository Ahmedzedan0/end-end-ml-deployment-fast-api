"""
Script Name: starter/train_model.py
Purpose: This script is used to train a machine learning model using the census dataset,
save the trained model along with the encoder and label binarizer, and evaluate the model's performance.
Author: Zidane
Date: 17-08-2024
"""

import os
import sys
import logging
import pandas as pd
import joblib
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
        data_path = os.path.join(current_dir, "../data/census.csv")
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

        # Remove leading and trailing spaces from column names
        data.columns = data.columns.str.strip()

        # Split data into train and test sets
        logger.info("Splitting data into train and test sets...")
        train, test = train_test_split(data, test_size=0.20)
        logger.info(f"Training set shape: {train.shape}, Test set shape: {test.shape}")

        # Process the training data
        logger.info("Processing training data...")
        X_train, y_train, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )
        logger.info(f"Processed training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")

        # Process the test data
        logger.info("Processing test data...")
        X_test, y_test, _, _ = process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        logger.info(f"Processed test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

        # Initialize and train the model
        logger.info("Initializing and training the model...")
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        logger.info("Model training completed.")

        # Save the model, encoder, and label binarizer
        model_dir = os.path.abspath(os.path.join(current_dir, "../model"))
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "random_forest_model.joblib")
        encoder_path = os.path.join(model_dir, "encoder.joblib")
        lb_path = os.path.join(model_dir, "lb.joblib")

        logger.info(f"Saving model to {model_path}")
        joblib.dump(trainer.model, model_path)
        logger.info(f"Saving encoder to {encoder_path}")
        joblib.dump(encoder, encoder_path)
        logger.info(f"Saving label binarizer to {lb_path}")
        joblib.dump(lb, lb_path)

        logger.info("Model, encoder, and label binarizer saved successfully.")

        # Evaluate the model
        logger.info("Evaluating the model on test data...")
        preds = trainer.predict(X_test)
        precision, recall, fbeta = trainer.compute_metrics(y_test, preds)
        logger.info(f"Evaluation Results - Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")

        # Compute performance on slices of data
        logger.info("Computing slice performance metrics...")
        trainer.compute_slice_performance(test, y_test, "education", encoder, lb, output_file="slice_output.txt")
        logger.info("Slice performance metrics computed and saved to slice_output.txt.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    # Save the model, encoder, and label binarizer
    logger.info("Saving model to %s", model_path)
    joblib.dump(trainer.model, model_path)
    logger.info("Saving encoder to %s", encoder_path)
    joblib.dump(encoder, encoder_path)
    logger.info("Saving label binarizer to %s", lb_path)
    joblib.dump(lb, lb_path)

    logger.info("Model, encoder, and label binarizer saved successfully.")

    # Validate the saved model
    try:
        loaded_model = joblib.load(model_path)
        logger.info("Model loaded successfully from %s for validation.", model_path)
    except Exception as e:
        logger.error("Failed to load the saved model for validation: %s", str(e))
        sys.exit(1)

    # Evaluate the model
    logger.info("Evaluating the model on test data...")
    preds = trainer.predict(X_test)
    precision, recall, fbeta = trainer.compute_metrics(y_test, preds)
    logger.info(f"Evaluation Results - Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")


if __name__ == "__main__":
    logger.info("Starting the training script...")
    main()
    logger.info("Training script completed.")
