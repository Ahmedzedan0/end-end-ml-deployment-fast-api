"""
Script Name: train_model.py
Purpose: This script is used to train a machine learning model using the census dataset, save the trained model along with the encoder and label binarizer, and evaluate the model's performance.
Author: Zidane
Date: 08-08-2024
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from model import ModelTrainer

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, '..'))

    from starter.ml.data import process_data

    data_path = os.path.join(current_dir, "../data/census.csv")
    data = pd.read_csv(data_path)

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

    # Split data into train and test sets
    train, test = train_test_split(data, test_size=0.20)

    # Process the training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Initialize and train the model
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)

    # Save the model, encoder, and label binarizer
    model_dir = os.path.abspath(os.path.join(current_dir, "../model"))
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "random_forest_model.joblib")
    encoder_path = os.path.join(model_dir, "encoder.joblib")
    lb_path = os.path.join(model_dir, "lb.joblib")

    joblib.dump(trainer.model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(lb, lb_path)

    # Evaluate the model
    preds = trainer.predict(X_test)
    precision, recall, fbeta = trainer.compute_metrics(y_test, preds)
    print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")

    # Compute performance on slices of data
    trainer.compute_slice_performance(test, y_test, "education", encoder, lb, output_file="slice_output.txt")

if __name__ == "__main__":
    main()
