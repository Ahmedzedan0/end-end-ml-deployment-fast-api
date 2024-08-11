"""
Script Name: train_model.py
Author: Zidane
Date: 08-08-2024
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics


data_path = os.path.join(os.path.dirname(__file__), "../../../c3-starter-code/starter/data/census.csv")
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

# Train the model
model = train_model(X_train, y_train)

# Save the model, encoder, and label binarizer
model_path = os.path.join(os.path.dirname(__file__), "model/random_forest_model.joblib")
encoder_path = os.path.join(os.path.dirname(__file__), "model/encoder.joblib")
lb_path = os.path.join(os.path.dirname(__file__), "model/lb.joblib")
joblib.dump(model, model_path)
joblib.dump(encoder, encoder_path)
joblib.dump(lb, lb_path)

# Evaluate the model
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")
