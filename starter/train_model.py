"""
Script Name: train_model.py
Author: Zidane
Date: 08-08-2024
"""
import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
sys.path.append(os.path.join(current_dir, '..'))

if __name__=="__main__":
    from starter.ml.data import process_data
    from starter.ml.model import train_model, inference, compute_model_metrics

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


# Train the model
model = train_model(X_train, y_train)

# Save the model, encoder, and label binarizer
model_dir = os.path.abspath(os.path.join(current_dir, "../model"))
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "random_forest_model.joblib")
encoder_path = os.path.join(model_dir, "encoder.joblib")
lb_path = os.path.join(model_dir, "lb.joblib")

try:
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
except Exception as e:
    print(f"Failed to save model: {e}")

try:
    joblib.dump(encoder, encoder_path)
    print(f"Encoder saved to {encoder_path}")
except Exception as e:
    print(f"Failed to save encoder: {e}")

try:
    joblib.dump(lb, lb_path)
    print(f"Label binarizer saved to {lb_path}")
except Exception as e:
    print(f"Failed to save label binarizer: {e}")


# Evaluate the model
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")
