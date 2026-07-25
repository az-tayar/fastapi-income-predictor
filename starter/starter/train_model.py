# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
data.columns = data.columns.str.strip()

for column in data.select_dtypes(include="object").columns:
    data[column] = data[column].str.strip()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data["salary"])

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

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
rfc_model = train_model(X_train, y_train)
preds = inference(rfc_model, X_test)
precision, recall, fbeta, accuracy = compute_model_metrics(y_test, preds)

# Print out the model metrics.
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F-beta: {fbeta:.3f}")
print(f"Accuracy: {accuracy:.3f}")

# save the model, encoder, and lb to disk
joblib.dump(rfc_model, "../model/rfc_model.pkl")
joblib.dump(encoder, "../model/encoder.pkl")
joblib.dump(lb, "../model/lb.pkl")

print(f"{'-' * 100}")
print(test.iloc[0])