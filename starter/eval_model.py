import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics

# Define paths for data and model saving
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "census.csv"
MODEL_PATH = PROJECT_ROOT / "model"
OUTPUT_FILE = PROJECT_ROOT / "slice_output.txt"

model = joblib.load(MODEL_PATH / "rfc_model.pkl")
encoder = joblib.load(MODEL_PATH / "encoder.pkl")
lb = joblib.load(MODEL_PATH / "lb.pkl")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"]

# Add code to load in the data.
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

for column in data.select_dtypes(include="object").columns:
    data[column] = data[column].str.strip()


def evaluate_model(data, feature_name, feature_value):

    _, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"])

    test_sliced = test[test[feature_name] == feature_value]

    # Process the input data using the same encoder and label binarizer used
    # during training
    X_test, y_test, _, _ = process_data(
        test_sliced, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    preds = inference(model, X_test)

    precision, recall, fbeta, accuracy = compute_model_metrics(y_test, preds)

    output = (
        f"{'-'*150}\n"
        f"Model evaluation for {feature_name} = {feature_value}:\n"
        f"Precision: {precision:.3f}\n"
        f"Recall: {recall:.3f}\n"
        f"F-beta: {fbeta:.3f}\n"
        f"Accuracy: {accuracy:.3f}\n\n"
        )
    
    with open(OUTPUT_FILE, "a") as f:
        f.write(output)


if __name__ == "__main__":

    OUTPUT_FILE.write_text("")

    evaluate_model(data, "sex", "Female")
    evaluate_model(data, "sex", "Male")

    evaluate_model(data, "race", "White")
    evaluate_model(data, "race", "Black")
