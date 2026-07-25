import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import inference, compute_model_metrics


model = joblib.load("../model/rfc_model.pkl")
encoder = joblib.load("../model/encoder.pkl")
lb = joblib.load("../model/lb.pkl")

cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
data.columns = data.columns.str.strip()

for column in data.select_dtypes(include="object").columns:
    data[column] = data[column].str.strip()


def evaluate_model(data, feature_name, feature_value):

    _, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data["salary"])

    test_sliced = test[test[feature_name] == feature_value]

    # Process the input data using the same encoder and label binarizer used during training
    X_test, y_test, _, _ = process_data(test_sliced, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    preds = inference(model, X_test)

    precision, recall, fbeta, accuracy = compute_model_metrics(y_test, preds)

    print(f"{'-'*150}")
    print(f"Model evaluation for {feature_name} = {feature_value}:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F-beta: {fbeta:.3f}")
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":

    evaluate_model(data, "sex", "Female")
    evaluate_model(data, "sex", "Male")

    evaluate_model(data, "race", "White")
    evaluate_model(data, "race", "Black")




    