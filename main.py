# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
from pathlib import Path

from starter.ml.data import process_data
from starter.ml.model import inference

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model"

model = joblib.load(MODEL_PATH / "rfc_model.pkl")
encoder = joblib.load(MODEL_PATH / "encoder.pkl")
lb = joblib.load(MODEL_PATH / "lb.pkl")

cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

app = FastAPI(title='Income Predictor API', description='Starter API for ML model deployment', version='1.0.0')

class InputData(BaseModel):

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 37,
                    "workclass": "Private",
                    "fnlgt": 178356,
                    "education": "HS-grad",
                    "education-num": 9,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Craft-repair",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )

class PredictionResponse(BaseModel):
    prediction: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Income Predictor API. Use the /predict endpoint to get income predictions."}


@app.post("/predict", response_model=PredictionResponse)
async def predict_income(input_data: InputData):
    # converting the input data to a DataFrame for processing
    input_dict = input_data.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    # Process the input data using the same encoder and label binarizer used during training
    X, _, _, _ = process_data(input_df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
    prediction = inference(model, X)
    prediction_label = lb.inverse_transform(prediction)[0]

    return {"prediction": str(prediction_label)}
