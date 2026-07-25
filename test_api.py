from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    r = client.get("/")

    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Income Predictor API. Use the /predict endpoint to get income predictions."}


def test_predict_income():
    input_data = {
        "age": 31,
        "workclass": "Self-emp-inc",
        "fnlgt": 117963,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    r = client.post("/predict", json=input_data)
    assert r.status_code == 200
    assert "prediction" in r.json()


def test_predict_income_invalid_data():
    input_data = {
        "age": "invalid_age",
        "workclass": "Self-emp-inc",
        "fnlgt": 117963,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/predict", json=input_data)
    assert r.status_code == 422
