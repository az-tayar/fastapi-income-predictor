from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    r = client.get("/")

    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Income Predictor API. Use the /predict endpoint to get income predictions."}


def test_predict_income_1():
    input_data = {
        "age": 49,
        "workclass": "Private",
        "fnlgt": 160187,
        "education": "9th",
        "education-num": 5,
        "marital-status": "Married-spouse-absent",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 16,
        "native-country": "Jamaica"
    }

    # the label for this input data is <=50K
    r = client.post("/predict", json=input_data)
    assert r.status_code == 200
    assert r.json()['prediction'] == '<=50K'


def test_predict_income_2():
    input_data = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
    # the label for this input data is >50K
    r = client.post("/predict", json=input_data)
    assert r.status_code == 200
    assert r.json()['prediction'] == '>50K'
