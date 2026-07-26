import requests

url = "https://income-predictor-app-efcfaabcbddc.herokuapp.com/predict"

input = {
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
    "native-country": "United-States"
}

r = requests.post(url, json=input)

print(f'r.status_code: {r.status_code}')
print(f'response: {r.json()}')
