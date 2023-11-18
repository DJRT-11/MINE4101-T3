from typing import List

import pandas as pd

from fastapi import FastAPI

from data_model import DataModel
from prediction_model import BaselineModel, PredictionModel


app = FastAPI()


@app.get("/")
def read_root():
    return { "message": "Hello world" }

@app.post("/1.0/predict")
def make_predictions(X: List[DataModel]):
    print(X)
    df = pd.DataFrame([x.dict() for x in X])
    prediction_model = BaselineModel()

    results = prediction_model.make_predictions(df)
    probs = prediction_model.get_probability(df)*100

    print(type(results))
    print(type(probs))

    return "Prediction: "+str(results)+"; probability: "+str(probs)

@app.post("/1.0/explain")
def explain(X: List[DataModel]):
    print(X)
    df = pd.DataFrame([x.dict() for x in X])
    prediction_model = BaselineModel()

    coefs = prediction_model.get_coefs(df)
    print(type(coefs))
    return coefs.tolist()

@app.post("/2.0/predict")
def make_predictions(X: List[DataModel]):
    print(X)
    df = pd.DataFrame([x.dict() for x in X])
    prediction_model = PredictionModel()

    results = prediction_model.make_predictions(df)
    probs = prediction_model.get_probability(df)*100

    print(type(results))
    print(type(probs))

    return "Prediction: "+str(results)+"; probability: "+str(probs)

@app.post("/2.0/explain")
def make_predictions(X: List[DataModel]):
    print(X)
    df = pd.DataFrame([x.dict() for x in X])
    prediction_model = PredictionModel()

    coefs = prediction_model.get_coefs(df)
    print(type(coefs))
    return coefs.tolist()