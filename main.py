from typing import List

import pandas as pd
import numpy as np

from fastapi import FastAPI

from data_model import DataModel
from prediction_model import BaselineModel, PredictionModel
from training.functions.preprocessing_final import preprocessing_json


app = FastAPI()


@app.get("/")
def read_root():
    return { "message": "Hello world" }

@app.post("/1.0/predict")
def make_predictions(X: List[DataModel]):
    df = preprocessing_json(X)
    prediction_model = BaselineModel()

    preds = []
    probs = []

    for i, r in df.iterrows():
        row_df = pd.DataFrame([r], columns=df.columns)

        pred = (prediction_model.make_predictions(row_df)).item()
        prob = (prediction_model.get_probability(row_df)*100).item()

        preds.append(pred)
        probs.append(prob)

    out_df = pd.DataFrame({'Prediction': preds, 'Probability': probs})    
    out_json = out_df.to_json(orient='records')
    return out_json

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
    df = preprocessing_json(X)
    prediction_model = PredictionModel()

    preds = []
    probs = []

    for i, r in df.iterrows():
        row_df = pd.DataFrame([r], columns=df.columns)

        pred = (prediction_model.make_predictions(row_df)).item()
        prob = (prediction_model.get_probability(row_df)*100).item()

        preds.append(pred)
        probs.append(prob)

    out_df = pd.DataFrame({'Prediction': preds, 'Probability': probs})    
    out_json = out_df.to_json(orient='records')
    return out_json

@app.post("/2.0/explain")
def make_predictions(X: List[DataModel]):
    print(X)
    df = pd.DataFrame([x.dict() for x in X])
    prediction_model = PredictionModel()

    coefs = prediction_model.get_coefs(df)
    print(type(coefs))
    return coefs.tolist()