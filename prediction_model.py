from joblib import load
import shap

class BaselineModel:

    def __init__(self):
        self.model = load("models/churn-baseline-v1.0.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
    
    def get_probability(self, data):
        probs = self.model.predict_proba(data)[:,1]
        return probs
    
    def get_coefs(self, data):     
        coefs = self.model.steps[1][1].coef_
        return coefs
    
class PredictionModel:

    def __init__(self):
        self.model = load("models/churn-v1.0.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
    
    def get_probability(self, data):
        probs = self.model.predict_proba(data)[:,1]
        return probs
    
    def get_coefs(self, data):     
        coefs = self.model.steps[2][1].feature_importances_
        return coefs