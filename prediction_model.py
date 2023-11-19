from joblib import load
import numpy as np

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
        coefs = self.model.coef_[0,:].tolist()
        names = self.model.feature_names_in_.tolist()
        
        out = {}
        for key in names:
            for value in coefs:
                out[key] = value
                coefs.remove(value)
                break

        out_3 = dict(sorted(out.items(), key=lambda x: abs(x[1]), reverse=True)[:3])

        return out_3
    
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
        coefs = self.model.steps[1][1].feature_importances_.tolist()
        names = self.model.feature_names_in_.tolist()
        
        out = {}
        for key in names:
            for value in coefs:
                out[key] = value
                coefs.remove(value)
                break

        out_3 = dict(sorted(out.items(), key=lambda x: abs(x[1]), reverse=True)[:3])

        return out_3