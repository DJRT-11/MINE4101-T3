from joblib import load

class PredictionModel:

    def __init__(self):
        self.model = load("models/churn-baseline-v1.0.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
    
    def get_probability(self, data):
        probs = self.model.predict_proba(data)[:,1]
        return probs