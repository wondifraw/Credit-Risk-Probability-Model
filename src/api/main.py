from fastapi import FastAPI, HTTPException
from .pydantic_models import CreditRiskRequest, CreditRiskResponse
import pickle
import pandas as pd

app = FastAPI()

# Load model from pickle file in the project root
with open('RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict", response_model=CreditRiskResponse)
def predict_risk(request: CreditRiskRequest):
    try:
        # Convert request to DataFrame
        input_df = pd.DataFrame([request.dict()])
        # Predict risk probability
        proba = model.predict(input_df)
        # If model returns probability for both classes, take the positive class
        if hasattr(proba, '__len__') and len(proba) == 1 and hasattr(proba[0], '__len__'):
            risk_probability = float(proba[0][1])
        else:
            risk_probability = float(proba[0])
        return CreditRiskResponse(risk_probability=risk_probability)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
