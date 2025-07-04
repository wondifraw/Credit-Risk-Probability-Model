from pydantic import BaseModel
from typing import List, Optional

class CreditRiskRequest(BaseModel):
    # Add all model input features here. Example features below:
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    # Add any additional features used by your model

class CreditRiskResponse(BaseModel):
    risk_probability: float
