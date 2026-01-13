from pydantic import BaseModel
from typing import Dict

class ChurnRequest(BaseModel):
    features: Dict[str, float | str | int]

class ChurnResponse(BaseModel):
    churn_probability: float

class ExplainRequest(BaseModel):
    features: Dict[str, float | str | int]
    prediction: str