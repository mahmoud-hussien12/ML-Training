from fastapi import FastAPI
import mlflow.sklearn
from api.schemas import ChurnRequest, ChurnResponse

app = FastAPI(title="Churn Prediction API")

MODEL_URI = "runs:/f57ccdd696be465cbc196f94cdeede6e/model"

model = mlflow.sklearn.load_model(MODEL_URI)

@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    import pandas as pd
    X = pd.DataFrame([request.features])
    proba = model.predict_proba(X)[0][1]

    return ChurnResponse(churn_probability=float(proba))
