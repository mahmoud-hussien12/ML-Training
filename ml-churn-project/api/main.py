from fastapi import FastAPI, HTTPException
import mlflow.sklearn
from api.schemas import ChurnRequest, ChurnResponse
from api.feature_schema import EXPECTED_FEATURES
import pandas as pd
from asgi_correlation_id import CorrelationIdMiddleware, correlation_id

from api import logger
app = FastAPI(title="Churn Prediction API")
app.add_middleware(CorrelationIdMiddleware)

MODEL_URI = "runs:/f57ccdd696be465cbc196f94cdeede6e/model"
model = mlflow.sklearn.load_model(MODEL_URI)
logger.log_info("model loaded", data={
    "model_type": type(model).__name__,
    "model_uri": MODEL_URI,
    "model_version": "v1"
})


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    missing_features = EXPECTED_FEATURES - request.features.keys()
    extra_features = request.features.keys() - EXPECTED_FEATURES
    logger.log_info(
        "prediction_request",
        data={
            "features": request.features,
            "missing_features": missing_features,
            "extra_features": extra_features
        }
    )

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {sorted(missing_features)}"
        )

    if extra_features:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected features: {sorted(extra_features)}"
        )
    X = pd.DataFrame([request.features])
    proba = model.predict_proba(X)[0][1]
    logger.log_info(
        "prediction_response",
        data={
            "churn_probability": proba
        }
    )
    return ChurnResponse(churn_probability=float(proba))

@app.get("/health")
def health_check():
    return {"status": "ok"}