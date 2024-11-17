from models.crypto_predictor import train_model, predict_prices
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Input schema for validation
class CryptoPredictionRequest(BaseModel):
    coin: str
    days: int = 90
    future_days: int = 7

@app.post("/crypto/predict")
def get_crypto_predictions(request: CryptoPredictionRequest):
    model, scaler, data = train_model(request.coin, days=request.days, epochs=5)
    predictions = predict_prices(model, scaler, data, future_days=request.future_days)
    return predictions.to_dict(orient="records")
