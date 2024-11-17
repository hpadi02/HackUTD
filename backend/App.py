from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.models.crypto_predictor import (
    train_model,
    predict_prices_with_explainability,
    load_pretrained_model,
)
from backend.services.blockchain import BlockchainService
from backend.services.data_fetcher import DataFetcher
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Fetch RPC URL from .env file
rpc_url = os.getenv("RPC_URL")
if not rpc_url:
    raise ValueError("RPC URL not found in .env file.")

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust for production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Blockchain Service
blockchain = BlockchainService(rpc_url)

# Input schema for crypto predictions
class CryptoPredictionRequest(BaseModel):
    coin: str
    days: int = 90
    future_days: int = 7


@app.post("/crypto/predict_with_explainability")
def get_predictions_with_explainability(request: CryptoPredictionRequest):
    """
    Endpoint for predicting future cryptocurrency prices with explanations.
    """
    try:
        # Load a pretrained model if training is skipped
        model = load_pretrained_model()
        
        # Train a model if needed
        if not model:
            model, scaler, data = train_model(request.coin, days=request.days, epochs=5)
        else:
            data = DataFetcher.fetch_historical_data(request.coin, "usd", request.days)
            _, scaler, _ = prepare_data(data)

        predictions_with_explanations = predict_prices_with_explainability(
            model, scaler, data, future_days=request.future_days
        )
        return predictions_with_explanations.to_dict(orient="records")
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/wallet/create")
def create_wallet():
    """
    Endpoint for creating a new wallet.
    """
    try:
        return blockchain.create_wallet()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating wallet: {str(e)}")


@app.get("/wallet/balance/{wallet_address}")
def get_wallet_balance(wallet_address: str):
    """
    Endpoint for fetching the balance of a wallet.
    """
    try:
        balance = blockchain.get_wallet_balance(wallet_address)
        return {"balance": balance}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching wallet balance: {str(e)}")


class TransactionRequest(BaseModel):
    private_key: str
    to_address: str
    amount_eth: float


@app.post("/transaction/send")
def send_transaction(request: TransactionRequest):
    """
    Endpoint for sending cryptocurrency from one wallet to another.
    """
    try:
        tx_hash = blockchain.send_transaction(request.private_key, request.to_address, request.amount_eth)
        return {"tx_hash": tx_hash}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending transaction: {str(e)}")


@app.get("/transaction/status/{tx_hash}")
def get_transaction_status(tx_hash: str):
    """
    Endpoint for checking the status of a transaction.
    """
    try:
        status = blockchain.get_transaction_status(tx_hash)
        return {"status": status}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transaction status: {str(e)}")


@app.get("/crypto/price/{coin}")
def get_real_time_price(coin: str, vs_currency: str = "usd"):
    """
    Endpoint for fetching the real-time price of a cryptocurrency.
    """
    try:
        return DataFetcher.fetch_real_time_price(coin, vs_currency)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching real-time price: {str(e)}")


@app.get("/crypto/historical/{coin}")
def get_historical_data(coin: str, vs_currency: str = "usd", days: int = 90):
    """
    Endpoint for fetching historical price data for a cryptocurrency.
    """
    try:
        return DataFetcher.fetch_historical_data(coin, vs_currency, days).to_dict(orient="records")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")
