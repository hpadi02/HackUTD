from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.models.crypto_predictor import train_model, predict_prices
from backend.services.blockchain import BlockchainService


app = FastAPI()

# Initialize Blockchain Service
rpc_url = "https://eth-mainnet.g.alchemy.com/v2/4sLlSsUxxF2ANTGbi0vSEBISVzKDGHkc"
blockchain = BlockchainService(rpc_url)

# Input schema for crypto predictions
class CryptoPredictionRequest(BaseModel):
    coin: str
    days: int = 90
    future_days: int = 7

@app.post("/crypto/predict")
def get_crypto_predictions(request: CryptoPredictionRequest):
    """
    Endpoint for predicting future cryptocurrency prices.
    """
    model, scaler, data = train_model(request.coin, days=request.days, epochs=5)
    predictions = predict_prices(model, scaler, data, future_days=request.future_days)
    return predictions.to_dict(orient="records")

@app.get("/wallet/create")
def create_wallet():
    """
    Endpoint for creating a new wallet.
    """
    return blockchain.create_wallet()

@app.get("/wallet/balance/{wallet_address}")
def get_wallet_balance(wallet_address: str):
    """
    Endpoint for fetching the balance of a wallet.
    :param wallet_address: The wallet address to check the balance for.
    """
    try:
        balance = blockchain.get_wallet_balance(wallet_address)
        return {"balance": balance}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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

@app.get("/transaction/status/{tx_hash}")
def get_transaction_status(tx_hash: str):
    """
    Endpoint for checking the status of a transaction.
    :param tx_hash: The transaction hash to check.
    """
    try:
        status = blockchain.get_transaction_status(tx_hash)
        return {"status": status}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
