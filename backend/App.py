from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import jwt
from datetime import datetime, timedelta
import random
from pydantic import BaseModel

class UserLogin(BaseModel):
    email: str
    password: str

class UserRegister(BaseModel):
    email: str
    password: str
    name: str
    phone: str

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
users = {}
crypto_prices = {
    "bitcoin": 45000,
    "ethereum": 2800,
    "dogecoin": 0.15,
    "solana": 95,
    "cardano": 0.50
}

# JWT settings
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

@app.get("/")
async def root():
    return {"message": "HERMES API is running"}

@app.post("/api/users/register")
async def register(user: UserRegister):
    if user.email in users:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    users[user.email] = {
        "password": user.password,
        "name": user.name,
        "phone": user.phone,
        "balance": 10000,
        "crypto": {}
    }
    token = jwt.encode({"email": user.email}, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": token}

@app.post("/api/users/login")
async def login(user: UserLogin):
    if user.email not in users or users[user.email]["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = jwt.encode({"email": user.email}, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": token}

@app.get("/api/users/account")
async def get_account(request: Request):
    try:
        token = request.headers.get("x-auth-token")
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload["email"]
        
        if email not in users:
            raise HTTPException(status_code=404, detail="User not found")
            
        user_data = users[email].copy()
        user_data.pop("password", None)  # Remove password from response
        user_data["email"] = email  # Add email to response
        return user_data
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/api/crypto/price/{crypto}")
async def get_price(crypto: str):
    if crypto not in crypto_prices:
        raise HTTPException(status_code=404, detail="Cryptocurrency not found")
    variation = random.uniform(-0.05, 0.05)
    price = crypto_prices[crypto] * (1 + variation)
    return {"price": round(price, 2)}

@app.get("/api/crypto/risk_score/{crypto}")
async def get_risk_score(crypto: str):
    if crypto not in crypto_prices:
        raise HTTPException(status_code=404, detail="Cryptocurrency not found")
    risk_score = random.uniform(1, 10)
    return {"risk_score": round(risk_score, 1)}

@app.get("/api/crypto/historical/{crypto}")
async def get_historical(crypto: str, timeframe: str = "1y"):
    if crypto not in crypto_prices:
        raise HTTPException(status_code=404, detail="Cryptocurrency not found")
    
    base_price = crypto_prices[crypto]
    data = []
    days = 365 if timeframe == "1y" else 180 if timeframe == "6m" else 90 if timeframe == "3m" else 30
    
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i)
        variation = random.uniform(-0.2, 0.2)
        price = base_price * (1 + variation)
        data.append({
            "timestamp": date.isoformat(),
            "price": round(price, 2)
        })
    
    return data

@app.post("/api/users/buy")
async def buy_crypto(request: Request):
    try:
        token = request.headers.get("x-auth-token")
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")

        form = await request.form()
        try:
            amount = float(form.get("amount", "0"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid amount format")
            
        selectedCrypto = form.get("selectedCrypto")
        if not selectedCrypto:
            raise HTTPException(status_code=400, detail="No cryptocurrency selected")

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload["email"]
        
        if email not in users:
            raise HTTPException(status_code=404, detail="User not found")
            
        user = users[email]
        
        if amount <= 0:
            raise HTTPException(status_code=400, detail="Amount must be greater than 0")
        
        if selectedCrypto not in crypto_prices:
            raise HTTPException(status_code=404, detail="Cryptocurrency not found")
            
        current_price = crypto_prices[selectedCrypto]
        total_cost = amount * current_price
        
        if total_cost > user["balance"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient funds. Required: ${total_cost:.2f}, Available: ${user['balance']:.2f}"
            )
        
        user["balance"] -= total_cost
        user["crypto"][selectedCrypto] = user["crypto"].get(selectedCrypto, 0) + amount
        
        return f"Successfully bought {amount} {selectedCrypto} for ${total_cost:.2f}"
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/users/sell")
async def sell_crypto(request: Request):
    try:
        token = request.headers.get("x-auth-token")
        if not token:
            raise HTTPException(status_code=401, detail="No token provided")

        data = await request.form()
        amount = float(data.get("amount"))
        selectedCrypto = data.get("selectedCrypto")

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload["email"]
        
        if email not in users:
            raise HTTPException(status_code=404, detail="User not found")
            
        user = users[email]
        
        if amount <= 0:
            raise HTTPException(status_code=400, detail="Invalid amount")
        
        if selectedCrypto not in crypto_prices:
            raise HTTPException(status_code=404, detail="Cryptocurrency not found")
            
        if selectedCrypto not in user["crypto"] or user["crypto"][selectedCrypto] < amount:
            raise HTTPException(status_code=400, detail="Insufficient crypto balance")
        
        current_price = crypto_prices[selectedCrypto]
        total_value = amount * current_price
        
        user["balance"] += total_value
        user["crypto"][selectedCrypto] -= amount
        
        if user["crypto"][selectedCrypto] == 0:
            del user["crypto"][selectedCrypto]
        
        return f"Successfully sold {amount} {selectedCrypto}"
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid amount format")
