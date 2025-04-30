from pydantic import BaseModel
import hashlib
import jwt
import os
from datetime import datetime, timedelta

class User(BaseModel):
    name: str
    email: str
    phone: str
    password: str  # This will be hashed before storage
    wallet_address: str = None

# In-memory storage for users (replace with a database in production)
users_db = {}

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_token(user_email: str) -> str:
    # Create JWT token that expires in 24 hours
    expiration = datetime.utcnow() + timedelta(hours=24)
    token = jwt.encode(
        {"email": user_email, "exp": expiration},
        os.getenv("JWT_SECRET", "your-secret-key"),  # Use environment variable in production
        algorithm="HS256"
    )
    return token

def register_user(user_data: dict) -> dict:
    if user_data["email"] in users_db:
        raise ValueError("Email already registered")
    
    # Hash the password before storing
    user_data["password"] = hash_password(user_data["password"])
    users_db[user_data["email"]] = user_data
    
    return {
        "email": user_data["email"],
        "name": user_data["name"],
        "message": "User registered successfully"
    }

def authenticate_user(email: str, password: str) -> dict:
    if email not in users_db:
        raise ValueError("User not found")
    
    user = users_db[email]
    if user["password"] != hash_password(password):
        raise ValueError("Invalid password")
    
    token = create_token(email)
    return {
        "token": token,
        "email": email,
        "name": user["name"]
    } 