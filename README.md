# HERMES

**HackUTD XI — PNC Track Winner**

A crypto-powered retirement fund platform that uses AI to assess portfolio risk and automate long-term wealth building through blockchain technology.

---

## Overview

HERMES is a smart retirement savings platform that bridges traditional finance and DeFi. A neural network evaluates a user's crypto portfolio risk and suggests rebalancing strategies optimized for long-term growth. Users connect their crypto wallet, receive an AI-generated risk score, and can automate contributions to a retirement fund — no broker required.

---

## Features

- **AI Risk Scoring** — Neural network analyzes portfolio composition, asset volatility, and market conditions to produce a real-time risk score
- **Crypto Wallet Integration** — Connect via Web3 to read wallet balances and portfolio data
- **Live Market Data** — Real-time price and market cap data from the CoinGecko API
- **Retirement Fund Simulation** — Projects long-term growth under different contribution and risk scenarios
- **Interactive Dashboard** — Visualizes portfolio allocation, risk trends, and projections with Chart.js
- **Secure Auth** — JWT-based authentication with bcrypt password hashing

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, React Router, Chart.js, Bootstrap 5 |
| Backend | FastAPI (Python), Express.js (Node) |
| AI/ML | TensorFlow / PyTorch, Scikit-learn, NumPy, Pandas |
| Blockchain | Web3.py, Ethereum RPC |
| Database | MongoDB (Mongoose) |
| Market Data | CoinGecko API |
| Auth | JWT, bcryptjs |

---

## Project Structure

```
HackUTD/
├── backend/          # FastAPI Python backend (ML model + blockchain logic)
├── new-frontend/     # React frontend
├── requirements.txt  # Python dependencies
└── package.json      # Node.js dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- MongoDB (local or Atlas)
- A Web3-compatible Ethereum RPC URL (e.g., Infura, Alchemy)

### 1. Clone the Repository

```bash
git clone https://github.com/hpadi02/HackUTD.git
cd HackUTD
```

### 2. Set Up the Python Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```env
MONGO_URI=your_mongodb_connection_string
RPC_URL=your_ethereum_rpc_url
JWT_SECRET=your_jwt_secret
COINGECKO_API_KEY=your_api_key
```

### 4. Set Up the Frontend

```bash
cd new-frontend
npm install
npm start
```

The app runs at `http://localhost:3000`, with the backend at `http://localhost:8000`.

---

## Neural Network Risk Model

The model ingests portfolio asset weights, 30-day price volatility, market cap classifications, and cross-asset correlations. It outputs a risk score from 1 to 10:

| Score | Tier |
|---|---|
| 1–3 | Conservative |
| 4–6 | Moderate |
| 7–9 | Aggressive |
| 10 | Speculative |

---

## License

This project was built for HackUTD XI. See [LICENSE](LICENSE) for details.
