import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import requests
import datetime
import shap
import time
from torch.utils.data import DataLoader, TensorDataset
from typing import Union


# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Fetch Historical Data with Retry Logic
def fetch_crypto_data(coin: str, vs_currency: str = 'usd', days: Union[int, str] = 90):
    """
    Fetch historical price data for a cryptocurrency. Handles rate limits.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}

    for attempt in range(5):  # Retry up to 5 times
        response = requests.get(url, params=params)
        if response.status_code == 200:
            prices = response.json()['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        elif response.status_code == 429:
            print("Rate limit exceeded. Retrying in 10 seconds...")
            time.sleep(10)
        else:
            print(f"Error fetching data (Attempt {attempt + 1}): {response.status_code} - {response.text}")
            time.sleep(5)
    raise Exception("Failed to fetch data after multiple attempts.")


# Prepare Data for LSTM
def prepare_data(data: pd.DataFrame, look_back: int = 60):
    """
    Prepare data for LSTM by creating sequences.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['price'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler


# Train Model with DataLoader
def train_model(coin: str, days: int = 90, epochs: int = 10, batch_size: int = 32):
    """
    Train an LSTM model for cryptocurrency prediction.
    """
    data = fetch_crypto_data(coin, days=days)
    X, y, scaler = prepare_data(data)
    X = X.unsqueeze(-1)

    train_size = int(len(X) * 0.8)
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)

    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    return model, scaler, data


# Predict Prices with Explainability
def predict_prices_with_explainability(model, scaler, data: pd.DataFrame, future_days: int = 7):
    """
    Predict future prices using the model and provide explanations via SHAP.
    """
    look_back = 60
    last_sequence = data['price'][-look_back:].values.reshape(-1, 1)
    scaled_sequence = scaler.transform(last_sequence)

    input_seq = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)

    model.eval()
    predictions = []
    explanations = []

    explainer = shap.DeepExplainer(model, input_seq)
    print("Initialized SHAP explainer.")

    for _ in range(future_days):
        with torch.no_grad():
            pred = model(input_seq).item()
            predictions.append(pred)

        shap_values = explainer.shap_values(input_seq)
        explanations.append(shap_values[0].tolist())

        new_input = torch.tensor([[pred]], dtype=torch.float32)
        input_seq = torch.cat((input_seq[:, 1:, :], new_input.unsqueeze(0)), dim=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = [data['timestamp'].iloc[-1] + datetime.timedelta(days=i) for i in range(1, future_days + 1)]

    return pd.DataFrame({
        'date': future_dates,
        'predicted_price': predicted_prices.flatten(),
        'explanation': explanations
    })


# Interactive Data Handler for Timeframes
def get_data_for_timeframe(data: pd.DataFrame, timeframe: str = "1y"):
    """
    Filter data based on the given timeframe (e.g., '1y', '6m', '3m').
    """
    now = data['timestamp'].max()
    if timeframe == "1y":
        start = now - pd.DateOffset(years=1)
    elif timeframe == "6m":
        start = now - pd.DateOffset(months=6)
    elif timeframe == "3m":
        start = now - pd.DateOffset(months=3)
    elif timeframe == "1m":
        start = now - pd.DateOffset(months=1)
    else:
        raise ValueError("Unsupported timeframe. Use '1y', '6m', '3m', or '1m'.")
    return data[data['timestamp'] >= start]


# Load Pretrained Model (Optional)
def load_pretrained_model():
    """
    Load a pretrained model if available, otherwise return None.
    """
    try:
        model = torch.load("pretrained_model.pth")
        model.eval()
        print("Loaded pretrained model.")
        return model
    except FileNotFoundError:
        print("No pretrained model found. Training will be required.")
        return None
