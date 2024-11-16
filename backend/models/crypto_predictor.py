<<<<<<< Updated upstream
=======
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import requests
import datetime

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

# Fetch Historical Data
def fetch_crypto_data(coin: str, vs_currency: str = 'usd', days: int = 90):
    """
    Fetch historical price data for a specific cryptocurrency.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {'vs_currency': vs_currency, 'days': days}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        prices = response.json()['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    else:
        raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")

# Prepare Data for LSTM
def prepare_data(data: pd.DataFrame, look_back: int = 60):
    """
    Prepare data for LSTM model by creating sequences.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['price'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# Train the Model
def train_model(coin: str, days: int = 90, epochs: int = 10, batch_size: int = 32):
    """
    Fetch data, prepare it, and train an LSTM model for prediction.
    """
    data = fetch_crypto_data(coin, days=days)
    X, y, scaler = prepare_data(data)
    X = X.unsqueeze(-1)  # Add a channel dimension for LSTM input

    train_size = int(len(X) * 0.8)
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(train_X), batch_size):
            batch_X = train_X[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model, scaler, data

# Make Predictions
def predict_prices(model, scaler, data: pd.DataFrame, future_days: int = 7):
    """
    Predict future prices using the trained model.
    """
    look_back = 60
    last_sequence = data['price'][-look_back:].values.reshape(-1, 1)
    scaled_sequence = scaler.transform(last_sequence)
    input_seq = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)  # Corrected to 3D

    model.eval()
    predictions = []
    with torch.no_grad():
        for _ in range(future_days):
            # Pass the 3D input (batch_size, sequence_length, input_size) to the LSTM
            pred = model(input_seq).item()
            predictions.append(pred)

            # Update the input sequence
            new_input = torch.tensor([[pred]], dtype=torch.float32)
            input_seq = torch.cat((input_seq[:, 1:, :], new_input.unsqueeze(0)), dim=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = [data['timestamp'].iloc[-1] + datetime.timedelta(days=i) for i in range(1, future_days + 1)]
    
    return pd.DataFrame({'date': future_dates, 'predicted_price': predicted_prices.flatten()})
>>>>>>> Stashed changes
