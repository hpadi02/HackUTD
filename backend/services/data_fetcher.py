import requests
import pandas as pd
from fastapi import HTTPException

class DataFetcher:
    """
    A class to fetch and manage cryptocurrency data.
    """
    BASE_URL = "https://api.coingecko.com/api/v3"

    @staticmethod
    def fetch_real_time_price(coin: str, vs_currency: str = "usd"):
        """
        Fetch the real-time price of a cryptocurrency.
        """
        url = f"{DataFetcher.BASE_URL}/simple/price"
        params = {"ids": coin, "vs_currencies": vs_currency}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if coin in data:
                return {"coin": coin, "price": data[coin][vs_currency]}
            else:
                raise HTTPException(status_code=404, detail="Cryptocurrency not found.")
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch real-time price: {response.text}",
            )

    @staticmethod
    def fetch_historical_data(coin: str, vs_currency: str = "usd", days: int = 90):
        """
        Fetch historical price data for a cryptocurrency.
        """
        url = f"{DataFetcher.BASE_URL}/coins/{coin}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            prices = response.json().get("prices", [])
            if prices:
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                return df
            else:
                raise HTTPException(
                    status_code=404, detail="No historical data found for the given coin."
                )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch historical data: {response.text}",
            )
