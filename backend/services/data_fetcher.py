import time
from cachetools import TTLCache
import requests
import pandas as pd
from fastapi import HTTPException


class DataFetcher:
    """
    A class to fetch and manage cryptocurrency data with caching and retry logic.
    """
    BASE_URL = "https://api.coingecko.com/api/v3"
    CACHE = TTLCache(maxsize=100, ttl=300)  # Cache results for 5 minutes

    @staticmethod
    def fetch_real_time_price(coin: str, vs_currency: str = "usd"):
        """
        Fetch the real-time price of a cryptocurrency with caching and retry logic.
        """
        cache_key = f"real_time_price_{coin}_{vs_currency}"
        if cache_key in DataFetcher.CACHE:
            return DataFetcher.CACHE[cache_key]

        url = f"{DataFetcher.BASE_URL}/simple/price"
        params = {"ids": coin, "vs_currencies": vs_currency}

        for attempt in range(5):  # Retry logic
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if coin in data:
                    DataFetcher.CACHE[cache_key] = {"coin": coin, "price": data[coin][vs_currency]}
                    return DataFetcher.CACHE[cache_key]
                else:
                    raise HTTPException(status_code=404, detail="Cryptocurrency not found.")
            elif response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Error fetching real-time price: {response.status_code} - {response.text}")
                time.sleep(5)
        raise HTTPException(
            status_code=500, detail="Failed to fetch real-time price after multiple attempts."
        )

    @staticmethod
    def fetch_historical_data(coin: str, vs_currency: str = "usd", timeframe: str = "1y"):
        """
        Fetch historical price data for a cryptocurrency using caching and retry logic.
        """
        cache_key = f"historical_data_{coin}_{vs_currency}_{timeframe}"
        if cache_key in DataFetcher.CACHE:
            return DataFetcher.CACHE[cache_key]

        url = f"{DataFetcher.BASE_URL}/coins/{coin}/market_chart"
        params = {"vs_currency": vs_currency, "days": timeframe}

        for attempt in range(5):  # Retry logic
            response = requests.get(url, params=params)
            if response.status_code == 200:
                prices = response.json().get("prices", [])
                if prices:
                    df = pd.DataFrame(prices, columns=["timestamp", "price"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    DataFetcher.CACHE[cache_key] = df
                    return df
                else:
                    raise HTTPException(
                        status_code=404, detail="No historical data found for the given coin."
                    )
            elif response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Error fetching historical data: {response.status_code} - {response.text}")
                time.sleep(5)
        raise HTTPException(
            status_code=500, detail="Failed to fetch historical data after multiple attempts."
        )

    @staticmethod
    def fetch_market_data():
        """
        Fetch general market data to analyze trends with retry logic.
        """
        cache_key = "market_data"
        if cache_key in DataFetcher.CACHE:
            return DataFetcher.CACHE[cache_key]

        url = f"{DataFetcher.BASE_URL}/global"

        for attempt in range(5):  # Retry logic
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                DataFetcher.CACHE[cache_key] = data
                return data
            elif response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Error fetching market data: {response.status_code} - {response.text}")
                time.sleep(5)
        raise HTTPException(
            status_code=500, detail="Failed to fetch market data after multiple attempts."
        )

    @staticmethod
    def fetch_supported_coins():
        """
        Fetch a list of supported coins from the API with caching and retry logic.
        """
        cache_key = "supported_coins"
        if cache_key in DataFetcher.CACHE:
            return DataFetcher.CACHE[cache_key]

        url = f"{DataFetcher.BASE_URL}/coins/list"

        for attempt in range(5):  # Retry logic
            response = requests.get(url)
            if response.status_code == 200:
                coins = response.json()
                coin_list = [coin["id"] for coin in coins]
                DataFetcher.CACHE[cache_key] = coin_list
                return coin_list
            elif response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Error fetching supported coins: {response.status_code} - {response.text}")
                time.sleep(5)
        raise HTTPException(
            status_code=500, detail="Failed to fetch supported coins after multiple attempts."
        )
