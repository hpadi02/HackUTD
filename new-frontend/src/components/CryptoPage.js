import React, { useState, useEffect } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";

const CryptoPage = () => {
  const [cryptoList, setCryptoList] = useState([
    "bitcoin",
    "ethereum",
    "dogecoin",
    "solana",
    "cardano",
  ]); // List of available cryptocurrencies
  const [selectedCrypto, setSelectedCrypto] = useState("bitcoin");
  const [realTimePrice, setRealTimePrice] = useState(null);
  const [historicalData, setHistoricalData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [buyAmount, setBuyAmount] = useState("");
  const [sellAmount, setSellAmount] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    setLoading(true);
    setErrorMessage("");
    fetchRealTimePrice();
    fetchHistoricalData();
    fetchPredictions();
  }, [selectedCrypto]);

  const fetchRealTimePrice = async () => {
    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/crypto/price/${selectedCrypto}`,
        { timeout: 10000 } // 10-second timeout
      );
      setRealTimePrice(response.data.price);
    } catch (error) {
      console.error("Error fetching real-time price:", error);
      setErrorMessage("Failed to fetch real-time price. Please try again.");
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/crypto/historical/${selectedCrypto}?days=90`,
        { timeout: 10000 } // 10-second timeout
      );
      setHistoricalData(response.data);
    } catch (error) {
      console.error("Error fetching historical data:", error);
      setErrorMessage("Failed to fetch historical data. Please try again.");
    }
  };

  const fetchPredictions = async () => {
    try {
      const response = await axios.post(
        `http://127.0.0.1:8000/crypto/predict`,
        {
          coin: selectedCrypto,
          days: 90,
          future_days: 7,
        },
        { timeout: 10000 } // 10-second timeout
      );
      setPredictionData(response.data);
    } catch (error) {
      console.error("Error fetching prediction data:", error);
      setErrorMessage("Failed to fetch prediction data. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleBuy = () => {
    if (buyAmount > 0) {
      alert(`You have bought ${buyAmount} of ${selectedCrypto}!`);
    } else {
      alert("Please enter a valid amount to buy.");
    }
  };

  const handleSell = () => {
    if (sellAmount > 0) {
      alert(`You have sold ${sellAmount} of ${selectedCrypto}!`);
    } else {
      alert("Please enter a valid amount to sell.");
    }
  };

  const renderHistoricalChart = () => {
    if (!historicalData) return null;

    const data = {
      labels: historicalData.map((entry) =>
        new Date(entry.timestamp).toLocaleDateString()
      ),
      datasets: [
        {
          label: `${selectedCrypto} Price (Last 90 Days)`,
          data: historicalData.map((entry) => entry.price),
          fill: false,
          borderColor: "blue",
          tension: 0.1,
        },
      ],
    };

    return <Line data={data} />;
  };

  const renderPredictionChart = () => {
    if (!predictionData) return null;

    const data = {
      labels: predictionData.map((entry) =>
        new Date(entry.date).toLocaleDateString()
      ),
      datasets: [
        {
          label: `${selectedCrypto} Predicted Prices (Next 7 Days)`,
          data: predictionData.map((entry) => entry.predicted_price),
          fill: false,
          borderColor: "green",
          tension: 0.1,
        },
      ],
    };

    return <Line data={data} />;
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Crypto Dashboard</h1>
      {errorMessage && <p style={{ color: "red" }}>{errorMessage}</p>}
      <div>
        <label>Select a cryptocurrency: </label>
        <select
          value={selectedCrypto}
          onChange={(e) => setSelectedCrypto(e.target.value)}
        >
          {cryptoList.map((crypto) => (
            <option key={crypto} value={crypto}>
              {crypto}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginTop: "20px" }}>
        <h2>Real-Time Price</h2>
        {realTimePrice ? (
          <p>
            {selectedCrypto} is currently priced at ${realTimePrice} USD.
          </p>
        ) : (
          <p>Loading real-time price...</p>
        )}
      </div>

      <div style={{ marginTop: "20px" }}>
        <h2>Historical Data</h2>
        {renderHistoricalChart() || <p>Loading historical data...</p>}
      </div>

      <div style={{ marginTop: "20px" }}>
        <h2>Predicted Prices</h2>
        {renderPredictionChart() || <p>Loading predictions...</p>}
      </div>

      <div style={{ marginTop: "20px" }}>
        <h2>Buy/Sell {selectedCrypto}</h2>
        <div>
          <label>Buy Amount: </label>
          <input
            type="number"
            value={buyAmount}
            onChange={(e) => setBuyAmount(e.target.value)}
          />
          <button onClick={handleBuy}>Buy</button>
        </div>
        <div style={{ marginTop: "10px" }}>
          <label>Sell Amount: </label>
          <input
            type="number"
            value={sellAmount}
            onChange={(e) => setSellAmount(e.target.value)}
          />
          <button onClick={handleSell}>Sell</button>
        </div>
      </div>
    </div>
  );
};

export default CryptoPage;
