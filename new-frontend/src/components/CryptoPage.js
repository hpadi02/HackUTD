import React, { useState, useEffect } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  TimeScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import "chartjs-adapter-date-fns";

// Register Chart.js components and plugins
ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  TimeScale,
  Title,
  Tooltip,
  Legend,
  zoomPlugin
);

const CryptoPage = () => {
  const [cryptoList] = useState(["bitcoin", "ethereum", "dogecoin", "solana", "cardano"]);
  const [selectedCrypto, setSelectedCrypto] = useState("bitcoin");
  const [realTimePrice, setRealTimePrice] = useState(null);
  const [historicalData, setHistoricalData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [historicalScale, setHistoricalScale] = useState("1y");
  const [buyAmount, setBuyAmount] = useState("");
  const [sellAmount, setSellAmount] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const fetchRealTimePrice = async () => {
    try {
      const response = await axios.get(`http://127.0.0.1:8000/crypto/price/${selectedCrypto}`);
      setRealTimePrice(response.data.price);
    } catch (error) {
      console.error("Error fetching real-time price:", error);
      setErrorMessage("Failed to fetch real-time price.");
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/crypto/historical/${selectedCrypto}?timeframe=${historicalScale}`
      );
      setHistoricalData(response.data);
    } catch (error) {
      console.error("Error fetching historical data:", error);
      setErrorMessage("Failed to fetch historical data.");
    }
  };

  const fetchPredictionData = async () => {
    try {
      const response = await axios.post(`http://127.0.0.1:8000/crypto/predict_with_explainability`, {
        coin: selectedCrypto,
        future_days: 365,
      });
      setPredictionData(response.data);
    } catch (error) {
      console.error("Error fetching prediction data:", error);
      setErrorMessage("Failed to fetch prediction data.");
    }
  };

  useEffect(() => {
    fetchRealTimePrice();
    fetchHistoricalData();
    fetchPredictionData();
  }, [selectedCrypto, historicalScale]);

  const handleBuy = () => alert(`You bought ${buyAmount} of ${selectedCrypto}!`);
  const handleSell = () => alert(`You sold ${sellAmount} of ${selectedCrypto}!`);

  const renderHistoricalChart = () => {
    if (!historicalData) return <p>Loading historical data...</p>;

    const data = {
      labels: historicalData.map((entry) => new Date(entry.timestamp).toLocaleDateString()),
      datasets: [
        {
          label: `${selectedCrypto} Historical Prices`,
          data: historicalData.map((entry) => entry.price),
          borderColor: "blue",
        },
      ],
    };

    const options = {
      responsive: true,
      plugins: { zoom: { zoom: { wheel: { enabled: true }, mode: "x" } } },
      scales: { x: { type: "time" } },
    };

    return <Line data={data} options={options} />;
  };

  const renderPredictionChart = () => {
    if (!predictionData) return <p>Loading predictions...</p>;

    const data = {
      labels: predictionData.map((entry) => new Date(entry.date).toLocaleDateString()),
      datasets: [
        {
          label: `${selectedCrypto} Predicted Prices`,
          data: predictionData.map((entry) => entry.predicted_price),
          borderColor: "green",
        },
      ],
    };

    const options = {
      responsive: true,
      plugins: { zoom: { zoom: { wheel: { enabled: true }, mode: "x" } } },
      scales: { x: { type: "time" } },
    };

    return <Line data={data} options={options} />;
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Crypto Dashboard</h1>
      {errorMessage && <p style={{ color: "red" }}>{errorMessage}</p>}
      <div>
        <label>Select a cryptocurrency: </label>
        <select value={selectedCrypto} onChange={(e) => setSelectedCrypto(e.target.value)}>
          {cryptoList.map((crypto) => (
            <option key={crypto} value={crypto}>
              {crypto}
            </option>
          ))}
        </select>
      </div>
      <div>
        <h2>Real-Time Price</h2>
        <p>{realTimePrice ? `$${realTimePrice}` : "Loading real-time price..."}</p>
      </div>
      <div>
        <h2>Historical Data</h2>
        <select value={historicalScale} onChange={(e) => setHistoricalScale(e.target.value)}>
          <option value="1y">1 Year</option>
          <option value="6m">6 Months</option>
          <option value="3m">3 Months</option>
        </select>
        {renderHistoricalChart()}
      </div>
      <div>
        <h2>Predicted Prices</h2>
        {renderPredictionChart()}
      </div>
      <div>
        <h2>Buy/Sell {selectedCrypto}</h2>
        <input type="number" value={buyAmount} onChange={(e) => setBuyAmount(e.target.value)} />
        <button onClick={handleBuy}>Buy</button>
        <input type="number" value={sellAmount} onChange={(e) => setSellAmount(e.target.value)} />
        <button onClick={handleSell}>Sell</button>
      </div>
    </div>
  );
};

export default CryptoPage;
