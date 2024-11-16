import React, { useState, useEffect } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";

const CryptoPage = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post("http://127.0.0.1:8000/crypto/predict", {
        coin: "bitcoin",
        days: 90,
        future_days: 7,
      });
      setPredictions(response.data);
    } catch (err) {
      setError("Failed to fetch predictions. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions();
  }, []);

  const chartData = {
    labels: predictions.map((p) => new Date(p.date).toLocaleDateString()),
    datasets: [
      {
        label: "Predicted Prices (USD)",
        data: predictions.map((p) => p.predicted_price),
        borderColor: "rgba(75,192,192,1)",
        backgroundColor: "rgba(75,192,192,0.2)",
        borderWidth: 2,
        fill: true,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: true,
        position: "top",
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Date",
        },
      },
      y: {
        title: {
          display: true,
          text: "Price (USD)",
        },
      },
    },
  };

  return (
    <div className="crypto-page">
      <h1>Crypto Price Predictions</h1>
      {loading && <p>Loading predictions...</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && predictions.length > 0 && (
        <Line data={chartData} options={chartOptions} />
      )}
      {!loading && !error && predictions.length === 0 && (
        <p>No data available. Please try again later.</p>
      )}
    </div>
  );
};

export default CryptoPage;
