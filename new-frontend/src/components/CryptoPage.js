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
import Navbar from "./Navbar"; // Import Navbar component

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

const RiskMeter = ({ score }) => {
  const needlePosition = Math.min(Math.max(score, 1), 10) * 18; // Map score to degrees (1-10 scale)
  const getColor = () => {
    if (score > 7) return "red";
    if (score > 4) return "yellow";
    return "green";
  };

  return (
    <div style={{ position: "relative", width: "250px", height: "140px", margin: "0 auto" }}>
      <svg viewBox="0 0 200 100" style={{ width: "100%" }}>
        <path
          d="M10,100 Q100,-50 190,100"
          fill="none"
          stroke="#aaa"
          strokeWidth="10"
        />
        <path
          d="M10,100 Q100,-50 190,100"
          fill="none"
          stroke={getColor()}
          strokeWidth="8"
          strokeDasharray="200"
          style={{ animation: "dash 1s ease-in-out" }}
        />
        <line
          x1="100"
          y1="100"
          x2={100 + 90 * Math.cos((Math.PI * needlePosition) / 180)}
          y2={100 - 90 * Math.sin((Math.PI * needlePosition) / 180)}
          stroke="#000"
          strokeWidth="4"
          style={{ transition: "transform 0.5s ease" }}
        />
      </svg>
      <div style={{ textAlign: "center", marginTop: "10px", color: "#f5c518", fontWeight: "bold" }}>
        Risk Score: {score}
      </div>
      <p style={{ textAlign: "center", color: "#f5c518", marginTop: "10px" }}>
        AI Powered Risk meter that trains on the data of the last year. 1 is secure, 10 is risky. A score of {score} indicates {score > 7 ? "high risk" : score > 4 ? "moderate risk" : "low risk"}.
      </p>
    </div>
  );
};

const CryptoPage = () => {
  const [cryptoList] = useState(["bitcoin", "ethereum", "dogecoin", "solana", "cardano"]);
  const [selectedCrypto, setSelectedCrypto] = useState("bitcoin");
  const [realTimePrice, setRealTimePrice] = useState(null);
  const [riskScore, setRiskScore] = useState(null);
  const [historicalData, setHistoricalData] = useState(null);
  const [historicalScale, setHistoricalScale] = useState("1y");
  const [buyAmount, setBuyAmount] = useState("");
  const [sellAmount, setSellAmount] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState(""); // Added successMessage state
  const [accountInfo, setAccountInfo] = useState(null); // Added accountInfo state

  // Fetch account info
  const fetchAccountInfo = async () => {
    try {
      const token = localStorage.getItem("token");
      const response = await axios.get("/api/users/account", {
        headers: { "x-auth-token": token },
      });
      setAccountInfo(response.data);
    } catch (error) {
      console.error("Error fetching account info:", error);
    }
  };

  // Handle Buy
  const handleBuy = async () => {
    if (!buyAmount || buyAmount <= 0) {
      setErrorMessage("Enter a valid amount to buy.");
      return;
    }
    try {
      const token = localStorage.getItem("token");
      const response = await axios.post(
        "/api/users/buy",
        {
          amount: buyAmount,
          selectedCrypto,
          price: realTimePrice,
        },
        {
          headers: { "x-auth-token": token },
        }
      );
      setSuccessMessage(response.data);
      fetchAccountInfo(); // Refresh account details
    } catch (error) {
      setErrorMessage(error.response?.data || "Error buying cryptocurrency.");
    }
  };

  // Handle Sell
  const handleSell = async () => {
    if (!sellAmount || sellAmount <= 0) {
      setErrorMessage("Enter a valid amount to sell.");
      return;
    }
    try {
      const token = localStorage.getItem("token");
      const response = await axios.post(
        "/api/users/sell",
        {
          amount: sellAmount,
          selectedCrypto,
          price: realTimePrice,
        },
        {
          headers: { "x-auth-token": token },
        }
      );
      setSuccessMessage(response.data);
      fetchAccountInfo(); // Refresh account details
    } catch (error) {
      setErrorMessage(error.response?.data || "Error selling cryptocurrency.");
    }
  };

  const fetchRealTimePrice = async () => {
    try {
      const response = await axios.get(`http://127.0.0.1:8000/crypto/price/${selectedCrypto}`);
      setRealTimePrice(response.data.price);
    } catch (error) {
      console.error("Error fetching real-time price:", error);
      setErrorMessage("Failed to fetch real-time price.");
    }
  };

  const fetchRiskScore = async () => {
    try {
      const response = await axios.get(`http://127.0.0.1:8000/crypto/risk_score/${selectedCrypto}`);
      setRiskScore(response.data.risk_score);
    } catch (error) {
      console.error("Error fetching risk score:", error);
      setErrorMessage("Failed to fetch risk score.");
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/crypto/historical/${selectedCrypto}?timeframe=${historicalScale}`
      );
      const filteredData = response.data.filter((entry) => {
        const now = new Date();
        const timestamp = new Date(entry.timestamp);
        switch (historicalScale) {
          case "1y":
            return timestamp >= new Date(now.setFullYear(now.getFullYear() - 1));
          case "6m":
            return timestamp >= new Date(now.setMonth(now.getMonth() - 6));
          case "3m":
            return timestamp >= new Date(now.setMonth(now.getMonth() - 3));
          case "1m":
            return timestamp >= new Date(now.setMonth(now.getMonth() - 1));
          default:
            return true;
        }
      });
      setHistoricalData(filteredData);
    } catch (error) {
      console.error("Error fetching historical data:", error);
      setErrorMessage("Failed to fetch historical data.");
    }
  };

  useEffect(() => {
    fetchRealTimePrice();
    fetchRiskScore();
    fetchHistoricalData();
    fetchAccountInfo(); // Fetch the user's account information on page load
  }, [selectedCrypto, historicalScale]);

  const renderHistoricalChart = () => {
    if (!historicalData) return <p>Loading historical data...</p>;

    const data = {
      labels: historicalData.map((entry) => entry.timestamp),
      datasets: [
        {
          label: `${selectedCrypto} Historical Prices`,
          data: historicalData.map((entry) => entry.price),
          borderColor: "#daa520",
          backgroundColor: "rgba(218, 165, 32, 0.3)",
          fill: true,
        },
      ],
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          labels: {
            color: "#f5c518",
            font: {
              size: 16,
            },
          },
        },
        zoom: {
          zoom: {
            wheel: {
              enabled: true,
            },
            mode: "x",
          },
        },
      },
      scales: {
        x: {
          type: "time",
          time: {
            unit: historicalScale === "1y" ? "month" : "day",
            tooltipFormat: "MMM dd yyyy",
          },
          ticks: {
            color: "#f5c518",
          },
          grid: {
            color: "rgba(255, 255, 255, 0.1)",
          },
        },
        y: {
          ticks: {
            color: "#f5c518",
          },
          grid: {
            color: "rgba(255, 255, 255, 0.1)",
          },
          title: {
            display: true,
            text: "Price (USD)",
            color: "#f5c518",
          },
        },
      },
    };

    return <Line data={data} options={options} />;
  };

  return (
    <div
      style={{
        backgroundColor: "#ffffff",
        color: "#004d00",
        padding: "20px",
        fontFamily: "'Poppins', sans-serif",
        minHeight: "100vh",
      }}
    >
      <Navbar />
      <h1 style={{ textAlign: "center", fontSize: "2.5rem", marginBottom: "40px", marginTop: "20px" }}>
        Crypto Dashboard
      </h1>
      {errorMessage && <p style={{ color: "red", textAlign: "center", marginBottom: "20px" }}>{errorMessage}</p>}
      {successMessage && <p style={{ color: "green", textAlign: "center", marginBottom: "20px" }}>{successMessage}</p>}

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "40px",
        }}
      >
        {/* Crypto Selector */}
        <div style={{ width: "60%", marginBottom: "30px" }}>
          <label
            style={{
              display: "block",
              fontSize: "1.4rem",
              marginBottom: "15px",
            }}
          >
            Select a cryptocurrency:
          </label>
          <select
            value={selectedCrypto}
            onChange={(e) => setSelectedCrypto(e.target.value)}
            style={{
              padding: "15px",
              borderRadius: "5px",
              border: "1px solid #004d00",
              backgroundColor: "#ffffff",
              color: "#004d00",
              width: "100%",
              fontSize: "1.1rem",
            }}
          >
            {cryptoList.map((crypto) => (
              <option key={crypto} value={crypto}>
                {crypto}
              </option>
            ))}
          </select>
        </div>

        {/* Real-Time Price */}
        <div style={{ marginBottom: "30px" }}>
          <h2 style={{ fontSize: "1.8rem", textAlign: "center", marginBottom: "10px" }}>Real-Time Price</h2>
          <p
            style={{
              fontSize: "1.4rem",
              textAlign: "center",
              padding: "15px",
              borderRadius: "5px",
              border: "1px solid #004d00",
              backgroundColor: "#ffffff",
              color: realTimePrice ? "#004d00" : "#d43f3f",
              minWidth: "200px",
            }}
          >
            {realTimePrice ? `$${realTimePrice}` : "Loading real-time price..."}
          </p>
        </div>

        {/* Risk Meter */}
        <div style={{ marginBottom: "30px" }}>
          <h2 style={{ fontSize: "1.8rem", textAlign: "center", marginBottom: "15px" }}>Risk Score</h2>
          {riskScore !== null ? (
            <RiskMeter score={riskScore} />
          ) : (
            <p style={{ textAlign: "center" }}>Loading risk score...</p>
          )}
        </div>

        {/* Historical Data */}
        <div style={{ width: "80%", marginTop: "40px", marginBottom: "30px" }}>
          <h2 style={{ fontSize: "1.8rem", textAlign: "center", marginBottom: "15px" }}>Historical Data</h2>
          <select
            value={historicalScale}
            onChange={(e) => setHistoricalScale(e.target.value)}
            style={{
              padding: "15px",
              borderRadius: "5px",
              border: "1px solid #004d00",
              backgroundColor: "#ffffff",
              color: "#004d00",
              width: "100%",
              marginBottom: "25px",
              fontSize: "1.1rem",
            }}
          >
            <option value="1y">1 Year</option>
            <option value="6m">6 Months</option>
            <option value="3m">3 Months</option>
            <option value="1m">1 Month</option>
          </select>
          {renderHistoricalChart()}
        </div>

        {/* Buy/Sell Section */}
        <div style={{ marginTop: "40px", width: "60%" }}>
          <h2 style={{ fontSize: "1.8rem", textAlign: "center", marginBottom: "15px" }}>Buy/Sell {selectedCrypto}</h2>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "20px",
            }}
          >
            <input
              type="number"
              placeholder="Amount to Buy"
              value={buyAmount}
              onChange={(e) => setBuyAmount(e.target.value)}
              style={{
                padding: "15px",
                borderRadius: "5px",
                border: "1px solid #004d00",
                backgroundColor: "#ffffff",
                color: "#004d00",
                fontSize: "1.1rem",
              }}
            />
            <button
              onClick={handleBuy}
              style={{
                padding: "15px 20px",
                borderRadius: "5px",
                backgroundColor: "#004d00",
                color: "#ffffff",
                fontWeight: "bold",
                border: "none",
                cursor: "pointer",
                transition: "all 0.3s ease",
              }}
            >
              Buy
            </button>
            <input
              type="number"
              placeholder="Amount to Sell"
              value={sellAmount}
              onChange={(e) => setSellAmount(e.target.value)}
              style={{
                padding: "15px",
                borderRadius: "5px",
                border: "1px solid #004d00",
                backgroundColor: "#ffffff",
                color: "#004d00",
                fontSize: "1.1rem",
              }}
            />
            <button
              onClick={handleSell}
              style={{
                padding: "15px 20px",
                borderRadius: "5px",
                backgroundColor: "#004d00",
                color: "#ffffff",
                fontWeight: "bold",
                border: "none",
                cursor: "pointer",
                transition: "all 0.3s ease",
              }}
            >
              Sell
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CryptoPage;
