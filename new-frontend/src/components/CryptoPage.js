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

const RiskMeter = ({ score }) => {
  // Map score (1-10) to degrees (180-0), so 1 points to green (180°) and 10 points to red (0°)
  const needlePosition = 180 - (Math.min(Math.max(score, 1), 10) - 1) * 20;
  const getColor = () => {
    if (score > 7) return "#e74c3c"; // red
    if (score > 4) return "#f1c40f"; // yellow
    return "#27ae60"; // green
  };

  // Tooltip state
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div style={{ position: "relative", width: "260px", height: "160px", margin: "0 auto" }}>
      <svg viewBox="0 0 200 110" style={{ width: "100%" }}>
        <defs>
          <linearGradient id="riskGradient" x1="100%" y1="0%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="#e74c3c" />
            <stop offset="50%" stopColor="#f1c40f" />
            <stop offset="100%" stopColor="#27ae60" />
          </linearGradient>
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="#888" />
          </filter>
        </defs>
        <path
          d="M10,100 Q100,-30 190,100"
          fill="none"
          stroke="#eee"
          strokeWidth="14"
          filter="url(#shadow)"
        />
        <path
          d="M10,100 Q100,-30 190,100"
          fill="none"
          stroke="url(#riskGradient)"
          strokeWidth="10"
          strokeDasharray="200"
        />
        <line
          x1="100"
          y1="100"
          x2={100 + 80 * Math.cos((Math.PI * needlePosition) / 180)}
          y2={100 - 80 * Math.sin((Math.PI * needlePosition) / 180)}
          stroke="#222"
          strokeWidth="5"
          strokeLinecap="round"
          style={{ transition: "all 0.5s cubic-bezier(.4,2,.6,1)", filter: "url(#shadow)" }}
        />
        <circle cx="100" cy="100" r="7" fill="#fff" stroke="#888" strokeWidth="2" />
      </svg>
      <div style={{ textAlign: "center", marginTop: "-10px" }}>
        <span style={{ fontSize: 28, fontWeight: 700, color: getColor() }}>Risk Score: {score}</span>
        <span
          style={{ marginLeft: 8, cursor: "pointer", color: "#888" }}
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          <svg width="18" height="18" viewBox="0 0 20 20" style={{ verticalAlign: "middle" }}>
            <circle cx="10" cy="10" r="9" fill="#fff" stroke="#888" strokeWidth="2" />
            <text x="10" y="15" textAnchor="middle" fontSize="13" fill="#888">i</text>
          </svg>
        </span>
        {showTooltip && (
          <div style={{
            position: "absolute", left: "50%", top: 0, transform: "translate(-50%, -110%)",
            background: "#fff", color: "#333", border: "1px solid #ccc", borderRadius: 6, padding: 10, fontSize: 13, boxShadow: "0 2px 8px rgba(0,0,0,0.12)", zIndex: 10, width: 220
          }}>
            AI-powered risk score based on the last year of data. 1 = secure, 10 = risky. This score helps you assess the volatility and risk of this cryptocurrency.
          </div>
        )}
      </div>
      <div style={{ textAlign: "center", color: "#888", fontSize: 15, marginTop: 4 }}>
        {score > 7 ? "High risk" : score > 4 ? "Moderate risk" : "Low risk"}
      </div>
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

  // Clear messages after 5 seconds
  useEffect(() => {
    if (errorMessage || successMessage) {
      const timer = setTimeout(() => {
        setErrorMessage("");
        setSuccessMessage("");
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [errorMessage, successMessage]);

  // Fetch account info
  const fetchAccountInfo = async () => {
    try {
      const token = localStorage.getItem("token");
      const response = await axios.get("/api/users/account", {
        headers: { "x-auth-token": token },
      });
      setAccountInfo(response.data);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || "Error fetching account info";
      setErrorMessage(errorMsg);
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
      if (!token) {
        setErrorMessage("Please log in to make transactions.");
        return;
      }

      const formData = new FormData();
      formData.append("amount", buyAmount);
      formData.append("selectedCrypto", selectedCrypto);
      
      const response = await axios.post("/api/users/buy", formData, {
        headers: {
          "x-auth-token": token,
          "Content-Type": "multipart/form-data"
        }
      });
      setSuccessMessage(typeof response.data === 'string' ? response.data : 'Purchase successful');
      setBuyAmount("");
      await fetchAccountInfo(); // Refresh account details
      await fetchRealTimePrice(); // Refresh price
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || "Error buying cryptocurrency";
      setErrorMessage(errorMsg);
      if (error.response?.status === 401) {
        // Handle unauthorized error - maybe redirect to login
        localStorage.removeItem("token");
      }
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
      if (!token) {
        setErrorMessage("Please log in to make transactions.");
        return;
      }

      const formData = new FormData();
      formData.append("amount", sellAmount);
      formData.append("selectedCrypto", selectedCrypto);
      
      const response = await axios.post("/api/users/sell", formData, {
        headers: {
          "x-auth-token": token,
          "Content-Type": "multipart/form-data"
        }
      });
      setSuccessMessage(typeof response.data === 'string' ? response.data : 'Sale successful');
      setSellAmount("");
      await fetchAccountInfo(); // Refresh account details
      await fetchRealTimePrice(); // Refresh price
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || "Error selling cryptocurrency";
      setErrorMessage(errorMsg);
      if (error.response?.status === 401) {
        // Handle unauthorized error - maybe redirect to login
        localStorage.removeItem("token");
      }
    }
  };

  const fetchRealTimePrice = async () => {
    try {
      const response = await axios.get(`/api/crypto/price/${selectedCrypto}`);
      setRealTimePrice(response.data.price);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || "Failed to fetch real-time price";
      setErrorMessage(errorMsg);
    }
  };

  const fetchRiskScore = async () => {
    try {
      const response = await axios.get(`/api/crypto/risk_score/${selectedCrypto}`);
      setRiskScore(response.data.risk_score);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || "Failed to fetch risk score";
      setErrorMessage(errorMsg);
    }
  };

  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get(
        `/api/crypto/historical/${selectedCrypto}?timeframe=${historicalScale}`
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
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || "Failed to fetch historical data";
      setErrorMessage(errorMsg);
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
