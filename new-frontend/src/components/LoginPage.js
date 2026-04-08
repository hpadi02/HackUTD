// src/components/LoginPage.js
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "../styles/LoginPage.css";
import logo from "../assets/HERMES.png"; // Import the logo

const LoginPage = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      if (email && password) {
        const response = await axios.post("/api/users/login", {
          email,
          password,
        });
        localStorage.setItem("token", response.data.token); // Save JWT token
        alert(`Welcome, ${email}!`);
        navigate("/account"); // Redirect to the account page
      } else {
        setErrorMessage("Please enter both email and password.");
      }
    } catch (error) {
      if (error.response && error.response.status === 400) {
        setErrorMessage(error.response.data);
      } else {
        setErrorMessage("An unexpected error occurred. Please try again.");
      }
    }
  };

  return (
    <div className="login-container">
      {/* Company Logo */}
      <div className="company-logo">
        <img src={logo} alt="Company Logo" className="logo-image" />
        <p>Your partner in crypto banking.</p>
      </div>

      {/* Login Box */}
      <div className="login-box">
        <form onSubmit={handleSubmit}>
          <h2>Welcome</h2>
          <p>Online & Mobile Security</p>
          {errorMessage && <p className="text-danger">{errorMessage}</p>}
          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input
              type="email"
              id="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value.trim())} // Trim any trailing spaces
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <div className="form-options">
            <label>
              <input type="checkbox" /> Save email
            </label>
          </div>
          <button type="submit" className="btn-login">
            Sign On
          </button>
          <div className="extra-links">
            <a href="#forgot-password">Forgot Password?</a>
            <a href="/register">New to HERMES®?</a>
          </div>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;
