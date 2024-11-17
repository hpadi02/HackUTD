import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/LoginPage.css";
import logo from "../assets/HERMES.png"; // Import the logo

const LoginPage = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSubmit = (event) => {
    event.preventDefault();
    if (username && password) {
      alert(`Welcome, ${username}!`);
      navigate("/account"); // Redirect to the account page
    } else {
      alert("Please enter both username and password.");
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
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
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
              <input type="checkbox" /> Save username
            </label>
          </div>
          <button type="submit" className="btn-login">
            Sign On
          </button>
          <div className="extra-links">
            <a href="#forgot-password">Forgot Password/Username?</a>
            <a href="#enroll">New to HERMES®?</a>
          </div>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;
