// src/components/Navbar.js
import React from "react";
import { Link, useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import logo from "../assets/HERMES.png";

const Navbar = () => {
  const navigate = useNavigate();
  const token = localStorage.getItem("token");

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  if (!token) {
    return null; // Don't show navbar if not authenticated
  }

  return (
    <nav
      className="navbar navbar-expand-lg"
      style={{
        backgroundColor: "#ffff",
        color: "#d4af37",
        padding: "10px 20px",
        fontFamily: "'Poppins', sans-serif",
      }}
    >
      <div className="container-fluid">
        <Link
          className="navbar-brand d-flex align-items-center"
          to="/account"
          style={{
            color: "#d4af37",
            fontSize: "1.5rem",
            fontWeight: "bold",
          }}
        >
          <img
            src={logo}
            alt="HERMES Logo"
            style={{
              height: "80px",
              width: "auto",
              marginRight: "10px",
            }}
          />
          HERMES
        </Link>
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
          style={{ borderColor: "#ffffff" }}
        >
          <span className="navbar-toggler-icon" style={{ color: "#ffffff" }}></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item">
              <Link
                className="nav-link"
                to="/account"
                style={{
                  color: "#ffffff",
                  padding: "10px",
                  transition: "color 0.3s ease",
                }}
              >
                My Account
              </Link>
            </li>
            <li className="nav-item">
              <Link
                className="nav-link"
                to="/crypto"
                style={{
                  color: "#ffffff",
                  padding: "10px",
                  transition: "color 0.3s ease",
                }}
              >
                Crypto
              </Link>
            </li>
            <li className="nav-item">
              <button
                className="nav-link text-danger"
                onClick={handleLogout}
                style={{
                  color: "#f5c518",
                  padding: "10px",
                  transition: "color 0.3s ease",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                Logout
              </button>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;