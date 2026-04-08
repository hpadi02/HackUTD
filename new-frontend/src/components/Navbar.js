// src/components/Navbar.js
import React from "react";
import { Link } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import logo from "../assets/HERMES.png";

const Navbar = () => {
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
        {/* Use the imported image */}
        <img
          src={logo} // Reference the imported image
          alt="HERMES Logo"
          style={{
            height: "80px", // Adjust the height to your preference
            width: "auto", // Maintain aspect ratio
            marginRight: "10px", // Add spacing between logo and text (optional)
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
              <Link
                className="nav-link text-danger"
                to="/"
                style={{
                  color: "#f5c518",
                  padding: "10px",
                  transition: "color 0.3s ease",
                }}
              >
                Logout
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;