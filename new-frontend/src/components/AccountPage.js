// src/components/AccountPage.js
import React from "react";
import Navbar from "./Navbar";

const AccountPage = () => {
  return (
    <div>
      <Navbar />
      <div className="container mt-4">
        <h1>My Account</h1>
        <div className="card mb-4">
          <div className="card-body">
            <h5 className="card-title">Balance</h5>
            <p className="card-text">$5,000.00</p>
          </div>
        </div>
        <div className="card mb-4">
          <div className="card-body">
            <h5 className="card-title">Credit Score</h5>
            <p className="card-text">750</p>
          </div>
        </div>
        <div className="card mb-4">
          <div className="card-body">
            <h5 className="card-title">Statements</h5>
            <p className="card-text">Recent transactions and statements go here.</p>
          </div>
        </div>
        <div className="card">
          <div className="card-body">
            <h5 className="card-title">Contact Info</h5>
            <p className="card-text">Email: user@example.com</p>
            <p className="card-text">Phone: (123) 456-7890</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AccountPage;
