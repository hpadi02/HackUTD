// src/components/AccountPage.js
import React, { useEffect, useState } from "react";
import Navbar from "./Navbar";
import axios from "axios";
import "../styles/AccountPage.css"; // Importing the CSS file

const AccountPage = () => {
  const [userData, setUserData] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    // Fetch user account info
    const fetchAccountInfo = async () => {
      try {
        const token = localStorage.getItem("token"); // Assuming token is stored in localStorage
        const response = await axios.get("/api/users/account", {
          headers: {
            "x-auth-token": token,
          },
        });
        setUserData(response.data);
      } catch (error) {
        setErrorMessage("Failed to load account information. Please try again.");
      }
    };
    fetchAccountInfo();
  }, []);

  return (
    <div>
      <Navbar />
      <div className="container mt-4">
        <h1>My Account</h1>
        {errorMessage && <p className="text-danger">{errorMessage}</p>}
        {userData ? (
          <>
            <div className="card mb-4">
              <div className="card-body">
                <h5 className="card-title">Balance</h5>
                <p className="card-text">{`${userData.currency} ${userData.balance.toFixed(2)}`}</p>
              </div>
            </div>
            <div className="card mb-4">
              <div className="card-body">
                <h5 className="card-title">Credit Score</h5>
                <p className="card-text">{userData.creditScore}</p>
              </div>
            </div>
            <div className="card mb-4">
              <div className="card-body">
                <h5 className="card-title">Statements</h5>
                {userData.statements.length > 0 ? (
                  <ul>
                    {userData.statements.map((statement, index) => (
                      <li key={index}>{statement}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="card-text">No recent transactions or statements available.</p>
                )}
              </div>
            </div>
            <div className="card">
              <div className="card-body">
                <h5 className="card-title">Contact Info</h5>
                <p className="card-text">Email: {userData.email}</p>
                <p className="card-text">Phone: {userData.phone}</p>
              </div>
            </div>
          </>
        ) : (
          <p>Loading account information...</p>
        )}
      </div>
    </div>
  );
};

export default AccountPage;
