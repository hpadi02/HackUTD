// src/components/AccountPage.js
import React, { useEffect, useState } from "react";
import axios from "axios";
import "../styles/AccountPage.css"; // Importing the CSS file

const AccountPage = () => {
  const [userData, setUserData] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    // Fetch user account info
    const fetchAccountInfo = async () => {
      try {
        const token = localStorage.getItem("token");
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
    <div className="container mt-4">
      <h1>My Account</h1>
      {errorMessage && <p className="text-danger">{errorMessage}</p>}
      {userData ? (
        <>
          <div className="card mb-4">
            <div className="card-body">
              <h5 className="card-title">Balance</h5>
              <p className="card-text">${userData.balance.toFixed(2)}</p>
            </div>
          </div>
          <div className="card mb-4">
            <div className="card-body">
              <h5 className="card-title">Cryptocurrency Holdings</h5>
              {Object.entries(userData.crypto).length > 0 ? (
                <ul className="list-unstyled">
                  {Object.entries(userData.crypto).map(([crypto, amount]) => (
                    <li key={crypto}>{crypto}: {amount}</li>
                  ))}
                </ul>
              ) : (
                <p className="card-text">No cryptocurrency holdings.</p>
              )}
            </div>
          </div>
          <div className="card">
            <div className="card-body">
              <h5 className="card-title">Personal Info</h5>
              <p className="card-text">Name: {userData.name}</p>
              <p className="card-text">Email: {userData.email}</p>
              <p className="card-text">Phone: {userData.phone}</p>
            </div>
          </div>
        </>
      ) : (
        <p>Loading account information...</p>
      )}
    </div>
  );
};

export default AccountPage;
