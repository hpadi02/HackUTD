// src/App.js
import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import LoginPage from "./components/LoginPage";
import AccountPage from "./components/AccountPage";
import CryptoPage from "./components/CryptoPage";
import "./App.css";

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<LoginPage />} />
          <Route path="/account" element={<AccountPage />} />
          <Route path="/crypto" element={<CryptoPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
