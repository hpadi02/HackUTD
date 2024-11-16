import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import CryptoPage from "./components/CryptoPage";

const App = () => {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/crypto" element={<CryptoPage />} />
        <Route path="/my-account" element={<h1>My Account Page</h1>} />
        <Route path="/logout" element={<h1>Logout Page</h1>} />
      </Routes>
    </Router>
  );
};

export default App;
