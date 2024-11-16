import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav className="navbar">
      <Link to="/my-account">My Account</Link>
      <Link to="/crypto">Crypto</Link>
      <Link to="/logout">Logout</Link>
    </nav>
  );
};

export default Navbar;
