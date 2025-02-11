import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Home = () => {
  const { user, logout } = useAuth();

  return (
    <div className="page home-page">
      <nav className="navbar">
        <h1>MindEase</h1>
        <div className="nav-links">
          <Link to="/prediction">Assessment</Link>
          <Link to="/chat">Chat Support</Link>
          <button onClick={logout}>Logout</button>
        </div>
      </nav>
      <div className="welcome-section">
        <h2>Welcome, {user?.username}!</h2>
        <p>Choose a service to get started:</p>
        <div className="service-cards">
          <Link to="/prediction" className="card">
            <h3>Mental Health Assessment</h3>
            <p>Take our comprehensive assessment to evaluate your mental well-being</p>
          </Link>
          <Link to="/chat" className="card">
            <h3>Chat Support</h3>
            <p>Talk to our AI assistant for guidance and support</p>
          </Link>
        </div>
      </div>
    </div>
  );
};
export default Home;

