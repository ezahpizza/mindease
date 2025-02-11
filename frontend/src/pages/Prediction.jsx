import React from 'react';
import { Link } from 'react-router-dom';
import PredictionForm from '../components/PredictionForm';

const Prediction = () => (
  <div className="page prediction-page">
    <nav className="navbar">
      <Link to="/">Home</Link>
      <h1>Mental Health Assessment</h1>
    </nav>
    <div className="content">
      <PredictionForm />
    </div>
  </div>
);
export default Prediction;

