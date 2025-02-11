// src/components/Dashboard.js
import React, { useState, useEffect } from 'react';
import { getUserPredictions } from '../services/api';

const Dashboard = () => {
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        loadPredictions();
    }, []);

    const loadPredictions = async () => {
        try {
        const data = await getUserPredictions();
        setPredictions(data);
        } catch (err) {
        setError('Failed to load predictions');
        } finally {
        setLoading(false);
        }
    };

    if (loading) return <div>Loading...</div>;
    if (error) return <div className="error">{error}</div>;

    return (
        <div className="dashboard">
        <h2>Your Assessment History</h2>
        <div className="predictions-list">
            {predictions.length === 0 ? (
            <p>No assessments taken yet.</p>
            ) : (
            predictions.map((pred, index) => (
                <div key={index} className="prediction-card">
                <div className="prediction-header">
                    <h3>Assessment {index + 1}</h3>
                    <span className="date">
                    {new Date(pred.timestamp).toLocaleDateString()}
                    </span>
                </div>
                <div className="prediction-details">
                    <p className="score">Risk Score: {pred.prediction.toFixed(2)}</p>
                    <p className="label">Result: {pred.prediction_label}</p>
                </div>
                </div>
            ))
            )}
        </div>
        </div>
    );
    };

export default Dashboard;