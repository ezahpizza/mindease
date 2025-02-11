// src/components/PredictionForm.js
import React, { useState } from 'react';
import { predictionService } from '../services/api';

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    gender: 0,
    age: 0,
    city: 0,
    profession: 0,
    academic_pressure: 0,
    work_pressure: 0,
    cgpa: 0,
    study_satisfaction: 0,
    job_satisfaction: 0,
    sleep_duration: 0,
    dietary_habits: 0,
    degree: 0,
    suicidal_thoughts: 0,
    work_study_hours: 0,
    financial_stress: 0,
    mi_history: 0
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    try {
      const prediction = await predictionService.predict(formData);
      setResult(prediction);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.detail || 'Failed to get prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="prediction-form">
      <h2>Mental Health Assessment</h2>
      {error && <div className="error">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Gender (0-Male, 1-Female):</label>
          <select name="gender" value={formData.gender} onChange={handleChange}>
            <option value="0">Male</option>
            <option value="1">Female</option>
          </select>
        </div>

        <div className="form-group">
          <label>Age:</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
            min="0"
            max="100"
          />
        </div>

        <div className="form-group">
          <label>City Type (0-Rural, 1-Urban):</label>
          <select name="city" value={formData.city} onChange={handleChange}>
            <option value="0">Rural</option>
            <option value="1">Urban</option>
          </select>
        </div>

        <div className="form-group">
          <label>Profession (0-Student, 1-Employed, 2-Both):</label>
          <select name="profession" value={formData.profession} onChange={handleChange}>
            <option value="0">Student</option>
            <option value="1">Employed</option>
            <option value="2">Both</option>
          </select>
        </div>

        <div className="form-group">
          <label>Academic Pressure (0-10):</label>
          <input
            type="range"
            name="academic_pressure"
            value={formData.academic_pressure}
            onChange={handleChange}
            min="0"
            max="10"
            step="0.1"
          />
          <span>{formData.academic_pressure}</span>
        </div>

        <div className="form-group">
          <label>Work Pressure (0-10):</label>
          <input
            type="range"
            name="work_pressure"
            value={formData.work_pressure}
            onChange={handleChange}
            min="0"
            max="10"
            step="0.1"
          />
          <span>{formData.work_pressure}</span>
        </div>

        <div className="form-group">
          <label>CGPA (0-10):</label>
          <input
            type="number"
            name="cgpa"
            value={formData.cgpa}
            onChange={handleChange}
            min="0"
            max="10"
            step="0.1"
          />
        </div>

        <div className="form-group">
          <label>Study Satisfaction (0-10):</label>
          <input
            type="range"
            name="study_satisfaction"
            value={formData.study_satisfaction}
            onChange={handleChange}
            min="0"
            max="10"
            step="0.1"
          />
          <span>{formData.study_satisfaction}</span>
        </div>

        <div className="form-group">
          <label>Job Satisfaction (0-10):</label>
          <input
            type="range"
            name="job_satisfaction"
            value={formData.job_satisfaction}
            onChange={handleChange}
            min="0"
            max="10"
            step="0.1"
          />
          <span>{formData.job_satisfaction}</span>
        </div>

        <div className="form-group">
          <label>Sleep Duration (0-Less than 6hrs, 1-6-8hrs, 2-More than 8hrs):</label>
          <select name="sleep_duration" value={formData.sleep_duration} onChange={handleChange}>
            <option value="0">Less than 6 hours</option>
            <option value="1">6-8 hours</option>
            <option value="2">More than 8 hours</option>
          </select>
        </div>

        <div className="form-group">
          <label>Dietary Habits (0-Unhealthy, 1-Moderate, 2-Healthy):</label>
          <select name="dietary_habits" value={formData.dietary_habits} onChange={handleChange}>
            <option value="0">Unhealthy</option>
            <option value="1">Moderate</option>
            <option value="2">Healthy</option>
          </select>
        </div>

        <div className="form-group">
          <label>Degree (0-Bachelors, 1-Masters, 2-PhD):</label>
          <select name="degree" value={formData.degree} onChange={handleChange}>
            <option value="0">Bachelors</option>
            <option value="1">Masters</option>
            <option value="2">PhD</option>
          </select>
        </div>

        <div className="form-group">
          <label>Suicidal Thoughts (0-No, 1-Yes):</label>
          <select name="suicidal_thoughts" value={formData.suicidal_thoughts} onChange={handleChange}>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div className="form-group">
          <label>Work/Study Hours per Day (0-24):</label>
          <input
            type="number"
            name="work_study_hours"
            value={formData.work_study_hours}
            onChange={handleChange}
            min="0"
            max="24"
            step="0.5"
          />
        </div>

        <div className="form-group">
          <label>Financial Stress (0-10):</label>
          <input
            type="range"
            name="financial_stress"
            value={formData.financial_stress}
            onChange={handleChange}
            min="0"
            max="10"
            step="0.1"
          />
          <span>{formData.financial_stress}</span>
        </div>

        <div className="form-group">
          <label>Mental Illness History (0-No, 1-Yes):</label>
          <select name="mi_history" value={formData.mi_history} onChange={handleChange}>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>
        
        <button type="submit">Get Prediction</button>
      </form>

      {result && (
        <div className="prediction-result">
          <h3>Prediction Result</h3>
          <p>Risk Score: {result.prediction.toFixed(2)}</p>
          <p>Assessment: {result.prediction_label}</p>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
