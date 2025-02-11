import React from 'react';
import { Link } from 'react-router-dom';
import RegisterForm from '../components/RegisterForm';

const Register = () => (
  <div className="page register-page">
    <div className="auth-container">
      <RegisterForm />
      <p>Already have an account? <Link to="/login">Login</Link></p>
    </div>
  </div>
);
export default Register;
