import React from 'react';
import { Link } from 'react-router-dom';
import LoginForm from '../components/LoginForm';

const Login = () => (
  <div className="page login-page">
    <div className="auth-container">
      <LoginForm />
      <p>Don't have an account? <Link to="/register">Register</Link></p>
    </div>
  </div>
);
export default Login;
