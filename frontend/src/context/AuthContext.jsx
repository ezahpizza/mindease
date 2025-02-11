import React, { createContext, useState, useContext, useEffect } from 'react';
import { login, register } from '../services/auth';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));

  useEffect(() => {
    if (token) {
      // Optionally validate token here
      localStorage.setItem('token', token);
    } else {
      localStorage.removeItem('token');
    }
  }, [token]);

  const loginUser = async (username, password) => {
    try {
      const response = await login(username, password);
      setToken(response.access_token);
      setUser({ username });
      return response;
    } catch (error) {
      console.error('Login failed', error);
      throw error;
    }
  };

  const registerUser = async (username, email, password) => {
    try {
      const response = await register(username, email, password);
      return response;
    } catch (error) {
      console.error('Registration failed', error);
      throw error;
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ 
      user, 
      token, 
      login: loginUser, 
      register: registerUser, 
      logout 
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
