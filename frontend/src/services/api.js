import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Auth services
export const authService = {
  async register(userData) {
    const response = await api.post('/register', userData);
    return response.data;
  },

  async login(username, password) {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    const response = await api.post('/token', formData);
    if (response.data.access_token) {
      localStorage.setItem('token', response.data.access_token);
    }
    return response.data;
  },

  logout() {
    localStorage.removeItem('token');
  }
};

// Prediction service
export const predictionService = {
  async predict(formData) {
    const response = await api.post('/predict', formData);
    return response.data;
  },

  async getPredictionHistory() {
    const response = await api.get('/user/predictions');
    return response.data;
  }
};

// Chat service
export const chatService = {
  async sendMessage(message) {
    const response = await api.post('/chat', { message });
    return response.data;
  },

  async getChatHistory() {
    const response = await api.get('/chat/history');
    return response.data;
  }
};

export default api;