import React from 'react';
import { Link } from 'react-router-dom';
import ChatBot from '../components/ChatBot';

const Chat = () => (
  <div className="page chat-page">
    <nav className="navbar">
      <Link to="/">Home</Link>
      <h1>Chat Support</h1>
    </nav>
    <div className="content">
      <ChatBot />
    </div>
  </div>
);
export default Chat;
