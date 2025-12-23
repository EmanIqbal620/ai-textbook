// ChatInterface.js
import React, { useState } from 'react';
import './ChatInterface.css';

const ChatInterface = () => {
  const [inputValue, setInputValue] = useState('');
  const [responses, setResponses] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    // Add user query to display
    const userQuery = { id: Date.now(), text: inputValue, sender: 'user' };
    setResponses(prev => [...prev, userQuery]);

    try {
      // Determine backend URL - using deployed Hugging Face Space
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'https://emaniqbal-b.hf.space';

      // Send query to backend
      const response = await fetch(`${backendUrl}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          user_id: 'web_user'
        }),
      });

      if (!response.ok) {
        throw new Error('Backend request failed');
      }

      const data = await response.json();

      // Add backend response to display
      const botResponse = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot'
      };
      setResponses(prev => [...prev, botResponse]);
    } catch (error) {
      // Add error message to display
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Error, try again',
        sender: 'bot'
      };
      setResponses(prev => [...prev, errorMessage]);
    }

    setInputValue('');
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>AI Assistant</h2>
      </div>

      <div className="chat-messages">
        {responses.map((msg) => (
          <div
            key={msg.id}
            className={`message ${msg.sender}-message`}
          >
            {msg.text}
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Enter your query..."
          className="chat-input"
        />
        <button type="submit" className="submit-button">
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;