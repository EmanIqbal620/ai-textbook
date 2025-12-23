// ChatInterface.js
import React, { useState } from 'react';
import './ChatInterface.css';

const ChatInterface = () => {
  const [inputValue, setInputValue] = useState('');
  const [responses, setResponses] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [submitDisabled, setSubmitDisabled] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Disable submit button and show loading state
    setIsLoading(true);
    setSubmitDisabled(true);

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
    } finally {
      // Reset loading states
      setIsLoading(false);
      setSubmitDisabled(false);
      setInputValue('');
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // Add a typing indicator for better UX
  const renderMessages = () => {
    const messages = [...responses];

    // Add loading indicator if currently loading
    if (isLoading) {
      messages.push({
        id: 'loading',
        text: '...',
        sender: 'bot',
        isTyping: true
      });
    }

    return messages.map((msg) => (
      <div
        key={msg.id}
        className={`message ${msg.sender}-message ${msg.isTyping ? 'typing-indicator' : ''}`}
      >
        {msg.isTyping ? (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        ) : (
          msg.text
        )}
      </div>
    ));
  };

  return (
    <>
      {/* Chat toggle button */}
      <button className="chat-toggle-button" onClick={toggleChat}>
        <img src="/static/img/chatbot-icon.svg" alt="Chat" width="40" height="40" />
      </button>

      {/* Chat container - only show when open */}
      {isOpen && (
        <div className="chat-container">
          <div className="chat-header">
            <h2>ASK AI</h2>
            <button className="close-button" onClick={toggleChat}>×</button>
          </div>

          <div className="chat-messages">
            {renderMessages()}
          </div>

          <form onSubmit={handleSubmit} className="chat-input-form">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Enter your query..."
              className="chat-input"
              disabled={submitDisabled}
            />
            <button
              type="submit"
              className={`submit-button ${submitDisabled ? 'disabled' : ''}`}
              disabled={submitDisabled}
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </form>
        </div>
      )}
    </>
  );
};

export default ChatInterface;