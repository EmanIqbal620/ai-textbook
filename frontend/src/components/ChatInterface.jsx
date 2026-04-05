import React, { useState, useEffect } from 'react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Initialize session on component mount
  useEffect(() => {
    const initSession = async () => {
      try {
        const response = await fetch('/api/chat/start', {
          method: 'POST',
        });
        const data = await response.json();
        setSessionId(data.sessionId);
      } catch (error) {
        console.error('Error initializing session:', error);
      }
    };

    initSession();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || !sessionId) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`/api/chat/${sessionId}/question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: inputValue }),
      });

      const data = await response.json();

      const botMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        confidence: data.confidence,
        sources: data.sources,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error getting response:', error);

      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.role}`}
            style={{
              display: 'flex',
              justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
              marginBottom: '10px',
            }}
          >
            <div
              style={{
                backgroundColor: message.role === 'user' ? '#007bff' : '#f8f9fa',
                color: message.role === 'user' ? 'white' : 'black',
                padding: '10px 15px',
                borderRadius: '10px',
                maxWidth: '70%',
              }}
            >
              <div>{message.content}</div>
              {message.sources && message.sources.length > 0 && (
                <small style={{ display: 'block', marginTop: '5px', opacity: 0.7 }}>
                  Sources: {message.sources.join(', ')}
                </small>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div
            style={{
              display: 'flex',
              justifyContent: 'flex-start',
              marginBottom: '10px',
            }}
          >
            <div
              style={{
                backgroundColor: '#f8f9fa',
                color: 'black',
                padding: '10px 15px',
                borderRadius: '10px',
                maxWidth: '70%',
              }}
            >
              Thinking...
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} style={{ marginTop: '20px', display: 'flex' }}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a question about the textbook..."
          style={{
            flex: 1,
            padding: '10px',
            border: '1px solid #ccc',
            borderRadius: '4px 0 0 4px',
          }}
          disabled={!sessionId || isLoading}
        />
        <button
          type="submit"
          style={{
            padding: '10px 15px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '0 4px 4px 0',
            cursor: (!sessionId || isLoading) ? 'not-allowed' : 'pointer',
          }}
          disabled={!sessionId || isLoading}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;