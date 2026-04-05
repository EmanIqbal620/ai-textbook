import React, { useState, useRef, useEffect } from 'react';
import './App.css';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp: Date;
  responseTime?: number;
}

interface Source {
  chapter_name: string;
  source_url: string;
  score: number;
}

interface ChatResponse {
  response: string;
  sources: Source[];
  response_time: number;
  query: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load chat history from localStorage
    const savedMessages = localStorage.getItem('chatHistory');
    if (savedMessages) {
      try {
        const parsed = JSON.parse(savedMessages);
        setMessages(parsed.map((m: any) => ({
          ...m,
          timestamp: new Date(m.timestamp)
        })));
      } catch (e) {
        console.error('Failed to load chat history');
      }
    }
  }, []);

  useEffect(() => {
    // Save chat history to localStorage
    if (messages.length > 0) {
      localStorage.setItem('chatHistory', JSON.stringify(messages));
    }
  }, [messages]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const startTime = Date.now();
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          user_id: 'user_' + localStorage.getItem('userId') || 'anonymous',
          top_k: 5,
          max_tokens: 500,
          temperature: 0.7
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data: ChatResponse = await response.json();
      const endTime = Date.now();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        sources: data.sources,
        timestamp: new Date(),
        responseTime: data.response_time || (endTime - startTime) / 1000
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please check if the backend server is running and try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
      // Focus back on input
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const clearChat = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      setMessages([]);
      localStorage.removeItem('chatHistory');
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="logo">
            <svg className="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5z" strokeWidth="2"/>
              <path d="M2 17l10 5 10-5" strokeWidth="2"/>
              <path d="M2 12l10 5 10-5" strokeWidth="2"/>
            </svg>
            <span>Robotics Tutor</span>
          </div>
          <button 
            className="close-sidebar" 
            onClick={() => setIsSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            ×
          </button>
        </div>
        
        <div className="sidebar-content">
          <div className="quick-questions">
            <h3>Quick Questions</h3>
            <button onClick={() => setInput('What is ROS2?')}>What is ROS2?</button>
            <button onClick={() => setInput('What is a humanoid robot?')}>What is a humanoid?</button>
            <button onClick={() => setInput('Explain NVIDIA Isaac')}>What is NVIDIA Isaac?</button>
            <button onClick={() => setInput('What is URDF?')}>What is URDF?</button>
            <button onClick={() => setInput('Hardware requirements?')}>Hardware requirements?</button>
          </div>

          <div className="chat-info">
            <h3>About</h3>
            <p>This AI tutor helps you learn from the Humanoid Robotics textbook. Ask any question about ROS2, simulation, AI, or hardware!</p>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        {/* Header */}
        <header className="chat-header">
          <button 
            className="toggle-sidebar" 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            aria-label="Toggle sidebar"
          >
            ☰
          </button>
          <div className="header-title">
            <h1>Humanoid Robotics AI Tutor</h1>
            <p className="status-indicator">
              <span className="status-dot"></span>
              Online - Ready to help
            </p>
          </div>
          <button className="clear-chat" onClick={clearChat} aria-label="Clear chat">
            🗑️ Clear
          </button>
          <button 
            className="close-chat-btn" 
            onClick={() => setIsSidebarOpen(false)}
            aria-label="Close chatbot"
            style={{marginLeft: 'auto'}}
          >
            ✕ CLOSE
          </button>
        </header>

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">🤖</div>
              <h2>Welcome to Your Robotics Tutor!</h2>
              <p>I'm here to help you learn about humanoid robotics. Ask me anything!</p>
              <div className="suggestion-cards">
                <button onClick={() => setInput('What will I learn in this book?')} className="suggestion-card">
                  <span className="card-icon">📚</span>
                  <span>What will I learn?</span>
                </button>
                <button onClick={() => setInput('What are the prerequisites?')} className="suggestion-card">
                  <span className="card-icon">🎯</span>
                  <span>Prerequisites</span>
                </button>
                <button onClick={() => setInput('Explain ROS2 basics')} className="suggestion-card">
                  <span className="card-icon">🔧</span>
                  <span>ROS2 Basics</span>
                </button>
                <button onClick={() => setInput('What is simulation in robotics?')} className="suggestion-card">
                  <span className="card-icon">💻</span>
                  <span>Simulation</span>
                </button>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <div 
                  key={message.id} 
                  className={`message ${message.role} ${message.sources ? 'with-sources' : ''}`}
                >
                  <div className="message-avatar">
                    {message.role === 'user' ? (
                      <svg viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
                        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                      </svg>
                    ) : (
                      <span>🤖</span>
                    )}
                  </div>
                  <div className="message-content">
                    <div className="message-header">
                      <span className="message-role">{message.role === 'user' ? 'You' : 'Robotics Tutor'}</span>
                      <span className="message-time">{formatTime(message.timestamp)}</span>
                    </div>
                    <div className="message-text">{message.content}</div>
                    {message.sources && message.sources.length > 0 && (
                      <div className="message-sources">
                        <h4>📖 Sources:</h4>
                        <div className="sources-list">
                          {message.sources.map((source, idx) => (
                            <a 
                              key={idx} 
                              href={source.source_url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="source-item"
                            >
                              <span className="source-chapter">{source.chapter_name}</span>
                              <span className="source-score">Relevance: {(source.score * 100).toFixed(0)}%</span>
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                    {message.responseTime && (
                      <div className="response-time">Response time: {message.responseTime.toFixed(2)}s</div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message assistant loading">
                  <div className="message-avatar">
                    <span>🤖</span>
                  </div>
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <p className="loading-text">Searching the textbook...</p>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <form className="input-form" onSubmit={handleSubmit}>
          <div className="input-wrapper">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about humanoid robotics..."
              rows={1}
              disabled={isLoading}
            />
            <button 
              type="submit" 
              className="send-button"
              disabled={!input.trim() || isLoading}
              aria-label="Send message"
            >
              {isLoading ? (
                <span className="loading-spinner"></span>
              ) : (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M22 2L11 13" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M22 2L15 22L11 13L2 9L22 2Z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              )}
            </button>
          </div>
          <p className="input-hint">Press Enter to send, Shift+Enter for new line</p>
        </form>
      </main>
    </div>
  );
}

export default App;
