import React, { useState, useRef, useEffect } from 'react';
import styles from './ChatInterface.module.css';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatResponse {
  response: string;
  sources: Array<Record<string, any>>;
  response_time: number;
  query: string;
}

const ChatInterface: React.FC = () => {
  const [inputValue, setInputValue] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedText, setSelectedText] = useState<string>('');
  const [isMinimized, setIsMinimized] = useState<boolean>(true);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Function to scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to toggle minimized/maximized state
  const toggleMinimized = () => {
    setIsMinimized(!isMinimized);
  };

  // Function to get selected text from the page
  const getSelectedText = () => {
    const selectedText = window.getSelection()?.toString().trim() || '';
    setSelectedText(selectedText);
    return selectedText;
  };

  // Function to handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message to chat
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Get any selected text from the page
      const currentSelectedText = getSelectedText();

      // Prepare the request payload
      const requestBody = {
        query: inputValue,
        user_selected_text: currentSelectedText || undefined,
        user_id: 'web_user', // Simple user ID for web interface
        max_tokens: 1000,
        temperature: 0.7,
        top_k: 5
      };

      // Determine backend URL - using deployed Hugging Face Space
      const backendUrl = 'https://emaniqbal-b.hf.space';

      // Send request to backend
      const response = await fetch(`${backendUrl}/api/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status} ${response.statusText}`);
      }

      const data: ChatResponse = await response.json();

      // Process response to enhance structure while removing ** markers for cleaner display
      let processedResponse = data.response;

      // Remove ** markers for cleaner display while adding better structure
      processedResponse = processedResponse
        .replace(/\*\*/g, '') // Remove ** markers for cleaner display
        .replace(/\.\s+(?=[A-Z])/g, '.\n') // Add line break after sentences followed by capital letters
        .replace(/\n\s*\n/g, '\n\n'); // Normalize multiple newlines

      // Make responses slightly more concise by limiting length if too long
      if (processedResponse.length > 1200) {
        const lines = processedResponse.split('\n');
        let shortenedText = '';
        for (const line of lines) {
          if (shortenedText.length + line.length < 1000) {
            shortenedText += line + '\n';
          } else {
            break;
          }
        }
        processedResponse = shortenedText.trim();
      }

      // Add bot response to chat
      const botMessage: Message = {
        id: Date.now().toString(),
        text: processedResponse,
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to chat
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: 'Error, try again',
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setInputValue('');
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage();
  };

  return (
    <div
      className={`${styles.chatContainer} ${isMinimized ? styles.minimized : ''}`}
      onClick={isMinimized ? toggleMinimized : undefined}
    >
      {isMinimized ? (
        <div className={styles.minimizedText} style={{color: '#0d1b2a', fontWeight: 'bold', fontSize: '1rem', fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"}}>
          ASK AI
        </div>
      ) : (
        <>
          <button
            className={styles.clearButton}
            onClick={() => setMessages([])}
            aria-label="Clear chat"
            title="Clear chat"
          >
            Clear
          </button>
          <button
            className={styles.chatToggleButton}
            onClick={toggleMinimized}
            aria-label={isMinimized ? "Open chat" : "Minimize chat"}
          >
            −
          </button>

          <div className={styles.chatHeader}>
            <h3>ASK AI</h3>
          </div>

          <div className={styles.messagesContainer}>
            {messages.length === 0 ? (
              <div className={styles.welcomeMessage}>
                <p>Ask the AI Assistant anything about the textbook!</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`${styles.message} ${
                    message.sender === 'user' ? styles.userMessage : styles.botMessage
                  }`}
                >
                  <div className={styles.messageContent} style={{whiteSpace: 'pre-line'}}>
                    {message.text}
                  </div>
                  <div className={styles.messageTimestamp}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className={`${styles.message} ${styles.botMessage}`}>
                <div className={styles.messageContent}>
                  <div className={styles.typingIndicator}>
                    <div className={styles.dot}></div>
                    <div className={styles.dot}></div>
                    <div className={styles.dot}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form className={styles.inputContainer} onSubmit={handleSubmit}>
            <div className={styles.inputWrapper}>
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question..."
                className={styles.textInput}
                rows={2}
                disabled={isLoading}
              />
              <button
                type="submit"
                className={`${styles.sendButton} ${isLoading ? styles.disabled : ''}`}
                disabled={isLoading || !inputValue.trim()}
              >
                &gt;
              </button>
            </div>
          </form>
        </>
      )}
    </div>
  );
};

export default ChatInterface;