# Frontend Chat Interface

This is a React-based chat interface that connects to the FastAPI backend RAG chatbot.

## Files

- `ChatInterface.js` - Main React component with:
  - Text input field for user queries
  - Submit button
  - Area to display backend responses
  - Error handling (shows "Error, try again" when backend fails)

- `ChatInterface.css` - Styling for the chat interface

## Features

- Sends user queries to backend API
- Displays responses from the backend
- Shows loading states during API requests
- Error handling when backend is unavailable
- Supports selected text context (can be extended)

## Configuration

To connect to a different backend URL, set the `REACT_APP_BACKEND_URL` environment variable. Defaults to `http://localhost:8000`.

## Usage

The component can be integrated into any React application by importing and using it:

```jsx
import ChatInterface from './ChatInterface';

function App() {
  return (
    <div className="App">
      <ChatInterface />
    </div>
  );
}
```