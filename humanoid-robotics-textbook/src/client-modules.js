import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import ChatInterface from './components/ChatInterface';

if (ExecutionEnvironment.canUseDOM) {
  // Mount the chat interface when the DOM is available
  window.addEventListener('load', () => {
    const container = document.getElementById('chat-widget-container');
    if (container) {
      // Use React to render the ChatInterface into the container
      const React = require('react');
      const ReactDOM = require('react-dom/client');

      const root = ReactDOM.createRoot(container);
      root.render(React.createElement(ChatInterface));
    }
  });
}

export default [];