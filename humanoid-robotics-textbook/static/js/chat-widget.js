/**
 * TERMINAL-STYLE CHAT WIDGET - FIXED VERSION
 * - User/Robot icons on messages
 * - Clear chat button at top
 * - Stationary (no movement)
 */

(function() {
  // Configuration
  const CONFIG = {
    apiBaseUrl: 'http://localhost:8000/api/v1',
    widgetId: 'terminal-chat-widget',
    buttonId: 'terminal-chat-button',
    panelId: 'terminal-chat-panel',
  };

  // State
  let isOpen = false;
  let messages = [];
  let isLoading = false;
  let selectedText = '';

  /**
   * Create terminal styles
   */
  function createStyles() {
    const styles = `
      /* ============================================
         ROOT & GLOBAL
         ============================================ */
      #${CONFIG.widgetId}-container * {
        font-family: 'Courier New', monospace;
        box-sizing: border-box;
      }

      /* ============================================
         CHAT BUTTON (Bottom Right - STATIONARY)
         ============================================ */
      #${CONFIG.buttonId} {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 64px;
        height: 64px;
        background: #0a0e1a;
        border: 2px solid #00d4ff;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 8px #00d4ff, inset 0 0 10px rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
        z-index: 999998;
        animation: pulseGlow 2s ease-in-out infinite;
      }

      #${CONFIG.buttonId}:hover {
        transform: scale(1.1);
        box-shadow: 0 0 15px #00d4ff, inset 0 0 20px rgba(0, 212, 255, 0.4);
      }

      #${CONFIG.buttonId}.open {
        transform: rotate(90deg);
      }

      @keyframes pulseGlow {
        0%, 100% {
          box-shadow: 0 0 8px #00d4ff, inset 0 0 10px rgba(0, 212, 255, 0.2);
        }
        50% {
          box-shadow: 0 0 20px #00d4ff, inset 0 0 25px rgba(0, 212, 255, 0.5);
        }
      }

      /* Robot Icon */
      #${CONFIG.buttonId} svg {
        width: 32px;
        height: 32px;
        fill: #00d4ff;
      }

      /* Online Indicator */
      #${CONFIG.buttonId}::after {
        content: '';
        position: absolute;
        bottom: 4px;
        right: 4px;
        width: 12px;
        height: 12px;
        background: #06d6a0;
        border: 2px solid #0a0e1a;
        border-radius: 50%;
        animation: blinkOnline 1s ease-in-out infinite;
      }

      @keyframes blinkOnline {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
      }

      /* ============================================
         CHAT PANEL (STATIONARY - NO MOVEMENT)
         ============================================ */
      #${CONFIG.panelId} {
        position: fixed;
        bottom: 100px;
        right: 24px;
        width: 420px;
        max-width: calc(100vw - 48px);
        height: 600px;
        max-height: calc(100vh - 120px);
        background: #0a0e1a;
        border: 1px solid #00d4ff;
        box-shadow: 0 0 8px #00d4ff, inset 0 0 30px rgba(0, 212, 255, 0.05);
        display: none;
        flex-direction: column;
        overflow: hidden;
        z-index: 999997;
        border-radius: 4px;
      }

      #${CONFIG.panelId}.open {
        display: flex;
      }

      /* Grid Background Pattern */
      #${CONFIG.panelId}::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
          linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
        background-size: 20px 20px;
        pointer-events: none;
        z-index: 0;
      }

      /* ============================================
         HEADER WITH CLEAR BUTTON
         ============================================ */
      .terminal-header {
        display: flex;
        flex-direction: column;
        padding: 20px 16px 16px 16px;
        background: linear-gradient(180deg, rgba(0, 212, 255, 0.1) 0%, rgba(10, 14, 26, 0) 100%);
        border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        position: relative;
        z-index: 1;
      }

      .terminal-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 8px;
        padding-right: 40px;
      }

      .terminal-title {
        display: flex;
        align-items: center;
        gap: 12px;
        color: #00d4ff;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
        flex: 1;
      }

      .terminal-title svg {
        width: 24px;
        height: 24px;
        fill: #00d4ff;
      }

      .terminal-clear-btn {
        background: linear-gradient(135deg, rgba(239, 71, 111, 0.2), rgba(247, 37, 133, 0.2));
        border: 1px solid #ef476f;
        color: #ef476f;
        padding: 6px 12px;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        font-weight: 700;
        cursor: pointer;
        border-radius: 4px;
        transition: all 0.2s;
        letter-spacing: 1px;
        box-shadow: 0 0 8px rgba(239, 71, 111, 0.3);
        text-shadow: 0 0 5px rgba(239, 71, 111, 0.5);
        flex-shrink: 0;
        white-space: nowrap;
      }

      .terminal-clear-btn:hover {
        background: rgba(239, 71, 111, 0.4);
        box-shadow: 0 0 15px rgba(239, 71, 111, 0.8);
        transform: scale(1.05);
      }

      .terminal-subtitle {
        font-size: 12px;
        color: #06d6a0;
        display: flex;
        align-items: center;
        gap: 4px;
      }

      .terminal-subtitle::after {
        content: '_';
        animation: blinkCursor 1s step-end infinite;
      }

      @keyframes blinkCursor {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
      }

      .terminal-close {
        position: absolute;
        top: 12px;
        right: 12px;
        background: linear-gradient(135deg, rgba(239, 71, 111, 0.3), rgba(247, 37, 133, 0.3));
        border: 2px solid #ef476f;
        color: #ef476f;
        width: 36px;
        height: 36px;
        cursor: pointer;
        font-family: 'Courier New', monospace;
        font-size: 22px;
        font-weight: 900;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(239, 71, 111, 0.5);
        text-shadow: 0 0 5px rgba(239, 71, 111, 0.8);
        z-index: 100;
      }

      .terminal-close:hover {
        background: rgba(239, 71, 111, 0.5);
        box-shadow: 0 0 20px rgba(239, 71, 111, 1);
        transform: scale(1.15);
        color: white;
      }

      /* ============================================
         MESSAGES CONTAINER
         ============================================ */
      .terminal-messages {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
        position: relative;
        z-index: 1;
      }

      .terminal-messages::-webkit-scrollbar {
        width: 6px;
      }

      .terminal-messages::-webkit-scrollbar-track {
        background: #0a0e1a;
      }

      .terminal-messages::-webkit-scrollbar-thumb {
        background: #00d4ff;
        border-radius: 2px;
      }

      /* ============================================
         MESSAGE STYLES WITH ICONS
         ============================================ */
      .terminal-message {
        display: flex;
        gap: 10px;
        align-items: flex-start;
        max-width: 85%;
        animation: messageFadeIn 0.3s ease forwards;
        opacity: 0;
      }

      @keyframes messageFadeIn {
        from {
          opacity: 0;
          transform: translateX(-10px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }

      .terminal-message.user {
        align-self: flex-end;
        flex-direction: row-reverse;
      }

      .terminal-message.user .message-slide-in {
        animation-name: messageSlideInUser;
      }

      @keyframes messageSlideInUser {
        from {
          opacity: 0;
          transform: translateX(10px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }

      .terminal-message.assistant {
        align-self: flex-start;
      }

      /* Message Icons */
      .message-icon {
        width: 32px;
        height: 32px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        border: 1px solid;
      }

      .terminal-message.user .message-icon {
        background: rgba(0, 212, 255, 0.1);
        border-color: #00d4ff;
      }

      .terminal-message.assistant .message-icon {
        background: rgba(123, 44, 191, 0.1);
        border-color: #7b2cbf;
      }

      .message-icon svg {
        width: 20px;
        height: 20px;
      }

      .terminal-message.user .message-icon svg {
        fill: #00d4ff;
      }

      .terminal-message.assistant .message-icon svg {
        fill: #7b2cbf;
      }

      .message-content {
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        line-height: 1.6;
        position: relative;
        flex: 1;
      }

      /* User Message */
      .terminal-message.user .message-content {
        background: #0f1420;
        border-left: 3px solid #00d4ff;
        color: #00d4ff;
        box-shadow: 0 0 8px rgba(0, 212, 255, 0.2);
      }

      /* Bot Message */
      .terminal-message.assistant .message-content {
        background: #0a0e1a;
        border-left: 3px solid #7b2cbf;
        color: #e2e8f0;
        box-shadow: 0 0 8px rgba(123, 44, 191, 0.3);
      }

      .terminal-message.assistant .message-content::before {
        content: '> ';
        color: #00d4ff;
        font-weight: 700;
      }

      .message-text {
        white-space: pre-wrap;
        word-break: break-word;
      }

      /* ============================================
         SOURCES / CHIPS
         ============================================ */
      .message-sources {
        margin-top: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .source-chip {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        color: #00d4ff;
        padding: 4px 10px;
        font-size: 11px;
        border-radius: 4px;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        transition: all 0.2s;
        box-shadow: 0 0 4px rgba(0, 212, 255, 0.3);
      }

      .source-chip:hover {
        background: rgba(0, 212, 255, 0.2);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.6);
        transform: translateY(-2px);
      }

      .source-chip::before {
        content: '[';
        color: #06d6a0;
      }

      .source-chip::after {
        content: ']';
        color: #06d6a0;
      }

      /* ============================================
         TYPING INDICATOR
         ============================================ */
      .typing-indicator {
        color: #00d4ff;
        font-size: 12px;
        letter-spacing: 2px;
        animation: processingBlink 1.5s ease-in-out infinite;
      }

      @keyframes processingBlink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
      }

      /* ============================================
         INPUT AREA
         ============================================ */
      .terminal-input-container {
        padding: 16px;
        background: rgba(10, 14, 26, 0.95);
        border-top: 1px solid rgba(0, 212, 255, 0.3);
        position: relative;
        z-index: 1;
      }

      .terminal-input-wrapper {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #0a0e1a;
        border: 1px solid #00d4ff;
        padding: 12px 16px;
        border-radius: 4px;
        box-shadow: 0 0 8px rgba(0, 212, 255, 0.2);
      }

      .terminal-input-prefix {
        color: #00d4ff;
        font-weight: 700;
        font-size: 14px;
        flex-shrink: 0;
      }

      .terminal-input {
        flex: 1;
        background: transparent;
        border: none;
        color: #00d4ff;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        outline: none;
        resize: none;
        max-height: 100px;
      }

      .terminal-input::placeholder {
        color: rgba(0, 212, 255, 0.4);
      }

      .terminal-send-btn {
        background: #00d4ff;
        color: #0a0e1a;
        border: none;
        padding: 10px 20px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        cursor: pointer;
        border-radius: 4px;
        transition: all 0.2s;
        flex-shrink: 0;
      }

      .terminal-send-btn:hover:not(:disabled) {
        background: #06d6a0;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.6);
        transform: scale(1.05);
      }

      .terminal-send-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }

      /* ============================================
         SELECTION TOOLBAR
         ============================================ */
      #${CONFIG.widgetId}-selection-toolbar {
        position: fixed;
        background: #0a0e1a;
        border: 1px solid #00d4ff;
        box-shadow: 0 0 8px #00d4ff, inset 0 0 20px rgba(0, 212, 255, 0.1);
        padding: 8px 16px;
        border-radius: 4px;
        display: none;
        align-items: center;
        gap: 12px;
        z-index: 999999;
      }

      #${CONFIG.widgetId}-selection-toolbar.show {
        display: flex;
        animation: toolbarFadeIn 0.2s ease forwards;
      }

      @keyframes toolbarFadeIn {
        from {
          opacity: 0;
          transform: translateY(-10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      .selection-text {
        color: #00d4ff;
        font-size: 11px;
        letter-spacing: 1px;
      }

      .selection-analyze-btn {
        background: transparent;
        border: 1px solid #00d4ff;
        color: #00d4ff;
        padding: 6px 14px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        font-weight: 700;
        cursor: pointer;
        border-radius: 4px;
        transition: all 0.2s;
        box-shadow: 0 0 4px rgba(0, 212, 255, 0.3);
      }

      .selection-analyze-btn:hover {
        background: rgba(0, 212, 255, 0.2);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.6);
      }

      /* ============================================
         WELCOME SCREEN
         ============================================ */
      .terminal-welcome {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        text-align: center;
        padding: 40px 20px;
        position: relative;
        z-index: 1;
      }

      .terminal-welcome-icon {
        width: 64px;
        height: 64px;
        fill: #00d4ff;
        margin-bottom: 20px;
        filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.8));
      }

      .terminal-welcome-title {
        color: #00d4ff;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 3px;
        margin-bottom: 12px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
      }

      .terminal-welcome-text {
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.8;
        margin-bottom: 24px;
      }

      .terminal-suggestions {
        display: flex;
        flex-direction: column;
        gap: 8px;
        width: 100%;
        max-width: 320px;
      }

      .terminal-suggestion {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff;
        padding: 10px 14px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        cursor: pointer;
        text-align: left;
        transition: all 0.2s;
        border-radius: 4px;
      }

      .terminal-suggestion:hover {
        background: rgba(0, 212, 255, 0.15);
        border-color: #00d4ff;
        box-shadow: 0 0 8px rgba(0, 212, 255, 0.4);
        transform: translateX(4px);
      }

      .terminal-suggestion::before {
        content: '> ';
        color: #06d6a0;
        font-weight: 700;
      }

      /* ============================================
         RESPONSIVE
         ============================================ */
      @media (max-width: 480px) {
        #${CONFIG.panelId} {
          bottom: 100px;
          right: 12px;
          left: 12px;
          width: auto;
          max-width: none;
        }
      }
    `;

    const styleSheet = document.createElement('style');
    styleSheet.textContent = styles;
    document.head.appendChild(styleSheet);
  }

  /**
   * Create widget HTML
   */
  function createWidget() {
    // Chat button
    const button = document.createElement('button');
    button.id = CONFIG.buttonId;
    button.innerHTML = `
      <svg viewBox="0 0 24 24">
        <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18 2.5 2.5 0 0 0 10 15.5 2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5"/>
      </svg>
    `;
    button.setAttribute('aria-label', 'Open Terminal Chat');
    button.onclick = toggleWidget;
    document.body.appendChild(button);

    // Chat panel
    const panel = document.createElement('div');
    panel.id = CONFIG.panelId;
    panel.innerHTML = `
      <div class="terminal-header">
        <div class="terminal-title-row">
          <div class="terminal-title">
            <svg viewBox="0 0 24 24">
              <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.488.488 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 0 0-.48-.41h-3.84a.484.484 0 0 0-.48.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.488.488 0 0 0-.59.22L2.09 8.83a.488.488 0 0 0 .12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58a.488.488 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.27.41.48.41h3.84c.24 0 .44-.17.48-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32a.488.488 0 0 0-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6"/>
            </svg>
            <span>HUMANOID.AI</span>
          </div>
          <button class="terminal-clear-btn" id="terminal-clear-btn">[CLEAR CHAT]</button>
        </div>
        <button class="terminal-close" id="terminal-close" aria-label="Close chat">×</button>
        <div class="terminal-subtitle">SYSTEM READY</div>
      </div>
      <div class="terminal-messages" id="terminal-messages">
        <div class="terminal-welcome" id="terminal-welcome">
          <svg class="terminal-welcome-icon" viewBox="0 0 24 24">
            <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18 2.5 2.5 0 0 0 10 15.5 2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5"/>
          </svg>
          <div class="terminal-welcome-title">TERMINAL ACCESS</div>
          <div class="terminal-welcome-text">
            AI ROBOTICS TUTOR SYSTEM<br>
            ENTER YOUR QUERY BELOW
          </div>
          <div class="terminal-suggestions">
            <button class="terminal-suggestion" onclick="sendSuggestion('What is ROS2?')">What is ROS2?</button>
            <button class="terminal-suggestion" onclick="sendSuggestion('Explain humanoid robotics')">Explain humanoid robotics</button>
            <button class="terminal-suggestion" onclick="sendSuggestion('What is NVIDIA Isaac?')">What is NVIDIA Isaac?</button>
            <button class="terminal-suggestion" onclick="sendSuggestion('Hardware requirements')">Hardware requirements</button>
          </div>
        </div>
      </div>
      <div class="terminal-input-container">
        <div class="terminal-input-wrapper">
          <span class="terminal-input-prefix">&gt;&gt;</span>
          <textarea 
            class="terminal-input" 
            id="terminal-input" 
            placeholder="ENTER QUERY_"
            rows="1"
            onkeydown="handleInputKeydown(event)"
          ></textarea>
          <button class="terminal-send-btn" id="terminal-send" onclick="sendMessage()" disabled>EXECUTE</button>
        </div>
      </div>
    `;
    document.body.appendChild(panel);

    // Selection toolbar
    const toolbar = document.createElement('div');
    toolbar.id = CONFIG.widgetId + '-selection-toolbar';
    toolbar.innerHTML = `
      <span class="selection-text">TEXT SELECTED</span>
      <button class="selection-analyze-btn" onclick="analyzeSelection()">[ANALYZE SELECTION]</button>
    `;
    document.body.appendChild(toolbar);

    // Add selection listener
    document.addEventListener('mouseup', handleTextSelection);
  }

  /**
   * Toggle widget open/close
   */
  window.toggleWidget = function() {
    isOpen = !isOpen;
    const panel = document.getElementById(CONFIG.panelId);
    const button = document.getElementById(CONFIG.buttonId);
    
    if (isOpen) {
      panel.classList.add('open');
      button.classList.add('open');
      setTimeout(() => {
        const input = document.getElementById('terminal-input');
        if (input) input.focus();
      }, 100);
    } else {
      panel.classList.remove('open');
      button.classList.remove('open');
    }
  };

  /**
   * Clear chat
   */
  window.clearChat = function() {
    const messagesContainer = document.getElementById('terminal-messages');
    const welcomeScreen = document.getElementById('terminal-welcome');

    // Remove all messages from DOM
    if (messagesContainer) {
      messagesContainer.innerHTML = '';
    }

    // Show welcome screen
    if (welcomeScreen) {
      welcomeScreen.style.display = 'flex';
    }

    // Reset chat history
    if (window.chatMessages) {
      window.chatMessages = [];
    }
  };

  /**
   * Handle input keydown
   */
  window.handleInputKeydown = function(event) {
    const input = document.getElementById('terminal-input');
    const sendBtn = document.getElementById('terminal-send');
    
    if (input) {
      // Auto-resize
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 100) + 'px';
      
      // Enable/disable button
      if (sendBtn) {
        sendBtn.disabled = !input.value.trim();
      }
    }
    
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  /**
   * Send suggestion
   */
  window.sendSuggestion = function(text) {
    const input = document.getElementById('terminal-input');
    if (input) {
      input.value = text;
      input.focus();
      const sendBtn = document.getElementById('terminal-send');
      if (sendBtn) sendBtn.disabled = false;
      sendMessage();
    }
  };

  /**
   * Handle text selection
   */
  function handleTextSelection() {
    const selection = window.getSelection();
    const toolbar = document.getElementById(CONFIG.widgetId + '-selection-toolbar');
    
    if (selection && selection.toString().trim().length > 0) {
      selectedText = selection.toString().trim();
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      
      toolbar.style.top = (rect.top - 50) + 'px';
      toolbar.style.left = (rect.left + rect.width / 2 - 100) + 'px';
      toolbar.classList.add('show');
    } else {
      toolbar.classList.remove('show');
    }
  }

  /**
   * Analyze selected text
   */
  window.analyzeSelection = function() {
    if (selectedText) {
      const input = document.getElementById('terminal-input');
      if (input) {
        input.value = 'Analyze this: ' + selectedText;
        const sendBtn = document.getElementById('terminal-send');
        if (sendBtn) sendBtn.disabled = false;
      }
      const toolbar = document.getElementById(CONFIG.widgetId + '-selection-toolbar');
      if (toolbar) toolbar.classList.remove('show');
      if (!isOpen) toggleWidget();
    }
  };

  /**
   * Add message to UI with ICONS
   */
  function addMessageToUI(role, content, sources = []) {
    const messagesContainer = document.getElementById('terminal-messages');
    const welcomeScreen = document.getElementById('terminal-welcome');
    
    if (welcomeScreen) {
      welcomeScreen.style.display = 'none';
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `terminal-message ${role}`;
    
    // User icon (person silhouette)
    const userIcon = `
      <svg viewBox="0 0 24 24">
        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
      </svg>
    `;
    
    // Robot icon
    const robotIcon = `
      <svg viewBox="0 0 24 24">
        <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18 2.5 2.5 0 0 0 10 15.5 2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5"/>
      </svg>
    `;
    
    let sourcesHTML = '';
    if (sources && sources.length > 0) {
      sourcesHTML = `
        <div class="message-sources">
          ${sources.map(s => `
            <a href="${s.source_url}" target="_blank" rel="noopener" class="source-chip">
              ${s.chapter_name || s.page_title || 'SOURCE'}
            </a>
          `).join('')}
        </div>
      `;
    }

    messageDiv.innerHTML = `
      <div class="message-icon">
        ${role === 'user' ? userIcon : robotIcon}
      </div>
      <div class="message-content">
        <div class="message-text">${escapeHtml(content)}</div>
        ${sourcesHTML}
      </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  /**
   * Add typing indicator
   */
  function addTypingIndicator() {
    const messagesContainer = document.getElementById('terminal-messages');
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'terminal-message assistant';
    typingDiv.id = 'terminal-typing';
    typingDiv.innerHTML = `
      <div class="message-icon">
        <svg viewBox="0 0 24 24">
          <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18 2.5 2.5 0 0 0 10 15.5 2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5"/>
        </svg>
      </div>
      <div class="message-content">
        <div class="typing-indicator">PROCESSING...</div>
      </div>
    `;

    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  /**
   * Remove typing indicator
   */
  function removeTypingIndicator() {
    const typing = document.getElementById('terminal-typing');
    if (typing) {
      typing.remove();
    }
  }

  /**
   * Send message
   */
  window.sendMessage = async function() {
    const input = document.getElementById('terminal-input');
    const sendBtn = document.getElementById('terminal-send');
    const text = input?.value.trim();

    if (!text || isLoading) return;

    // Add user message
    addMessageToUI('user', text);
    input.value = '';
    input.style.height = 'auto';
    if (sendBtn) sendBtn.disabled = true;
    isLoading = true;

    // Add typing indicator
    addTypingIndicator();

    try {
      const response = await fetch(CONFIG.apiBaseUrl + '/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: text,
          selected_text: null,
        }),
      });

      removeTypingIndicator();

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      addMessageToUI('assistant', data.data?.answer || data.response || 'No response received', data.data?.sources || []);
      
    } catch (error) {
      removeTypingIndicator();
      addMessageToUI('assistant', 'ERROR: Connection failed. Ensure backend is running at ' + CONFIG.apiBaseUrl);
      console.error('Widget error:', error);
    } finally {
      isLoading = false;
      input?.focus();
    }
  };

  /**
   * Escape HTML
   */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Initialize widget
   */
  function init() {
    createStyles();
    createWidget();
    
    // Add event listeners after widget is created
    setTimeout(() => {
      const closeBtn = document.getElementById('terminal-close');
      const clearBtn = document.getElementById('terminal-clear-btn');
      
      if (closeBtn) {
        closeBtn.addEventListener('click', toggleWidget);
      }
      
      if (clearBtn) {
        clearBtn.addEventListener('click', clearChat);
      }
    }, 100);
    
    console.log('[TERMINAL] Chat widget initialized');
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
