class ChatService {
  constructor(baseURL = '') {
    this.baseURL = baseURL;
  }

  async startSession() {
    try {
      const response = await fetch(`${this.baseURL}/api/chat/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error starting session:', error);
      throw error;
    }
  }

  async askQuestion(sessionId, question) {
    try {
      const response = await fetch(`${this.baseURL}/api/chat/${sessionId}/question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error asking question:', error);
      throw error;
    }
  }

  async getConversationHistory(sessionId) {
    try {
      const response = await fetch(`${this.baseURL}/api/chat/${sessionId}/history`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error getting conversation history:', error);
      throw error;
    }
  }

  async endSession(sessionId) {
    try {
      const response = await fetch(`${this.baseURL}/api/chat/${sessionId}/end`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error ending session:', error);
      throw error;
    }
  }
}

export default new ChatService();