// Service to connect to the RAG backend
class ChatbotService {
  constructor() {
    // In a real implementation, this would connect to your actual RAG backend
    // For now, this is a placeholder that simulates the API
    this.baseUrl = process.env.RAG_API_URL || 'http://localhost:8000/api';
  }

  async sendMessage(message) {
    try {
      // In the actual implementation, this would call your RAG backend
      // that connects to Qdrant and Cohere for retrieval and generation
      const response = await fetch(`${this.baseUrl}/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: message,
          user_id: 'frontend-user',
          top_k: 5,
          // In a real implementation, the backend would handle:
          // 1. Converting the query to embeddings using Cohere
          // 2. Querying Qdrant for relevant textbook chunks
          // 3. Generating a response using those chunks
          // 4. Following constitution rules (no hallucination, etc.)
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error sending message to chatbot API:', error);
      // Fallback to simple logic if backend is not available
      return this.getFallbackResponse(message);
    }
  }

  // Fallback response function when backend is not available
  getFallbackResponse(message) {
    const lowerMsg = message.toLowerCase();

    // Check if the query relates to textbook topics
    if (lowerMsg.includes('ros') || lowerMsg.includes('ros 2')) {
      return {
        response: "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.",
        confidence: "HIGH",
        sources: ["/docs/intro", "/docs/module-1-ros2"]
      };
    } else if (lowerMsg.includes('gazebo')) {
      return {
        response: "Gazebo is a robot simulator that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. It's commonly used for testing robotic algorithms before deploying them on real hardware.",
        confidence: "HIGH",
        sources: ["/docs/module-2-simulation", "/docs/module-2-simulation/week-4"]
      };
    } else if (lowerMsg.includes('unity')) {
      return {
        response: "Unity is a powerful game engine that can be used for robotics simulation and visualization. It provides excellent graphics capabilities and can be integrated with ROS 2 for advanced robotic simulations.",
        confidence: "HIGH",
        sources: ["/docs/module-2-simulation", "/docs/module-2-simulation/week-5"]
      };
    } else if (lowerMsg.includes('nvidia') || lowerMsg.includes('isaac')) {
      return {
        response: "NVIDIA Isaac is a robotics platform that includes simulation tools, AI frameworks, and hardware accelerators for developing intelligent robots. It provides tools for perception, navigation, and manipulation tasks.",
        confidence: "HIGH",
        sources: ["/docs/module-3-ai-brain", "/docs/module-3-ai-brain/week-6"]
      };
    } else if (lowerMsg.includes('vision') || lowerMsg.includes('language') || lowerMsg.includes('action') || lowerMsg.includes('vla')) {
      return {
        response: "Vision-Language-Action (VLA) models enable robots to understand and respond to natural language commands while perceiving the visual world. These models combine computer vision and natural language processing to perform complex robotic tasks.",
        confidence: "HIGH",
        sources: ["/docs/module-4-vla", "/docs/module-4-vla/week-9"]
      };
    } else if (lowerMsg.includes('humanoid')) {
      return {
        response: "A humanoid robot is a robot designed to resemble the human form, both in structure and functionality. The context emphasizes the development of an autonomous humanoid system that integrates the Vision-Language-Action (VLA) pipeline, enabling the robot to understand voice commands, perceive its environment, and execute complex tasks.",
        confidence: "HIGH",
        sources: ["/docs/module-4-vla", "/docs/module-4-vla/week-9"]
      };
    } else if (lowerMsg.includes('module') || lowerMsg.includes('topic') || lowerMsg.includes('chapter')) {
      return {
        response: "Our textbook is organized into 4 main modules: 1) The Robotic Nervous System (ROS 2), 2) The Digital Twin (Gazebo & Unity), 3) The AI-Robot Brain (NVIDIA Isaac), and 4) Vision-Language-Action (VLA). Each module builds upon the previous one.",
        confidence: "HIGH",
        sources: ["/docs/book-intro", "/docs/intro"]
      };
    } else {
      // If topic not found in known keywords, return the default response
      return {
        response: "This topic is not covered in the book yet.",
        confidence: "NONE",
        sources: []
      };
    }
  }
}

export default new ChatbotService();