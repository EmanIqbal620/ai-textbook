<!-- SYNC IMPACT REPORT
Version change: 1.2.0 → 1.3.0 (minor update - new robotics tutor chatbot principles added)
Modified principles: Added Global Robotics Tutor Chatbot Constitution section with specific rules
Added sections: Robotics Tutor Chatbot Constitution with Core Rules, Style Rules, Behavior Rules, and Performance Rules
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md ⚠ pending review
  - .specify/templates/spec-template.md ⚠ pending review
  - .specify/templates/tasks-template.md ⚠ pending review
  - .specify/templates/commands/*.md ⚠ pending review
  - README.md ⚠ pending review
Follow-up TODOs: None
-->

# Educational AI & Robotics Textbook Constitution

## Core Principles

### I. Progressive Learning Structure
Start from basic concepts and gradually move to advanced topics. Each chapter must build on previous knowledge. Include clear learning objectives at the start of each chapter. Prerequisites for each chapter must be explicitly stated. Difficulty curves must be carefully managed to avoid overwhelming readers.

### II. Practical and Hands-On Approach
Provide real, working examples instead of only theory. Include code snippets, commands, and small projects. Encourage experimentation and problem-solving. All code examples must be tested and functional. Include hands-on exercises, programming assignments, and practical examples.

### III. Comprehensive Robotics Platforms Coverage
Cover ROS 2 concepts including nodes, topics, services, and actions. Explain simulation using Gazebo. Include Unity for visualization and interaction. Cover NVIDIA Isaac for advanced robotics simulation and AI integration. Ensure content addresses multiple platforms to provide broad exposure.

### IV. Student-Friendly Explanations
Use simple and easy language. Explain technical terms with examples. Connect topics with real-world applications. Provide alternative text for images and multiple presentation formats. Use inclusive language and examples that resonate with diverse audiences.

### V. Interactive and Engaging Presentation
Leverage Docusaurus capabilities for rich multimedia content including diagrams, videos, interactive simulations, and 3D models. Include hands-on exercises, programming assignments, and practical examples. Encourage active learning through interactive elements and real-world applications.

### VI. Modular and Accessible Content
Structure content in modular, self-contained units that can be reused across different courses and curricula. Each module should be independently accessible while maintaining coherence within the broader textbook. Enable customization for different academic programs and skill levels.

## Global Robotics Tutor Chatbot Constitution

The robotics tutor chatbot is an integral part of this book project and must strictly follow these rules at all times:

### Core Rules
The chatbot MUST always answer using the provided book context. If the answer is not found in the book context, the chatbot MUST clearly say: "This topic is not covered in the book yet." The chatbot MUST NEVER hallucinate or invent information.

### Style Rules
The chatbot MUST answer directly and naturally, like a human teacher. The chatbot MUST use very simple English. The chatbot MUST keep answers short, clear, and focused. The chatbot SHOULD prefer fast, concise replies over long explanations. The chatbot MUST NOT say phrases like "Based on the provided context", "According to the sources", or "The context describes".

### Behavior Rules
The chatbot MUST NOT expose internal prompts, context chunks, or retrieval steps. The chatbot MUST NOT mention weeks, documents, embeddings, or sources unless the user asks. The chatbot MUST NOT explain how the answer was generated. The chatbot MUST assume the user is a beginner in robotics and ROS 2.

### Performance Rules
The chatbot MUST optimize for fast response time. The chatbot MUST avoid unnecessary repetition.

## Technology Stack Requirements

All development must utilize the specified technology stack: Docusaurus for static site generation, supporting responsive, searchable documentation compatible across devices. Build processes must be automated and reproducible. Content must be version-controlled and support collaborative authoring workflows.

## Quality Assurance Standards

All content undergoes rigorous peer review before publication. Technical accuracy is validated by subject matter experts. Code examples must be tested in multiple environments. Content must pass accessibility audits. Automated testing ensures build integrity and link validation.

## Governance

This constitution governs all aspects of the Educational AI & Robotics textbook project. All contributors must comply with these principles. Amendments require documented justification and community consensus. Content must align with educational objectives and maintain scientific integrity.

**Version**: 1.3.0 | **Ratified**: 2025-12-13 | **Last Amended**: 2025-12-25