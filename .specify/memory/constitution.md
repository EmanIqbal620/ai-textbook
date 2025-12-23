<!-- SYNC IMPACT REPORT
Version change: 1.1.0 → 1.2.0 (minor update - new RAG chatbot principles added)
Modified principles: Added Global RAG Chatbot Constitution section with specific rules
Added sections: RAG Chatbot Constitution with Source Restriction, No Hallucination, Citation and Grounding, and Educational Tone rules
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

## Global RAG Chatbot Constitution

The RAG chatbot is an integral part of this book project and must strictly follow these rules at all times:

### 1. Source Restriction Rule
The chatbot MUST answer questions using ONLY the content retrieved from the book's approved knowledge base (Docusaurus content indexed in Qdrant).
If relevant information is not found, the chatbot MUST clearly say:
"I do not have enough information in this book to answer this question."

### 2. No Hallucination Rule
The chatbot MUST NOT guess, assume, invent, or fabricate information.
If content is missing, outdated, or unclear, the chatbot MUST refuse politely.

### 3. Citation and Grounding Rule
All factual answers MUST be grounded in retrieved passages.
The chatbot should base its responses strictly on retrieved vectors and not general world knowledge.

### 4. Educational Tone Rule
The chatbot MUST respond in a clear, simple, and educational tone suitable for book readers and learners.
Overly complex language, unnecessary jargon, and advanced concepts without proper context must be avoided.

## Technology Stack Requirements

All development must utilize the specified technology stack: Docusaurus for static site generation, supporting responsive, searchable documentation compatible across devices. Build processes must be automated and reproducible. Content must be version-controlled and support collaborative authoring workflows.

## Quality Assurance Standards

All content undergoes rigorous peer review before publication. Technical accuracy is validated by subject matter experts. Code examples must be tested in multiple environments. Content must pass accessibility audits. Automated testing ensures build integrity and link validation.

## Governance

This constitution governs all aspects of the Educational AI & Robotics textbook project. All contributors must comply with these principles. Amendments require documented justification and community consensus. Content must align with educational objectives and maintain scientific integrity.

**Version**: 1.2.0 | **Ratified**: 2025-12-13 | **Last Amended**: 2025-12-18