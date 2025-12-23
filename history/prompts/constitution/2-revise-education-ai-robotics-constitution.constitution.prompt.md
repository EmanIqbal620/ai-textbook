---
id: 2
title: Revise Education AI Robotics Constitution
stage: constitution
date: 2025-12-14
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution, ai-robotics, education, docusaurus]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
---

## Prompt

```text
Create a clear and practical set of principles for building an Educational AI & Robotics textbook using Docusaurus.

The constitution must follow these principles:

1. Learning Structure
   - Start from basic concepts and gradually move to advanced topics.
   - Each chapter must build on previous knowledge.
   - Include clear learning objectives at the start of each chapter.

2. Practical and Hands-On Approach
   - Provide real, working examples instead of only theory.
   - Include code snippets, commands, and small projects.
   - Encourage experimentation and problem-solving.

3. Robotics Platforms and Tools Coverage
   - Cover ROS 2 concepts, nodes, topics, services, and actions.
   - Explain simulation using Gazebo.
   - Include Unity for visualization and interaction.
   - Cover NVIDIA Isaac for advanced robotics simulation and AI integration.

4. Student-Friendly Explanations
   - Use simple and easy language.
   - Explain technical terms with examples.
   - Connect topics with real-world applications.
```

## Response snapshot

Revised the constitution to reflect clear and practical principles for building an Educational AI & Robotics textbook using Docusaurus. The updated constitution now includes specific principles for progressive learning structure, practical hands-on approach, comprehensive robotics platforms coverage (ROS 2, Gazebo, Unity, NVIDIA Isaac), and student-friendly explanations. The document maintains focus on educational objectives while leveraging Docusaurus capabilities for effective content delivery.

## Outcome

- ✅ Impact: Updated constitution with specific educational robotics principles
- 🧪 Tests: No tests required for constitution update
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Continue with textbook content development
- 🧠 Reflection: Constitution now properly aligned with educational robotics requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Begin implementing textbook modules based on constitution