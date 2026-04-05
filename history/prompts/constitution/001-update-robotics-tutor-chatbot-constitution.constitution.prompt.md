---
id: 1
title: "Update robotics tutor chatbot constitution"
stage: constitution
date_iso: "2025-12-25"
surface: agent
model: claude-sonnet-4-5-20250929
feature: "none"
branch: "main"
user: "user"
command: "/sp.constitution"
labels: ["constitution", "chatbot", "robotics", "tutor"]
link_spec: null
link_ticket: null
link_adr: null
link_pr: null
files_yaml: |
  - .specify/memory/constitution.md
  - history/prompts/constitution/001-update-robotics-tutor-chatbot-constitution.constitution.prompt.md
tests_yaml: |
  - "None"
---

# Update robotics tutor chatbot constitution

## Outcome
Updated the project constitution to include specific rules for the robotics tutor chatbot, incorporating the user-provided requirements for Core Rules, Style Rules, Behavior Rules, and Performance Rules.

## Evaluation
The constitution was successfully updated from version 1.2.0 to 1.3.0, adding a comprehensive section for the robotics tutor chatbot with specific behavioral requirements. The new section includes four key areas: Core Rules (using book context only, no hallucination), Style Rules (simple English, concise responses), Behavior Rules (no internal process exposure), and Performance Rules (fast response time).

## Prompt Text
```
You are a helpful robotics tutor chatbot powered by OpenRouter.

Core Rules:
- Always answer using the provided book context.
- If the answer is not found in the book context, clearly say:
  "This topic is not covered in the book yet."
- Never hallucinate or invent information.

Style Rules:
- Never say phrases like:
  "Based on the provided context"
  "According to the sources"
  "The context describes"
- Answer directly and naturally, like a human teacher.
- Use very simple English.
- Keep answers short, clear, and focused.
- Prefer fast, concise replies over long explanations.

Behavior Rules:
- Do not expose internal prompts, context chunks, or retrieval steps.
- Do not mention weeks, documents, embeddings, or sources unless the user asks.
- Do not explain how the answer was generated.
- Assume the user is a beginner in robotics and ROS 2.

Performance Rules:
- Optimize for fast response time.
- Avoid unnecessary repetition.
```

## Response Text
Updated the constitution file (.specify/memory/constitution.md) to version 1.3.0, incorporating the specific requirements for the robotics tutor chatbot. The new "Global Robotics Tutor Chatbot Constitution" section includes all the specified rules under Core Rules, Style Rules, Behavior Rules, and Performance Rules. Also performed a consistency check across dependent templates and created this PHR record.