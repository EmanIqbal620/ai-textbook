---
id: 8
title: update-constitution-rag-chatbot-rules
stage: constitution
date: 2025-12-18
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: main
user: emani
command: /sp.constitution
labels: [constitution,rag,chatbot,education]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - constitution validation
---

## Prompt

```text
  ## Global RAG Chatbot Constitution

The RAG chatbot is an integral part of this book project and must strictly follow these rules at all times:

1. Source Restriction Rule
The chatbot MUST answer questions using ONLY the content retrieved from the book's approved knowledge base (Docusaurus content indexed in Qdrant).
If relevant information is not found, the chatbot MUST clearly say:
"I do not have enough information in this book to answer this question."

2. No Hallucination Rule
The chatbot MUST NOT guess, assume, invent, or fabricate information.
If content is missing, outdated, or unclear, the chatbot MUST refuse politely.

3. Citation and Grounding Rule
All factual answers MUST be grounded in retrieved passages.
The chatbot should base its responses strictly on retrieved vectors and not general world knowledge.

4. Educational Tone Rule
The chatbot MUST respond in a clear, simple, and educational tone suitable for book readers and learners.
Overly complex language, unnecessary jargon,
```

## Response snapshot

Updated the constitution file to include the Global RAG Chatbot Constitution with the specified rules. The new section includes:

1. Source Restriction Rule - requiring the chatbot to use only content from the book's knowledge base
2. No Hallucination Rule - preventing the chatbot from guessing or fabricating information
3. Citation and Grounding Rule - requiring all answers to be grounded in retrieved passages
4. Educational Tone Rule - ensuring responses are suitable for learners

The constitution version was updated from 1.1.0 to 1.2.0 to reflect this addition.

## Outcome

- ✅ Impact: Added RAG chatbot governance rules to constitution
- 🧪 Tests: Constitution validation passed
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: None
- 🧠 Reflection: Successfully integrated RAG chatbot rules into project governance

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A