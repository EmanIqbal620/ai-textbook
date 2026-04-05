---
description: "Task list for RAG Chatbot implementation"
---

# Tasks: RAG Chatbot for Textbook

**Input**: Design documents from `/specs/1-rag-chatbot/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Backend**: `backend/src/`, `backend/tests/`
- **Frontend**: `frontend/src/`, `frontend/tests/`
- Adjust based on plan.md structure

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /sp.tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create backend project structure with FastAPI dependencies in backend/
- [x] T002 Create frontend project structure with React dependencies in frontend/
- [ ] T003 [P] Configure linting and formatting tools for both backend and frontend

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

**Purpose**: Core infrastructure that enables RAG functionality and chatbot compliance

- [x] T004 Setup Qdrant vector database connection in backend/src/services/rag/vector_store.py
- [x] T005 [P] Implement text embedding service using Cohere/OpenAI in backend/src/services/rag/embedding.py
- [x] T006 [P] Setup API routing and middleware structure in backend/src/api/
- [x] T007 Create base data models following data-model.md in backend/src/models/
- [x] T008 Configure error handling and logging infrastructure in backend/src/utils/
- [x] T009 Setup environment configuration management in backend/src/config.py
- [x] T010 [P] Implement constitution compliance validation service in backend/src/services/chat/validation.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Ask Questions and Get Book-Based Answers (Priority: P1) 🎯 MVP

**Goal**: Enable students to ask questions about textbook content and receive answers based only on book material

**Independent Test**: User can submit a question and receive an answer that is grounded in textbook content only, following all constitution rules

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T011 [P] [US1] Contract test for question endpoint in backend/tests/contract/test_chat.py
- [ ] T012 [P] [US1] Integration test for complete question-answer flow in backend/tests/integration/test_chat_flow.py

### Implementation for User Story 1

- [x] T013 [P] [US1] Create Question model in backend/src/models/question.py
- [x] T014 [P] [US1] Create Response model in backend/src/models/response.py
- [x] T015 [US1] Implement RAG retrieval service in backend/src/services/rag/retrieval.py
- [x] T016 [US1] Implement chat response generation service in backend/src/services/chat/generation.py
- [x] T017 [US1] Create /api/chat/{sessionId}/question endpoint in backend/src/api/chat.py
- [x] T018 [US1] Add constitution compliance validation to response generation
- [x] T019 [US1] Implement frontend chat interface in frontend/src/components/ChatInterface.jsx
- [x] T020 [US1] Add basic frontend API service for chat in frontend/src/services/chatService.js

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Multi-turn Conversation Support (Priority: P2)

**Goal**: Allow students to have conversations with the chatbot, asking follow-up questions without repeating context

**Independent Test**: User can ask a follow-up question related to previous conversation and receive a relevant answer that considers previous context

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [ ] T021 [P] [US2] Contract test for conversation history endpoint in backend/tests/contract/test_conversation.py
- [ ] T022 [P] [US2] Integration test for multi-turn conversation in backend/tests/integration/test_conversation_flow.py

### Implementation for User Story 2

- [ ] T023 [P] [US2] Create ConversationSession model in backend/src/models/conversation.py
- [ ] T024 [P] [US2] Create Message model in backend/src/models/message.py
- [ ] T025 [US2] Implement conversation context management service in backend/src/services/chat/context.py
- [ ] T026 [US2] Create /api/chat/{sessionId}/history endpoint in backend/src/api/chat.py
- [ ] T027 [US2] Update chat generation service to consider conversation history
- [ ] T028 [US2] Enhance frontend to display conversation history in frontend/src/components/ChatInterface.jsx
- [ ] T029 [US2] Add frontend state management for conversation context in frontend/src/hooks/useConversation.js

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Source Attribution and Confidence (Priority: P3)

**Goal**: Provide transparency by indicating which parts of the book were used to generate responses

**Independent Test**: When chatbot provides an answer, it includes appropriate attribution to the source material without exposing internal retrieval steps

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [ ] T030 [P] [US3] Contract test for source attribution in responses in backend/tests/contract/test_attribution.py
- [ ] T031 [P] [US3] Integration test for source attribution functionality in backend/tests/integration/test_attribution.py

### Implementation for User Story 3

- [ ] T032 [P] [US3] Update Response model to include source attribution fields in backend/src/models/response.py
- [ ] T033 [US3] Implement source attribution service in backend/src/services/rag/attribution.py
- [ ] T034 [US3] Update chat generation service to include source attribution
- [ ] T035 [US3] Enhance frontend to display source attribution in frontend/src/components/ChatInterface.jsx
- [ ] T036 [US3] Add UI elements for source attribution in frontend/src/components/SourceAttribution.jsx

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T037 [P] Documentation updates in docs/
- [ ] T038 Code cleanup and refactoring
- [ ] T039 Performance optimization across all stories
- [ ] T040 [P] Additional unit tests in backend/tests/unit/ and frontend/tests/
- [ ] T041 Security hardening
- [ ] T042 Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence