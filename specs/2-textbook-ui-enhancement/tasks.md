---
description: "Task list for Textbook UI Enhancement implementation"
---

# Tasks: Textbook UI Enhancement

**Input**: Design documents from `/specs/2-textbook-ui-enhancement/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Textbook UI**: `humanoid-robotics-textbook/src/`, `humanoid-robotics-textbook/static/`
- **Docusaurus**: `humanoid-robotics-textbook/`, `humanoid-robotics-textbook/docs/`
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

- [x] T001 Install and configure Docusaurus 3.x dependencies in humanoid-robotics-textbook/
- [x] T002 Set up Tailwind CSS and styled-components dependencies for styling
- [x] T003 [P] Install UI libraries: React Icons, Framer Motion for animations
- [x] T004 Configure Algolia DocSearch integration dependencies
- [x] T005 Set up testing framework: Jest, React Testing Library, Cypress

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

**Purpose**: Core UI infrastructure that enables all enhancement features

- [x] T006 Create base theme context and styling utilities in humanoid-robotics-textbook/src/styles/
- [x] T007 [P] Implement dark/light mode toggle functionality in humanoid-robotics-textbook/src/contexts/ThemeContext.js
- [x] T008 Create base component library in humanoid-robotics-textbook/src/components/UI/
- [x] T009 Set up API service layer for module data in humanoid-robotics-textbook/src/services/
- [x] T010 Configure responsive design utilities and breakpoints in humanoid-robotics-textbook/src/styles/
- [x] T011 [P] Create hooks for user preferences in humanoid-robotics-textbook/src/hooks/
- [x] T012 Implement accessibility utilities in humanoid-robotics-textbook/src/utils/accessibility.js

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Access Modern Textbook Interface (Priority: P1) 🎯 MVP

**Goal**: Enable students to access the textbook through a modern, visually appealing interface that provides clear navigation options and an intuitive reading experience

**Independent Test**: Student can land on the front page, see clear navigation options, and choose to either start reading or view the curriculum overview

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T013 [P] [US1] Unit test for FrontPage component in humanoid-robotics-textbook/src/components/FrontPage/__tests__/FrontPage.test.js
- [ ] T014 [P] [US1] Integration test for front page navigation in humanoid-robotics-textbook/src/components/FrontPage/__tests__/navigation.test.js

### Implementation for User Story 1

- [x] T015 [P] [US1] Create FrontPage component with soft pastel backgrounds in humanoid-robotics-textbook/src/components/FrontPage/FrontPage.jsx
- [x] T016 [P] [US1] Create "Start Reading" and "Curriculum Overview" buttons with hover animations in humanoid-robotics-textbook/src/components/FrontPage/Buttons.jsx
- [x] T017 [US1] Implement clean typography and learning outcomes preview in humanoid-robotics-textbook/src/components/FrontPage/Typography.jsx
- [x] T018 [US1] Create front page layout with pastel color scheme in humanoid-robotics-textbook/src/pages/index.js
- [x] T019 [US1] Add subtle hover animations for buttons using Framer Motion in humanoid-robotics-textbook/src/components/FrontPage/Buttons.jsx
- [x] T020 [US1] Implement curriculum overview page structure in humanoid-robotics-textbook/src/pages/curriculum-overview.jsx
- [x] T021 [US1] Create module listing component for curriculum overview in humanoid-robotics-textbook/src/components/CurriculumOverview/ModuleList.jsx

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Navigate Through Content with Enhanced Sidebar (Priority: P2)

**Goal**: Allow students to easily navigate between different modules and sections using an intuitive sidebar that shows their progress

**Independent Test**: Student can use the sticky floating sidebar to see module icons with hover summaries, track reading progress, and navigate to previous/next sections

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T022 [P] [US2] Unit test for NavigationSidebar component in humanoid-robotics-textbook/src/components/Navigation/__tests__/NavigationSidebar.test.js
- [ ] T023 [P] [US2] Integration test for sidebar navigation in humanoid-robotics-textbook/src/components/Navigation/__tests__/navigation.test.js

### Implementation for User Story 2

- [x] T024 [P] [US2] Create NavigationSidebar component with sticky positioning in humanoid-robotics-textbook/src/components/Navigation/NavigationSidebar.jsx
- [x] T025 [P] [US2] Implement module icons with color coding (ROS 2 → 🤖, Gazebo → ⚙️, etc.) in humanoid-robotics-textbook/src/components/Navigation/ModuleIcon.jsx
- [x] T026 [US2] Add hover functionality to show module name and topic summary in humanoid-robotics-textbook/src/components/Navigation/ModuleHover.jsx
- [x] T027 [US2] Implement click functionality to expand module content in humanoid-robotics-textbook/src/components/Navigation/ModuleExpand.jsx
- [x] T028 [US2] Create scroll progress indicator in humanoid-robotics-textbook/src/components/Navigation/ProgressIndicator.jsx
- [x] T029 [US2] Implement previous/next navigation buttons in humanoid-robotics-textbook/src/components/Navigation/PrevNextButtons.jsx
- [x] T030 [US2] Add navigation state management in humanoid-robotics-textbook/src/components/Navigation/NavigationState.js

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Access Interactive Features and Search (Priority: P3)

**Goal**: Provide students with access to the RAG chatbot and search functionality to find specific content within the textbook

**Independent Test**: Student can access the floating RAG chatbot at any time and use the search functionality to find relevant content

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T031 [P] [US3] Unit test for RAGChatbot component in humanoid-robotics-textbook/src/components/Chatbot/__tests__/RAGChatbot.test.js
- [ ] T032 [P] [US3] Unit test for Search functionality in humanoid-robotics-textbook/src/components/Search/__tests__/Search.test.js

### Implementation for User Story 3

- [x] T033 [P] [US3] Create floating RAG chatbot icon component in humanoid-robotics-textbook/src/components/Chatbot/RAGChatbot.jsx
- [x] T034 [P] [US3] Implement contextual side panel for chatbot interaction in humanoid-robotics-textbook/src/components/Chatbot/ChatPanel.jsx
- [x] T035 [US3] Integrate Algolia DocSearch functionality in humanoid-robotics-textbook/src/components/Search/Search.jsx
- [x] T036 [US3] Create search results display component in humanoid-robotics-textbook/src/components/Search/SearchResults.jsx
- [x] T037 [US3] Implement breadcrumbs navigation in humanoid-robotics-textbook/src/components/Navigation/Breadcrumbs.jsx
- [x] T038 [US3] Add table of contents per page functionality in humanoid-robotics-textbook/src/components/Navigation/TableOfContents.jsx
- [x] T039 [US3] Connect chatbot to existing RAG backend API in humanoid-robotics-textbook/src/services/chatbotService.js

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Module Page Enhancement

**Goal**: Enhance individual module pages with requested UI features

**Independent Test**: Each module page displays with proper structure (title, icon, examples, diagrams, etc.)

- [x] T040 [P] Create module header component with title, icon, description in humanoid-robotics-textbook/src/components/Modules/ModuleHeader.jsx
- [x] T041 [P] Implement tabs for Python/C++ examples in humanoid-robotics-textbook/src/components/Modules/CodeTabs.jsx
- [x] T042 Create interactive diagrams component (Mermaid, images) in humanoid-robotics-textbook/src/components/Modules/Diagrams.jsx
- [x] T043 Implement tips/warnings with soft accent colors in humanoid-robotics-textbook/src/components/Modules/TipsWarnings.jsx
- [x] T044 Add scroll-triggered animations for module highlights in humanoid-robotics-textbook/src/components/Modules/ScrollAnimations.jsx
- [x] T045 Create collapsible cards for Glossary & Resources in humanoid-robotics-textbook/src/components/Modules/CollapsibleCards.jsx

---

## Phase 7: Additional Sections Enhancement

**Goal**: Create enhanced additional sections as specified

**Independent Test**: Curriculum Overview, Learning Outcomes, and Glossary & Resources sections are properly implemented

- [x] T046 [P] Create Curriculum Overview table with modules/weeks in humanoid-robotics-textbook/src/components/CurriculumOverview/CurriculumTable.jsx
- [x] T047 [P] Implement Learning Outcomes with soft highlights in humanoid-robotics-textbook/src/components/LearningOutcomes/LearningOutcomesList.jsx
- [x] T048 Create Glossary & Resources as collapsible cards in humanoid-robotics-textbook/src/components/GlossaryResources/GlossaryCards.jsx
- [x] T049 Add soft highlights for Learning Outcomes in humanoid-robotics-textbook/src/components/LearningOutcomes/Highlights.jsx

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T050 [P] Documentation updates in docs/
- [ ] T051 Code cleanup and refactoring
- [ ] T052 Performance optimization across all stories
- [ ] T053 [P] Additional unit tests in humanoid-robotics-textbook/src/**/__tests__/
- [ ] T054 Accessibility improvements and WCAG compliance validation
- [ ] T055 Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Module Enhancement (Phase 6)**: Depends on foundational and User Story 2 completion
- **Additional Sections (Phase 7)**: Depends on foundational completion
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable
- **Module Enhancement (Phase 6)**: Depends on User Story 2 for navigation components
- **Additional Sections (Phase 7)**: Can start after Foundational (Phase 2)

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Components before integration
- Core implementation before advanced features
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Components within a story marked [P] can run in parallel
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
5. Add Module Enhancement → Test independently → Deploy/Demo
6. Add Additional Sections → Test independently → Deploy/Demo
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: Module Enhancement
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