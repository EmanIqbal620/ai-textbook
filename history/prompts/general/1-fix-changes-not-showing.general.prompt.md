---
id: 1
title: "Fix Changes Not Showing in Docusaurus"
stage: general
date: "2025-12-15"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "master"
user: "emani"
command: "general"
labels: ["docusaurus", "troubleshooting", "css", "development"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "humanoid-robotics-textbook/src/pages/index.tsx"
  - "humanoid-robotics-textbook/src/pages/index.module.css"
  - "humanoid-robotics-textbook/src/css/custom.css"
  - "humanoid-robotics-textbook/docusaurus.config.ts"
tests: []
---

## User Prompt
i have made changes of musules css ans index page and heasder and front but that are not showing why do and tell

## Assistant Response
I investigated the Docusaurus project and found that the user's changes to modules CSS, index page, header, and front components weren't showing because the development server wasn't properly restarted after making changes.

Key findings:
1. Project is a Docusaurus site with files in correct locations:
   - Index page: `src/pages/index.tsx`
   - Module CSS: `src/pages/index.module.css`
   - Global CSS: `src/css/custom.css`
   - Header: Within `HomepageHeader` function in `index.tsx`

2. Solutions provided:
   - Clear cache with `npm run clear`
   - Restart development server with `npm start`
   - Ensure CSS classes are properly referenced using `styles.className` syntax
   - Verify changes are saved in Git

## Outcome
User now understands that Docusaurus requires the development server to be running and restarted after changes to see updates. Provided complete troubleshooting steps to make changes visible.

## Evaluation
Successfully identified the root cause of the issue and provided actionable solutions. The user should now be able to see their changes after following the recommended steps.