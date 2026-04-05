# Quickstart: Textbook UI Enhancement

## Development Setup

### Prerequisites
- Node.js 18+
- npm or yarn package manager
- Git for version control
- A modern web browser for testing

### Project Setup
1. Navigate to the textbook directory (humanoid-robotics-textbook/)
2. Install dependencies: `npm install` or `yarn install`
3. Install additional dependencies for UI enhancements:
   ```bash
   npm install react-icons framer-motion @docusaurus/theme-classic @docusaurus/preset-classic
   npm install tailwindcss postcss autoprefixer
   npx tailwindcss init -p
   ```
4. Configure environment variables in `.env` file if needed

### Local Development
1. Start the development server: `npm run start` or `yarn start`
2. The site will open at `http://localhost:3000`
3. Edit components in the `src/` directory to see live updates

## Running the Enhanced UI

### 1. Front Page Enhancement
- The front page will display with soft, pastel backgrounds
- Two prominent buttons: "Start Reading" and "Curriculum Overview"
- Subtle hover animations and clean typography

### 2. Module Page Enhancement
- Each module will show title, icon, and short description
- Tabs for different code examples (Python/C++)
- Interactive diagrams and code snippets
- Tips/warnings with soft accent colors
- Scroll-triggered animations for highlights

### 3. Navigation Enhancement
- Sticky floating sidebar with color-coded module icons
- Hover to show module name and topic summary
- Click to navigate to modules
- Progress indicator and previous/next navigation

### 4. Additional Features
- Curriculum Overview with table of modules/weeks
- Learning Outcomes with soft highlights
- Glossary & Resources as collapsible cards
- Floating RAG chatbot icon that opens a contextual side panel
- Breadcrumbs, previous/next buttons, and table of contents per page

### 5. Theme Toggle
- Dark/light mode toggle available in the header
- User preferences are saved in local storage
- Responsive design works on all device sizes

## Testing the UI
- Run component tests: `npm run test` or `yarn test`
- Run E2E tests: `npm run test:e2e` or `yarn test:e2e`
- Check responsive design using browser dev tools
- Verify accessibility with tools like axe-core

## Validation
To validate the UI enhancement works correctly:
1. Check that the front page displays with soft, pastel colors
2. Verify the floating sidebar shows module icons with hover summaries
3. Confirm module pages display with proper structure (title, icon, examples, etc.)
4. Test navigation between modules
5. Verify dark/light mode toggle functionality
6. Check that the RAG chatbot is accessible via the floating icon
7. Validate search functionality works correctly