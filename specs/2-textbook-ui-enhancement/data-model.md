# Data Model: Textbook UI Enhancement

## Entities

### Module
- **id**: string (unique identifier for the module)
- **title**: string (the module title)
- **icon**: string (the icon representation - emoji or icon name)
- **description**: string (short description of the module)
- **topics**: array of strings (list of topics covered in the module)
- **learningOutcomes**: array of strings (learning outcomes for the module)
- **examples**: array of Example objects (code examples in different languages)
- **diagrams**: array of Diagram objects (diagrams and images for the module)
- **tipsWarnings**: array of TipWarning objects (tips and warnings for the module)
- **order**: number (the order in which modules appear)
- **colorCode**: string (the color code for the module in the sidebar)

### Example
- **id**: string (unique identifier for the example)
- **language**: string (the programming language: Python, C++, etc.)
- **code**: string (the actual code content)
- **description**: string (description of what the example demonstrates)
- **moduleRef**: string (reference to the parent module)

### Diagram
- **id**: string (unique identifier for the diagram)
- **type**: enum (IMAGE, MERMAID, SVG, etc.)
- **src**: string (source URL or embedded content)
- **altText**: string (alternative text for accessibility)
- **caption**: string (caption for the diagram)
- **moduleRef**: string (reference to the parent module)

### TipWarning
- **id**: string (unique identifier for the tip/warning)
- **type**: enum (TIP, WARNING, NOTE, IMPORTANT)
- **content**: string (the tip or warning content)
- **accentColor**: string (the soft accent color for the tip/warning)
- **moduleRef**: string (reference to the parent module)

### UserPreference
- **id**: string (user identifier)
- **theme**: enum (LIGHT, DARK)
- **fontSize**: string (user's preferred font size)
- **highContrast**: boolean (whether to use high contrast mode)
- **lastVisitedModule**: string (the last module the user visited)
- **bookmarks**: array of string (bookmarked modules)

### SearchIndex
- **id**: string (unique identifier for the search entry)
- **title**: string (title of the content)
- **content**: string (the searchable content)
- **moduleRef**: string (reference to the parent module)
- **url**: string (URL to the content)
- **type**: enum (MODULE, SECTION, EXAMPLE, DIAGRAM)

### NavigationItem
- **id**: string (unique identifier for the navigation item)
- **title**: string (the display name)
- **icon**: string (icon representation)
- **summary**: string (short summary shown on hover)
- **url**: string (URL to navigate to)
- **parentModule**: string (reference to parent module, if applicable)
- **order**: number (the order in the navigation)

## Relationships
- Module contains many Examples
- Module contains many Diagrams
- Module contains many TipWarnings
- UserPreference links to many Modules (via bookmarks)
- SearchIndex links to Modules
- NavigationItem links to Modules

## Validation Rules
- Module.title must be non-empty
- Module.id must be unique
- Module.order must be a positive integer
- Example.language must be a supported language
- TipWarning.type must be one of the defined values
- UserPreference.theme must be either LIGHT or DARK
- NavigationItem.order must be a positive integer

## State Transitions
- UserPreference: Guest → Authenticated (when user logs in)
- Module: Unvisited → InProgress → Completed (as user progresses)
- Theme: LIGHT → DARK or DARK → LIGHT (when user toggles theme)