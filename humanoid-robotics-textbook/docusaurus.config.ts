import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Humanoid Robotics Textbook',
  tagline: 'Master Physical AI & Humanoid Robotics - From ROS2 to Vision-Language-Action Systems',
  favicon: 'img/favicon.ico',

  // Production URL
  url: 'https://humanoid-robotics-textbook.vercel.app',
  baseUrl: '/',

  // Organization info
  organizationName: 'humanoid-robotics',
  projectName: 'humanoid-robotics-textbook',

  onBrokenLinks: 'ignore',
  onBrokenMarkdownLinks: 'ignore',

  // Internationalization
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Presets
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/humanoid-robotics/humanoid-robotics-textbook/tree/main/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          routeBasePath: '/',  // Make docs the main landing page
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.final.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  // Themes
  themes: [
    '@docusaurus/theme-mermaid',
  ],

  // Theme Configuration
  themeConfig: {
    // Social card
    image: 'img/humanoid-robotics-social-card.jpg',
    
    // Color mode
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    // Navbar
    navbar: {
      title: 'HUMANOID.AI',
      logo: {
        alt: 'Humanoid Robotics Logo',
        src: 'img/chatbot-icon.svg',
        width: 40,
        height: 40,
      },
      items: [
        {
          type: 'doc',
          docId: 'intro',
          position: 'left',
          label: 'LEARN',
          className: 'navbar-link navbar-link-learn',
        },
        {
          type: 'doc',
          docId: 'prerequisites',
          position: 'left',
          label: 'PREREQUISITES',
          className: 'navbar-link navbar-link-prereq',
        },
        {
          type: 'html',
          position: 'left',
          value: '<div class="navbar__divider"></div>',
        },
        {
          href: 'https://github.com/humanoid-robotics/humanoid-robotics-textbook',
          label: 'GITHUB',
          position: 'right',
          className: 'navbar-github-link',
        },
      ],
    },

    // Footer
    footer: {
      style: 'dark',
      links: [
        {
          title: '📖 Modules',
          items: [
            {
              label: 'Module 1: ROS 2 Fundamentals',
              to: '/docs/module-1-ros2/index',
            },
            {
              label: 'Module 2: Simulation & Digital Twins',
              to: '/docs/module-2-simulation/index',
            },
            {
              label: 'Module 3: NVIDIA Isaac AI',
              to: '/docs/module-3-ai-brain/index',
            },
            {
              label: 'Module 4: Vision-Language-Action',
              to: '/docs/module-4-vla/index',
            },
            {
              label: 'Module 5: Hardware Requirements',
              to: '/docs/module-5-hardware/index',
            },
            {
              label: 'Module 6: Assessment',
              to: '/docs/module-6-assessment/index',
            },
          ],
        },
        {
          title: '🔧 Resources',
          items: [
            {
              label: '📝 Glossary',
              to: '/docs/glossary',
            },
            {
              label: '📚 Bibliography',
              to: '/docs/bibliography',
            },
            {
              label: '🔗 Additional Resources',
              to: '/docs/additional-resources',
            },
            {
              label: '💬 Chat with AI Tutor',
              to: '/chat',
            },
          ],
        },
        {
          title: '👥 Community',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/humanoid-robotics/humanoid-robotics-textbook',
            },
            {
              label: 'Report an Issue',
              href: 'https://github.com/humanoid-robotics/humanoid-robotics-textbook/issues',
            },
            {
              label: 'Discussions',
              href: 'https://github.com/humanoid-robotics/humanoid-robotics-textbook/discussions',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Humanoid Robotics Textbook. Built with ❤️ using Docusaurus.`,
    },

    // Prism theme for code blocks
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'markdown'],
    },

    // Algolia DocSearch (configure with your own credentials)
    algolia: {
      appId: 'YOUR_ALGOLIA_APP_ID',
      apiKey: 'YOUR_ALGOLIA_API_KEY',
      indexName: 'humanoid-robotics-textbook',
      contextualSearch: true,
      searchParameters: {},
      searchPagePath: 'search',
    },

    // Announcement bar
    announcementBar: {
      id: 'support_us',
      content:
        '🎉 <strong>New:</strong> Module 4 on Vision-Language-Action Systems is now live! <a target="_blank" rel="noopener noreferrer" href="/docs/module-4-vla/index">Start learning →</a>',
      backgroundColor: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      textColor: '#ffffff',
      isCloseable: true,
    },

    // Live code block settings
    liveCodeBlock: {
      playgroundPosition: 'bottom',
    },

    // Docs sidebar
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
  } satisfies Preset.ThemeConfig,

  // Plugins
  plugins: [
    // Optional: Add sitemap plugin
    async function myPlugin(context, options) {
      return {
        name: 'docusaurus-tailwindcss',
        configurePostCss(postcssOptions) {
          return postcssOptions;
        },
      };
    },
  ],

  // Scripts to load - Chat Widget
  scripts: [
    {
      src: '/js/chat-widget.js',
      async: true,
      defer: true,
    },
  ],
};

export default config;
