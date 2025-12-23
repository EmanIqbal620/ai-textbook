import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

import path from 'path';
import fs from 'fs-extra';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive textbook for learning humanoid robotics with ROS 2, Gazebo, Unity, and NVIDIA Isaac',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://humanoid-robotics-textbook-zeta.vercel.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'your-organization', // Usually your GitHub org/user name.
  projectName: 'humanoid-robotics-textbook', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

 presets: [
  [
    'classic',
    {
      docs: {
        sidebarPath: './sidebars.ts',
        editUrl:
          'https://github.com/your-organization/humanoid-robotics-textbook/edit/main/',
      },
      blog: false,
      theme: {
        customCss: './src/css/custom.css',
      },
      sitemap: {
        changefreq: 'weekly',
        priority: 0.5,
        ignorePatterns: ['/tags/**'],
        
      },
    } satisfies Preset.Options,
  ],
],


  themes: [
    // Add the Mermaid theme
    '@docusaurus/theme-mermaid',
  ],

  plugins: [
    // Plugin to add the chat widget to all pages
    async function chatWidgetPlugin(context, options) {
      return {
        name: 'chat-widget-plugin',
        configureWebpack(config, isServer, { getStyleLoaders }) {
          return {
            // Add webpack aliases or modifications if needed
          };
        },
        async contentLoaded({ content, actions }) {
          // This is called after content is loaded
        },
        async postBuild(props) {
          // This is called after build
        },
        configureDevServer(app, server, { options }) {
          // Serve sitemap.xml during development
          app.get('/sitemap.xml', (req, res) => {
            const sitemapPath = path.join(process.cwd(), 'build', 'sitemap.xml');

            // Check if sitemap exists in build directory
            if (fs.existsSync(sitemapPath)) {
              // Serve the actual sitemap file if it exists
              res.sendFile(sitemapPath);
            } else {
              // If sitemap doesn't exist in build folder, generate a minimal one for development
              res.set('Content-Type', 'application/xml');
              res.send(`<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>http://localhost:3000/</loc>
    <changefreq>weekly</changefreq>
    <priority>0.5</priority>
  </url>
</urlset>`);
            }
          });
        },
        injectHtmlTags() {
          return {
            postBodyTags: [
              `<div id="chat-widget-container"></div>`,
            ],
          };
        },
      };
    },
  ],
  themes: [
    // Add the Mermaid theme
    '@docusaurus/theme-mermaid',
  ],
  clientModules: [
    require.resolve('./src/client-modules.js'),
  ],
  themeConfig: {
    // Replace with your project's social card
    image: 'img/robotics-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        src: 'img/robotics-logo.svg..avif',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/your-organization/humanoid-robotics-textbook',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Introduction',
              to: '/docs/book-intro',
            },
            {
              label: 'Module 1: ROS 2',
              to: '/docs/module-1-ros2',
            },
            {
              label: 'Module 2: Simulation',
              to: '/docs/module-2-simulation',
            },
            {
              label: 'Module 3: AI Brain',
              to: '/docs/module-3-ai-brain',
            },
            {
              label: 'Module 4: VLA',
              to: '/docs/module-4-vla',
            },
          ],
        },
        {
          title: 'Additional Modules',
          items: [
            {
              label: 'Module 5: Hardware Requirements',
              to: '/docs/module-5-hardware',
            },
            {
              label: 'Module 6: Assessment Guidelines',
              to: '/docs/module-6-assessment',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Glossary',
              to: '/docs/glossary',
            },
            {
              label: 'Bibliography',
              to: '/docs/bibliography',
            },
            {
              label: 'Additional Resources',
              to: '/docs/additional-resources',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/your-organization/humanoid-robotics-textbook',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'docker', 'csharp', 'cpp', 'java', 'rust'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
