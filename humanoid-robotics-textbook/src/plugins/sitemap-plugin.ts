import { LoadContext, Plugin } from '@docusaurus/types';
import path from 'path';
import fs from 'fs-extra';

// Custom plugin to serve sitemap.xml during development
export default function sitemapPlugin(context: LoadContext): Plugin<void> {
  const { siteDir } = context;

  return {
    name: 'custom-sitemap-plugin',

    async loadContent() {
      // During development, try to serve sitemap from build directory if it exists
      const buildSitemapPath = path.join(siteDir, 'build', 'sitemap.xml');

      if (await fs.pathExists(buildSitemapPath)) {
        // Copy sitemap to static directory for development serving
        const staticSitemapPath = path.join(siteDir, 'static', 'sitemap.xml');
        await fs.copy(buildSitemapPath, staticSitemapPath);
      }
    },

    configureWebpack(config, isServer, utils) {
      return {
        resolve: {
          alias: {
            // Add any aliases if needed
          },
        },
      };
    },
  };
}