import type { ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import ChatInterface from '@site/src/components/ChatInterface';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>

        <p className="hero__subtitle">
          {siteConfig.tagline}
        </p>

        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/book-intro"
          >
            Start Reading
          </Link>

          <Link
            className="button button--secondary button--lg"
            to="/docs/intro"
          >
            View Contents
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={siteConfig.title}
      description="A comprehensive textbook on Humanoid Robotics and Artificial Intelligence"
    >
      <HomepageHeader />

      <main>
        <section className={styles.chatSection}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <HomepageFeatures />
              </div>
            </div>
          </div>
        </section>
      </main>
      <ChatInterface />
    </Layout>
  );
}
