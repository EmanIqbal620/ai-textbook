import React, { useEffect } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

// Add smooth scrolling script
function SmoothScrollScript() {
  useEffect(() => {
    // Handle TOC link clicks
    const handleTOCClick = (e) => {
      const link = e.target.closest('a[href^="#"]');
      if (link && link.closest('[class*="tableOfContents"]')) {
        e.preventDefault();
        const href = link.getAttribute('href');
        const element = document.querySelector(href);
        if (element) {
          const offset = 120;
          const elementPosition = element.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.pageYOffset - offset;
          
          window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
          });
          
          window.history.pushState(null, '', href);
        }
      }
    };

    document.addEventListener('click', handleTOCClick);
    return () => document.removeEventListener('click', handleTOCClick);
  }, []);
  
  return null;
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Master Humanoid Robotics`}
      description="Humanoid Robotics Textbook - Learn advanced robotics concepts">
      <SmoothScrollScript />
      <main>
        <section className={styles.features}>
          <div className="container padding-horiz--md">

            {/* HERO SECTION - Dark Background System */}
            <div style={{
              position: 'relative',
              background: '#080c18',
              borderRadius: '8px',
              padding: '40px 40px 60px 40px',
              marginBottom: '40px',
              overflow: 'hidden',
              maxWidth: '1200px',
              margin: '0 auto 40px auto'
            }}>
              {/* Grid Overlay */}
              <div style={{
                position: 'absolute',
                inset: 0,
                backgroundImage: `linear-gradient(#1e3a5f22 1px, transparent 1px), linear-gradient(90deg, #1e3a5f22 1px, transparent 1px)`,
                backgroundSize: '40px 40px',
                animation: 'gridmove 8s linear infinite'
              }}></div>

              {/* Left Glow */}
              <div style={{
                position: 'absolute',
                left: '-100px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '300px',
                height: '300px',
                background: 'radial-gradient(circle, #00d4ff0a, transparent 70%)',
                borderRadius: '50%'
              }}></div>

              {/* Right Glow */}
              <div style={{
                position: 'absolute',
                right: '-100px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '300px',
                height: '300px',
                background: 'radial-gradient(circle, #7b2fff0a, transparent 70%)',
                borderRadius: '50%'
              }}></div>

              {/* Scan Line */}
              <div style={{
                position: 'absolute',
                left: 0,
                right: 0,
                height: '1px',
                background: 'linear-gradient(90deg, transparent, #00d4ff33, transparent)',
                animation: 'scan 4s linear infinite'
              }}></div>

              {/* Faded Robot SVG (Right Side) */}
              <svg viewBox="0 0 200 300" style={{
                position: 'absolute',
                right: '40px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '200px',
                height: '300px',
                opacity: 0.12,
                animation: 'float 4s ease-in-out infinite'
              }}>
                <rect x="85" y="20" width="30" height="35" rx="5" stroke="#00d4ff" strokeWidth="1.5" fill="none"/>
                <rect x="70" y="60" width="60" height="80" rx="8" stroke="#00d4ff" strokeWidth="1.5" fill="none"/>
                <line x1="50" y1="70" x2="70" y2="80" stroke="#00d4ff" strokeWidth="1.5"/>
                <line x1="150" y1="70" x2="130" y2="80" stroke="#00d4ff" strokeWidth="1.5"/>
                <line x1="50" y1="90" x2="70" y2="85" stroke="#00d4ff" strokeWidth="1.5"/>
                <line x1="150" y1="90" x2="130" y2="85" stroke="#00d4ff" strokeWidth="1.5"/>
                <line x1="85" y1="145" x2="85" y2="200" stroke="#00d4ff" strokeWidth="1.5"/>
                <line x1="115" y1="145" x2="115" y2="200" stroke="#00d4ff" strokeWidth="1.5"/>
                <line x1="85" y1="200" x2="60" y2="240" stroke="#00d4ff" strokeWidth="1.5"/>
                <line x1="115" y1="200" x2="140" y2="240" stroke="#00d4ff" strokeWidth="1.5"/>
              </svg>

              {/* Content */}
              <div style={{position: 'relative', zIndex: 1, maxWidth: '600px'}}>
                {/* Hero Tag */}
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '11px',
                  letterSpacing: '3px',
                  color: '#00d4ff',
                  border: '1px solid #00d4ff33',
                  background: '#00d4ff08',
                  borderRadius: '2px',
                  padding: '6px 16px',
                  marginBottom: '24px'
                }}>
                  <div style={{
                    width: '6px',
                    height: '6px',
                    background: '#00ff88',
                    borderRadius: '50%',
                    boxShadow: '0 0 6px #00ff88',
                    animation: 'pulse 2s infinite'
                  }}></div>
                  PHYSICAL AI // HUMANOID ROBOTICS
                </div>

                {/* Hero Title */}
                <h1 style={{
                  fontSize: '42px',
                  fontWeight: 700,
                  letterSpacing: '3px',
                  textShadow: '0 0 30px #00d4ff22',
                  marginBottom: '20px',
                  lineHeight: 1.3
                }}>
                  <span style={{color: '#e8f4f8', display: 'block'}}>HUMANOID</span>
                  <span style={{color: '#00d4ff', display: 'block'}}>ROBOTICS</span>
                </h1>

                {/* Hero Subtitle */}
                <p style={{
                  fontSize: '16px',
                  color: '#4a7a9b',
                  letterSpacing: '0.5px',
                  lineHeight: 1.8,
                  fontFamily: 'sans-serif',
                  marginBottom: '32px'
                }}>
                  Your comprehensive guide to building intelligent humanoid robots. Master <span style={{color: '#7a9ab8', fontFamily: 'monospace', fontSize: '15px'}}>ROS2</span>, <span style={{color: '#7a9ab8', fontFamily: 'monospace', fontSize: '15px'}}>Physical AI</span>, and <span style={{color: '#7a9ab8', fontFamily: 'monospace', fontSize: '15px'}}>VLA</span> systems through hands-on projects and real-world applications.
                </p>

                {/* Buttons */}
                <div style={{display: 'flex', gap: '16px', marginBottom: '28px'}}>
                  <Link to="/module-1-ros2" style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px',
                    background: '#00d4ff',
                    color: '#080c18',
                    fontSize: '12px',
                    letterSpacing: '1.5px',
                    fontWeight: 700,
                    padding: '12px 24px',
                    borderRadius: '3px',
                    textDecoration: 'none'
                  }}>
                    <svg viewBox="0 0 24 24" style={{width: '16px', height: '16px', fill: 'currentColor'}}>
                      <path d="M8 5v14l11-7z"/>
                    </svg>
                    START LEARNING
                  </Link>
                  <Link to="/module-1-ros2/week-1" style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '8px',
                    background: 'transparent',
                    color: '#e8f4f8',
                    border: '1px solid #1e3a5f',
                    borderRadius: '3px',
                    padding: '12px 24px',
                    fontSize: '12px',
                    letterSpacing: '1.5px',
                    textDecoration: 'none'
                  }}>
                    <svg viewBox="0 0 24 24" style={{width: '16px', height: '16px', fill: 'currentColor'}}>
                      <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                    </svg>
                    READ OVERVIEW
                  </Link>
                </div>

                {/* Stats Row */}
                <div style={{
                  display: 'flex',
                  gap: '32px',
                  borderTop: '1px solid #1e3a5f',
                  paddingTop: '20px'
                }}>
                  <div>
                    <div style={{fontSize: '24px', fontWeight: 'bold', color: '#00d4ff'}}>6</div>
                    <div style={{fontSize: '10px', letterSpacing: '1px', color: '#4a7a9b'}}>MODULES</div>
                  </div>
                  <div>
                    <div style={{fontSize: '24px', fontWeight: 'bold', color: '#7b2fff'}}>24+</div>
                    <div style={{fontSize: '10px', letterSpacing: '1px', color: '#4a7a9b'}}>WEEKS</div>
                  </div>
                  <div>
                    <div style={{fontSize: '24px', fontWeight: 'bold', color: '#00d4ff'}}>100%</div>
                    <div style={{fontSize: '10px', letterSpacing: '1px', color: '#4a7a9b'}}>FREE</div>
                  </div>
                  <div>
                    <div style={{fontSize: '24px', fontWeight: 'bold', color: '#00ff88'}}>LIVE</div>
                    <div style={{fontSize: '10px', letterSpacing: '1px', color: '#4a7a9b'}}>AI TUTOR</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Modules Grid - 3 per row, wraps to multiple rows */}
            <div className="module-card-wrapper">

            {/* Module 01 */}
            <div className="module-wrapper">
              <div className="module-label">
                <span className="module-num">MODULE // 01</span>
                <div className="status-dot"></div>
              </div>
              <div className="module-title">ROS 2 Fundamentals</div>

              <div className="module-card">
                <div className="icon-box">
                <svg viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="3"/>
                  <line x1="12" y1="2" x2="12" y2="6"/>
                  <line x1="12" y1="18" x2="12" y2="22"/>
                  <line x1="2" y1="12" x2="6" y2="12"/>
                  <line x1="18" y1="12" x2="22" y2="12"/>
                </svg>
              </div>
              <div>
                <p>Learn about Robot Operating System 2 fundamentals and advanced concepts including nodes, topics, services, and actions.</p>
                <div style={{margin: '12px 0'}}>
                  <span className="week-tag">WEEK-01 // INTRO</span>
                  <span className="week-tag">WEEK-02 // NODES</span>
                  <span className="week-tag purple">WEEK-03 // SERVICES</span>
                </div>
                <div style={{display: 'flex', gap: '16px', marginTop: '14px'}}>
                  <Link className="button button--primary" to="/module-1-ros2">
                    <svg viewBox="0 0 24 24" style={{width:'16px',height:'16px',fill:'currentColor'}}><path d="M8 5v14l11-7z"/></svg>
                    START LEARNING
                  </Link>
                  <Link className="button button--secondary" to="/module-1-ros2/week-1">
                    <svg viewBox="0 0 24 24" style={{width:'16px',height:'16px',fill:'currentColor'}}><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
                    READ OVERVIEW
                  </Link>
                </div>
                <div className="stats-row">
                  <div className="stat-item"><span className="stat-number cyan">6</span><span className="stat-label">MODULES</span></div>
                  <div className="stat-item"><span className="stat-number purple">24+</span><span className="stat-label">WEEKS</span></div>
                  <div className="stat-item"><span className="stat-number cyan">100%</span><span className="stat-label">FREE</span></div>
                  <div className="stat-item"><span className="stat-number green">LIVE</span><span className="stat-label">AI TUTOR</span></div>
                </div>
                <div className="progress-bar"><div className="progress-fill" style={{width: '0%'}}></div></div>
              </div>
            </div>
            </div>

            {/* Module 02 */}
            <div className="module-wrapper">
              <div className="module-label">
                <span className="module-num">MODULE // 02</span>
                <div className="status-dot"></div>
              </div>
              <div className="module-title">Simulation & Digital Twins</div>

              <div className="module-card">
                <div className="icon-box">
                <svg viewBox="0 0 24 24">
                  <path d="M21 16.5c0 .38-.21.71-.53.88l-7.9 4.44c-.16.12-.36.18-.57.18-.21 0-.41-.06-.57-.18l-7.9-4.44A.991.991 0 0 1 3 16.5v-9c0-.38.21-.71.53-.88l7.9-4.44c.16-.12.36-.18.57-.18.21 0 .41.06.57.18l7.9 4.44c.32.17.53.5.53.88v9z"/>
                  <path d="M12 2.02L4.12 6.5 12 10.98 19.88 6.5 12 2.02zM12 21.98l7.88-4.48v-9L12 12.98 4.12 8.5v9L12 21.98z" opacity=".5"/>
                </svg>
              </div>
              <div>
                <p>Master robotics simulation with Gazebo and Unity environments for safe testing and development.</p>
                <div style={{margin: '12px 0'}}>
                  <span className="week-tag">WEEK-04 // GAZEBO</span>
                  <span className="week-tag purple">WEEK-05 // UNITY</span>
                </div>
                <div style={{display: 'flex', gap: '10px', marginTop: '14px'}}>
                  <Link className="button button--primary" to="/module-2-simulation">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M8 5v14l11-7z"/></svg>
                    START LEARNING
                  </Link>
                  <Link className="button button--secondary" to="/module-2-simulation/week-4">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
                    READ OVERVIEW
                  </Link>
                </div>
                <div className="stats-row">
                  <div className="stat-item"><span className="stat-number cyan">6</span><span className="stat-label">MODULES</span></div>
                  <div className="stat-item"><span className="stat-number purple">24+</span><span className="stat-label">WEEKS</span></div>
                  <div className="stat-item"><span className="stat-number cyan">100%</span><span className="stat-label">FREE</span></div>
                  <div className="stat-item"><span className="stat-number green">LIVE</span><span className="stat-label">AI TUTOR</span></div>
                </div>
                <div className="progress-bar"><div className="progress-fill" style={{width: '0%'}}></div></div>
              </div>
            </div>
            </div>

            {/* Module 03 */}
            <div className="module-wrapper">
              <div className="module-label">
                <span className="module-num">MODULE // 03</span>
                <div className="status-dot"></div>
              </div>
              <div className="module-title">AI for Robotics</div>

              <div className="module-card">
                <div className="icon-box">
                <svg viewBox="0 0 24 24">
                  <rect x="3" y="3" width="7" height="7"/>
                  <rect x="14" y="3" width="7" height="7"/>
                  <rect x="3" y="14" width="7" height="7"/>
                  <rect x="14" y="14" width="7" height="7"/>
                </svg>
              </div>
              <div>
                <p>Understand AI for robotics with NVIDIA Isaac and Vision-Language-Action systems.</p>
                <div style={{margin: '12px 0'}}>
                  <span className="week-tag">WEEK-06 // ISAAC</span>
                  <span className="week-tag purple">WEEK-07 // PERCEPTION</span>
                  <span className="week-tag">WEEK-08 // PLANNING</span>
                </div>
                <div style={{display: 'flex', gap: '10px', marginTop: '14px'}}>
                  <Link className="button button--primary" to="/module-3-ai-brain">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M8 5v14l11-7z"/></svg>
                    START LEARNING
                  </Link>
                  <Link className="button button--secondary" to="/module-3-ai-brain/week-6">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
                    READ OVERVIEW
                  </Link>
                </div>
                <div className="stats-row">
                  <div className="stat-item"><span className="stat-number cyan">6</span><span className="stat-label">MODULES</span></div>
                  <div className="stat-item"><span className="stat-number purple">24+</span><span className="stat-label">WEEKS</span></div>
                  <div className="stat-item"><span className="stat-number cyan">100%</span><span className="stat-label">FREE</span></div>
                  <div className="stat-item"><span className="stat-number green">LIVE</span><span className="stat-label">AI TUTOR</span></div>
                </div>
                <div className="progress-bar"><div className="progress-fill" style={{width: '0%'}}></div></div>
              </div>
            </div>
            </div>

            {/* Module 04 */}
            <div className="module-wrapper">
              <div className="module-label">
                <span className="module-num">MODULE // 04</span>
                <div className="status-dot"></div>
              </div>
              <div className="module-title">VLA Systems</div>

              <div className="module-card">
                <div className="icon-box">
                <svg viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10"/>
                  <circle cx="12" cy="12" r="4"/>
                  <circle cx="12" cy="12" r="2"/>
                </svg>
              </div>
              <div>
                <p>Combine vision, language, and action for next-generation robot control.</p>
                <div style={{margin: '12px 0'}}>
                  <span className="week-tag purple">WEEK-09 // VISION</span>
                  <span className="week-tag">WEEK-10 // LANGUAGE</span>
                  <span className="week-tag purple">WEEK-11 // ACTION</span>
                </div>
                <div style={{display: 'flex', gap: '10px', marginTop: '14px'}}>
                  <Link className="button button--primary" to="/module-4-vla">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M8 5v14l11-7z"/></svg>
                    START LEARNING
                  </Link>
                  <Link className="button button--secondary" to="/module-4-vla/week-9">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
                    READ OVERVIEW
                  </Link>
                </div>
                <div className="stats-row">
                  <div className="stat-item"><span className="stat-number cyan">6</span><span className="stat-label">MODULES</span></div>
                  <div className="stat-item"><span className="stat-number purple">24+</span><span className="stat-label">WEEKS</span></div>
                  <div className="stat-item"><span className="stat-number cyan">100%</span><span className="stat-label">FREE</span></div>
                  <div className="stat-item"><span className="stat-number green">LIVE</span><span className="stat-label">AI TUTOR</span></div>
                </div>
                <div className="progress-bar"><div className="progress-fill" style={{width: '0%'}}></div></div>
              </div>
            </div>
            </div>

            {/* Module 05 */}
            <div className="module-wrapper">
              <div className="module-label">
                <span className="module-num">MODULE // 05</span>
                <div className="status-dot"></div>
              </div>
              <div className="module-title">Hardware Requirements</div>

              <div className="module-card">
                <div className="icon-box">
                <svg viewBox="0 0 24 24">
                  <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.488.488 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 0 0-.48-.41h-3.84a.484.484 0 0 0-.48.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.488.488 0 0 0-.59.22L2.09 8.83a.488.488 0 0 0 .12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58a.488.488 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.27.41.48.41h3.84c.24 0 .44-.17.48-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32a.488.488 0 0 0-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
                </svg>
              </div>
              <div>
                <p>Learn about CPU, GPU, sensors, and embedded systems required for humanoid robot development.</p>
                <div style={{margin: '12px 0'}}>
                  <span className="week-tag">CPU // GPU</span>
                  <span className="week-tag purple">SENSORS</span>
                  <span className="week-tag">ACTUATORS</span>
                </div>
                <div style={{display: 'flex', gap: '10px', marginTop: '14px'}}>
                  <Link className="button button--primary" to="/module-5-hardware">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M8 5v14l11-7z"/></svg>
                    START LEARNING
                  </Link>
                  <Link className="button button--secondary" to="/module-5-hardware/hardware-specifications">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
                    READ OVERVIEW
                  </Link>
                </div>
                <div className="stats-row">
                  <div className="stat-item"><span className="stat-number cyan">6</span><span className="stat-label">MODULES</span></div>
                  <div className="stat-item"><span className="stat-number purple">24+</span><span className="stat-label">WEEKS</span></div>
                  <div className="stat-item"><span className="stat-number cyan">100%</span><span className="stat-label">FREE</span></div>
                  <div className="stat-item"><span className="stat-number green">LIVE</span><span className="stat-label">AI TUTOR</span></div>
                </div>
                <div className="progress-bar"><div className="progress-fill" style={{width: '0%'}}></div></div>
              </div>
            </div>
            </div>

            {/* Module 06 */}
            <div className="module-wrapper">
              <div className="module-label">
                <span className="module-num">MODULE // 06</span>
                <div className="status-dot"></div>
              </div>
              <div className="module-title">Assessment & Projects</div>

              <div className="module-card">
                <div className="icon-box">
                <svg viewBox="0 0 24 24">
                  <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
                </svg>
              </div>
              <div>
                <p>Test your knowledge with hands-on projects, quizzes, and practical exams.</p>
                <div style={{margin: '12px 0'}}>
                  <span className="week-tag purple">PROJECTS</span>
                  <span className="week-tag">QUIZZES</span>
                  <span className="week-tag purple">EXAMS</span>
                </div>
                <div style={{display: 'flex', gap: '10px', marginTop: '14px'}}>
                  <Link className="button button--primary" to="/module-6-assessment">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M8 5v14l11-7z"/></svg>
                    START LEARNING
                  </Link>
                  <Link className="button button--secondary" to="/module-6-assessment/assessment-methods">
                    <svg viewBox="0 0 24 24" style={{width:'12px',height:'12px',fill:'currentColor'}}><path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>
                    READ OVERVIEW
                  </Link>
                </div>
                <div className="stats-row">
                  <div className="stat-item"><span className="stat-number cyan">6</span><span className="stat-label">MODULES</span></div>
                  <div className="stat-item"><span className="stat-number purple">24+</span><span className="stat-label">WEEKS</span></div>
                  <div className="stat-item"><span className="stat-number cyan">100%</span><span className="stat-label">FREE</span></div>
                  <div className="stat-item"><span className="stat-number green">LIVE</span><span className="stat-label">AI TUTOR</span></div>
                </div>
                <div className="progress-bar"><div className="progress-fill" style={{width: '0%'}}></div></div>
              </div>
            </div>

            </div>
            {/* End of Modules Grid */}

            </div>
            {/* End of Hero Section */}

          </div>
        </section>
      </main>
    </Layout>
  );
}
