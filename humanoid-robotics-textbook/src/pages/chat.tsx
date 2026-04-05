import React from 'react';
import Layout from '@theme/Layout';

export default function ChatPage() {
  return (
    <Layout
      title="AI Tutor Chat"
      description="Chat with our AI-powered robotics tutor"
    >
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px'
      }}>
        <div style={{
          width: '100%',
          maxWidth: '900px',
          background: 'white',
          borderRadius: '20px',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.15)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          height: '80vh'
        }}>
          {/* Header */}
          <div style={{
            padding: '24px 32px',
            background: 'linear-gradient(135deg, #0ea5e9, #8b5cf6)',
            color: 'white',
            display: 'flex',
            alignItems: 'center',
            gap: '16px'
          }}>
            <div style={{
              fontSize: '36px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '56px',
              height: '56px',
              background: 'rgba(255, 255, 255, 0.2)',
              borderRadius: '14px'
            }}>
              🤖
            </div>
            <div>
              <h1 style={{
                fontSize: '24px',
                fontWeight: '700',
                margin: '0 0 4px 0'
              }}>
                Humanoid Robotics AI Tutor
              </h1>
              <p style={{
                fontSize: '14px',
                margin: '0',
                opacity: '0.9'
              }}>
                Ask me anything about robotics, ROS2, simulation, AI, and more!
              </p>
            </div>
          </div>

          {/* Chat Container */}
          <div style={{
            flex: '1',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '40px',
            background: '#f8fafc'
          }}>
            <div style={{
              textAlign: 'center',
              maxWidth: '600px'
            }}>
              <div style={{
                fontSize: '80px',
                marginBottom: '24px'
              }}>
                💬
              </div>
              <h2 style={{
                fontSize: '28px',
                fontWeight: '700',
                color: '#1e293b',
                marginBottom: '16px'
              }}>
                Chat Interface Coming Soon
              </h2>
              <p style={{
                fontSize: '16px',
                color: '#64748b',
                lineHeight: '1.6',
                marginBottom: '32px'
              }}>
                We're building an amazing AI-powered chat experience to help you learn 
                humanoid robotics. The chat interface will be available here shortly.
              </p>
              
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '16px',
                marginBottom: '32px'
              }}>
                <div style={{
                  padding: '20px',
                  background: 'white',
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
                  border: '1px solid #e2e8f0'
                }}>
                  <div style={{fontSize: '32px', marginBottom: '12px'}}>⚡</div>
                  <h3 style={{fontSize: '16px', fontWeight: '600', margin: '0 0 8px 0', color: '#1e293b'}}>
                    Instant Answers
                  </h3>
                  <p style={{fontSize: '14px', color: '#64748b', margin: '0'}}>
                    Get immediate responses to your robotics questions
                  </p>
                </div>
                
                <div style={{
                  padding: '20px',
                  background: 'white',
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
                  border: '1px solid #e2e8f0'
                }}>
                  <div style={{fontSize: '32px', marginBottom: '12px'}}>📚</div>
                  <h3 style={{fontSize: '16px', fontWeight: '600', margin: '0 0 8px 0', color: '#1e293b'}}>
                    Textbook-Based
                  </h3>
                  <p style={{fontSize: '14px', color: '#64748b', margin: '0'}}>
                    Answers sourced from the textbook content
                  </p>
                </div>
                
                <div style={{
                  padding: '20px',
                  background: 'white',
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
                  border: '1px solid #e2e8f0'
                }}>
                  <div style={{fontSize: '32px', marginBottom: '12px'}}>🔗</div>
                  <h3 style={{fontSize: '16px', fontWeight: '600', margin: '0 0 8px 0', color: '#1e293b'}}>
                    Source References
                  </h3>
                  <p style={{fontSize: '14px', color: '#64748b', margin: '0'}}>
                    See exactly where answers come from
                  </p>
                </div>
              </div>

              <div style={{
                padding: '20px',
                background: 'linear-gradient(135deg, rgba(14, 165, 233, 0.1), rgba(139, 92, 246, 0.1))',
                borderRadius: '12px',
                border: '2px solid #0ea5e9'
              }}>
                <h3 style={{
                  fontSize: '18px',
                  fontWeight: '600',
                  color: '#1e293b',
                  margin: '0 0 12px 0'
                }}>
                  🎯 In the Meantime
                </h3>
                <p style={{
                  fontSize: '14px',
                  color: '#64748b',
                  margin: '0 0 16px 0'
                }}>
                  You can use our floating chat widget available on all textbook pages! 
                  Look for the chat bubble icon in the bottom-right corner.
                </p>
                <a 
                  href="/docs/intro"
                  style={{
                    display: 'inline-block',
                    padding: '12px 24px',
                    background: 'linear-gradient(135deg, #0ea5e9, #8b5cf6)',
                    color: 'white',
                    textDecoration: 'none',
                    borderRadius: '10px',
                    fontWeight: '600',
                    fontSize: '15px',
                    transition: 'transform 0.2s',
                  }}
                  onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
                  onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
                >
                  Browse Textbook →
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
