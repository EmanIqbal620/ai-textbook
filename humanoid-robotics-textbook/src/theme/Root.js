import React from 'react';
import ChatInterface from '@site/src/components/ChatInterface';

export default function Root({children}) {
  return (
    <>
      {children}
      <ChatInterface />
    </>
  );
}