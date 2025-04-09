'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import ClientOnly from './components/ClientOnly';

// Dynamically import ChatInterface with SSR disabled
const ChatInterface = dynamic(
  () => import('./components/ChatInterface'),
  { ssr: false }
);

export default function Home() {
  return (
    <div className="container mx-auto p-4 md:p-8 min-h-screen">
      <header className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">PaperSeeker</h1>
        <p className="text-gray-600 dark:text-gray-400">Find the research papers you need with AI assistance</p>
      </header>
      
      <main>
        <ClientOnly>
          <ChatInterface />
        </ClientOnly>
      </main>
      
      <footer className="mt-12 text-center text-sm text-gray-500 dark:text-gray-400">
        <p>© {new Date().getFullYear()} PaperSeeker - Your AI research assistant</p>
      </footer>
    </div>
  );
}
