'use client';

import React from 'react';

const ChatMessage = ({ message }) => {
  const isBot = message.sender === 'bot';
  
  // Thinking indicator
  if (message.isThinking) {
    return (
      <div className="flex justify-start mb-4">
        <div className="bg-gray-200 dark:bg-gray-800 p-3 rounded-lg max-w-[80%]">
          <div className="flex items-center space-x-2">
            <span className="text-gray-500 text-sm">{message.text}</span>
            <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
      <div 
        className={`p-3 rounded-lg max-w-[80%] ${
          isBot ? 'bg-gray-200 dark:bg-gray-800' : 'bg-blue-500 text-white'
        }`}
      >
        {message.messageTitle && (
          <div className="text-xs font-semibold mb-1 pb-1 border-b border-gray-300 dark:border-gray-600">
            {message.messageTitle}
          </div>
        )}
        
        <div className="whitespace-pre-wrap break-words text-sm">
          {message.text}
        </div>
        
        {message.citation && (
          <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 italic border-t border-gray-300 dark:border-gray-600 pt-1">
            <div className="font-medium">Source:</div>
            {message.citation}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage; 