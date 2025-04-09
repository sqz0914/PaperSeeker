'use client';

import React from 'react';

const ChatMessage = ({ message, isStreaming }) => {
  const isBot = message.sender === 'bot';
  
  // Only show cursor when this message is actively streaming
  const showCursor = isBot && isStreaming;
  
  return (
    <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
      <div className={`max-w-[80%] p-3 rounded-lg ${
        isBot 
          ? 'bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200 rounded-br-none' 
          : 'bg-blue-500 text-white rounded-bl-none'
      }`}>
        <p className="text-sm">
          {message.text}
          {showCursor && (
            <span className="inline-block w-2 h-4 ml-1 bg-gray-500 dark:bg-gray-400 animate-pulse"></span>
          )}
        </p>
        {message.citation && (
          <div className="mt-2 text-xs border-t pt-1 border-gray-300 dark:border-gray-700">
            <p className="font-medium">Citation:</p>
            <p>{message.citation}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage; 