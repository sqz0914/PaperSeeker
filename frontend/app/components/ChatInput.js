'use client';

import React, { useState } from 'react';

const ChatInput = ({ onSendMessage, disabled = false }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 mt-2">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Ask about a research paper..."
        className="flex-grow p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:border-gray-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
        disabled={disabled}
      />
      <button
        type="submit"
        disabled={!message.trim() || disabled}
        className="bg-blue-500 text-white px-4 py-2 rounded-lg disabled:opacity-50 hover:bg-blue-600 transition-colors disabled:cursor-not-allowed text-sm"
      >
        Send
      </button>
    </form>
  );
};

export default ChatInput; 