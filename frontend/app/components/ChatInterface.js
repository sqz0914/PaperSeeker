'use client';

import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';

const API_URL = 'http://localhost:8000';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  // Add welcome message on mount
  useEffect(() => {
    setMessages([{
      id: 1,
      text: "Hello! I'm PaperSeeker. Ask me about research papers, and I'll help you find relevant information.",
      sender: 'bot'
    }]);
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (text) => {
    if (!text.trim() || isLoading) return;
    
    // Add user message
    const newUserMessage = {
      id: messages.length + 1,
      text,
      sender: 'user'
    };
    
    // Add temporary bot message showing "thinking" state
    const thinkingMessage = {
      id: messages.length + 2,
      text: 'Analyzing papers and generating response...',
      sender: 'bot',
      isThinking: true
    };
    
    setMessages(prev => [...prev, newUserMessage, thinkingMessage]);
    setIsLoading(true);
    setError(null);
    
    try {
      // Make API request to search endpoint
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: text }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get a response from the server');
      }
      
      const data = await response.json();
      
      // Remove the thinking message
      setMessages(prev => prev.filter(msg => !msg.isThinking));
      
      // Process structured response with papers and conclusion only
      const nextId = messages.length + 2; // +1 for user message, +1 as starting point
      const newMessages = [];
      
      // Add messages for each paper first
      const papers = data.papers || [];
      if (papers.length > 0) {
        // Filter out any duplicate papers by title
        const uniquePapers = papers.filter((paper, index, self) => 
          index === self.findIndex(p => p.title.toLowerCase() === paper.title.toLowerCase())
        );
        
        uniquePapers.forEach((paper, index) => {
          // Format paper content
          const title = (paper.title || '').trim();
          const year = paper.year || '';
          const abstract = (paper.abstract || '').trim();
          const summary = (paper.summary || '').trim();
          const sources = paper.sources || '';
          
          // Format paper content with proper spacing and newlines
          const paperContent = `${title} ${year ? `(${year})` : ''}\n\nAbstract:\n${abstract}\n\nSummary:\n${summary}`;
          
          newMessages.push({
            id: nextId + index,
            text: paperContent,
            sender: 'bot',
            citation: sources,
            messageTitle: `Paper ${index + 1} of ${uniquePapers.length}`,
            paperIndex: index + 1,
            totalPapers: uniquePapers.length
          });
        });
      }
      
      // Add conclusion message at the end
      if (data.conclusion) {
        newMessages.push({
          id: nextId + papers.length,
          text: data.conclusion,
          sender: 'bot',
          messageTitle: papers.length > 0 ? 'Conclusion' : 'No Results'
        });
      } else if (data.introduction && papers.length === 0) {
        // If no papers and no conclusion, use introduction as the error message
        newMessages.push({
          id: nextId,
          text: data.introduction,
          sender: 'bot',
          messageTitle: 'No Results'
        });
      }
      
      // Add all new messages to the chat
      setMessages(prev => {
        // Filter out thinking message and add new structured messages
        const filteredPrev = prev.filter(msg => !msg.isThinking);
        return [...filteredPrev, ...newMessages];
      });
    } catch (err) {
      console.error('Error fetching response:', err);
      
      // Update with error message
      setMessages(prevMessages => {
        // Remove thinking message
        const filteredMessages = prevMessages.filter(msg => !msg.isThinking);
        
        // Add error message
        return [...filteredMessages, {
          id: filteredMessages.length + 1,
          text: "Sorry, I'm having trouble connecting to the research database. Please try again later.",
          sender: 'bot'
        }];
      });
      
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[80vh] max-w-2xl mx-auto border rounded-lg shadow-md overflow-hidden">
      <div className="bg-blue-600 text-white p-4">
        <h1 className="text-xl font-bold">PaperSeeker</h1>
        <p className="text-sm">Your research paper assistant</p>
      </div>
      
      <div className="flex-grow p-4 overflow-y-auto bg-white dark:bg-gray-900">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        
        {error && (
          <div className="text-center my-4">
            <p className="text-red-500 text-sm">
              Connection error. Make sure the backend server is running.
            </p>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="p-4 border-t dark:border-gray-700 bg-white dark:bg-gray-900">
        <ChatInput 
          onSendMessage={handleSendMessage} 
          disabled={isLoading} 
        />
        
        <div className="mt-3 text-xs text-gray-500">
          <p>
            Ask questions like "What are recent advances in COVID vaccines?" or 
            "Find papers about climate change mitigation strategies."
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface; 