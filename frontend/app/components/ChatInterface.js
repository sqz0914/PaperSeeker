'use client';

import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';

const API_URL = 'http://localhost:8000';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [availableTopics, setAvailableTopics] = useState([]);
  const [isLoadingTopics, setIsLoadingTopics] = useState(true);
  const [isInitializing, setIsInitializing] = useState(true);
  const [welcomeMessageComplete, setWelcomeMessageComplete] = useState(false);
  const [usingRealData, setUsingRealData] = useState(false);
  const messagesEndRef = useRef(null);

  // Fetch data source info when component mounts
  useEffect(() => {
    const fetchDataSourceInfo = async () => {
      try {
        const response = await fetch(`${API_URL}/`);
        if (response.ok) {
          const data = await response.json();
          setUsingRealData(data.papers_loaded > 0);
        }
      } catch (err) {
        console.error('Error fetching data source info:', err);
      }
    };

    fetchDataSourceInfo();
  }, []);

  // Fetch available topics when component mounts
  useEffect(() => {
    const fetchTopics = async () => {
      try {
        const response = await fetch(`${API_URL}/api/topics`);
        if (response.ok) {
          const data = await response.json();
          setAvailableTopics(data.topics || []);
        } else {
          console.error('Failed to fetch topics');
        }
      } catch (err) {
        console.error('Error fetching topics:', err);
      } finally {
        setIsLoadingTopics(false);
      }
    };

    fetchTopics();
  }, []);

  // Stream welcome message when component mounts
  useEffect(() => {
    const fetchWelcomeMessage = async () => {
      setIsInitializing(true);
      
      try {
        // Add welcome message directly to messages array with streaming flag
        const welcomeMessage = {
          id: 1,
          text: '',
          sender: 'bot',
          citation: null,
          isStreaming: true // Flag to identify this as streaming
        };
        
        setMessages([welcomeMessage]);
        
        const response = await fetch(`${API_URL}/api/welcome`);
        
        if (!response.ok) {
          throw new Error('Failed to get welcome message');
        }
        
        // Process the streamed response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let done = false;
        let currentText = '';
        
        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          
          if (done) break;
          
          // Process the chunks of data
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n').filter(line => line.trim());
          
          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              
              switch (data.type) {
                case 'start':
                  // Reset state for new message
                  currentText = '';
                  break;
                  
                case 'chunk':
                  // Add new content to the message and update in the array
                  currentText += data.content;
                  setMessages(prevMessages => {
                    const updatedMessages = [...prevMessages];
                    if (updatedMessages.length > 0) {
                      updatedMessages[0] = {
                        ...updatedMessages[0],
                        text: currentText
                      };
                    }
                    return updatedMessages;
                  });
                  break;
                  
                case 'end':
                  // When streaming is complete, simply update the message without the streaming flag
                  setMessages(prevMessages => {
                    const updatedMessages = [...prevMessages];
                    if (updatedMessages.length > 0) {
                      updatedMessages[0] = {
                        ...updatedMessages[0],
                        text: currentText,
                        isStreaming: false // Remove streaming flag
                      };
                    }
                    return updatedMessages;
                  });
                  setWelcomeMessageComplete(true);
                  break;
              }
            } catch (e) {
              console.error('Error parsing streamed data:', e);
            }
          }
        }
      } catch (err) {
        console.error('Error fetching welcome message:', err);
        
        // Fallback to static message if streaming fails
        setMessages([{
          id: 1,
          text: "Hello! I'm PaperSeeker. Ask me about research papers, and I'll help you find relevant information.",
          sender: 'bot',
          isStreaming: false
        }]);
        setWelcomeMessageComplete(true);
      } finally {
        setIsInitializing(false);
      }
    };

    fetchWelcomeMessage();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]); // Remove streamingMessage dependency since we don't use it anymore

  const handleSendMessage = async (text) => {
    // Don't process if initialization isn't complete
    if (isInitializing || !welcomeMessageComplete) {
      return;
    }
    
    // Add user message
    const newUserMessage = {
      id: messages.length + 1,
      text,
      sender: 'user'
    };
    
    // Add bot message as streaming initially
    const botMessage = {
      id: messages.length + 2,
      text: '',
      sender: 'bot',
      citation: null,
      isStreaming: true
    };
    
    setMessages(prev => [...prev, newUserMessage, botMessage]);
    setIsLoading(true);
    setError(null);
    
    try {
      // Fetch from streaming endpoint
      const response = await fetch(`${API_URL}/api/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: text }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to get a response from the server');
      }
      
      // Process the streamed response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      let done = false;
      let currentText = '';
      let citation = null;
      const botMessageIndex = messages.length + 1; // Index of the bot message we just added
      
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        
        if (done) break;
        
        // Process the chunks of data
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            switch (data.type) {
              case 'start':
                // Reset state for new message
                currentText = '';
                citation = null;
                break;
                
              case 'chunk':
                // Add new content to the message
                currentText += data.content;
                // Update the message in the messages array
                setMessages(prevMessages => {
                  const updatedMessages = [...prevMessages];
                  if (updatedMessages.length > botMessageIndex) {
                    updatedMessages[botMessageIndex] = {
                      ...updatedMessages[botMessageIndex],
                      text: currentText
                    };
                  }
                  return updatedMessages;
                });
                break;
                
              case 'citation':
                // Add citation
                citation = data.content;
                // Update the message in the messages array
                setMessages(prevMessages => {
                  const updatedMessages = [...prevMessages];
                  if (updatedMessages.length > botMessageIndex) {
                    updatedMessages[botMessageIndex] = {
                      ...updatedMessages[botMessageIndex],
                      citation
                    };
                  }
                  return updatedMessages;
                });
                break;
                
              case 'end':
                // When streaming is complete, update the message and mark as not streaming
                setMessages(prevMessages => {
                  const updatedMessages = [...prevMessages];
                  if (updatedMessages.length > botMessageIndex) {
                    updatedMessages[botMessageIndex] = {
                      ...updatedMessages[botMessageIndex],
                      text: currentText,
                      citation,
                      isStreaming: false
                    };
                  }
                  return updatedMessages;
                });
                break;
            }
          } catch (e) {
            console.error('Error parsing streamed data:', e);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching from API:', err);
      setError(err.message);
      
      // Update the bot message to show an error
      setMessages(prevMessages => {
        const updatedMessages = [...prevMessages];
        const botMessageIndex = updatedMessages.length - 1;
        if (botMessageIndex >= 0) {
          updatedMessages[botMessageIndex] = {
            ...updatedMessages[botMessageIndex],
            text: "Sorry, I'm having trouble connecting to the research database. Please try again later.",
            isStreaming: false
          };
        }
        return updatedMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleTopicClick = (topic) => {
    handleSendMessage(`Tell me about ${topic}`);
  };

  return (
    <div className="flex flex-col h-[80vh] max-w-2xl mx-auto border rounded-lg shadow-md overflow-hidden">
      <div className="bg-blue-600 text-white p-4">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold">PaperSeeker</h1>
            <p className="text-sm">Your research paper assistant</p>
          </div>
          {usingRealData && (
            <span className="bg-green-500 text-white text-xs font-medium px-2 py-1 rounded">
              Using Real Data
            </span>
          )}
        </div>
      </div>
      
      <div className="flex-grow p-4 overflow-y-auto bg-white dark:bg-gray-900">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} isStreaming={message.isStreaming || false} />
        ))}
        
        {isLoading && messages.length === 0 && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-200 dark:bg-gray-800 p-3 rounded-lg flex items-center space-x-2">
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
          </div>
        )}
        
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
          disabled={isInitializing || !welcomeMessageComplete} 
        />
        
        {/* Topic suggestions */}
        <div className="mt-3">
          <p className="text-xs text-gray-500 mb-2">
            Try asking about:
          </p>
          <div className="flex flex-wrap gap-2 max-h-20 overflow-y-auto">
            {isLoadingTopics ? (
              <span className="text-xs text-gray-400">Loading topics...</span>
            ) : (
              availableTopics.map(topic => (
                <button
                  key={topic}
                  onClick={() => handleTopicClick(topic)}
                  disabled={isInitializing || !welcomeMessageComplete}
                  className="px-2 py-1 bg-blue-100 hover:bg-blue-200 dark:bg-blue-900 dark:hover:bg-blue-800 rounded-full text-xs transition-colors disabled:opacity-50"
                >
                  {topic}
                </button>
              ))
            )}
          </div>
        </div>
        
        {usingRealData && (
          <div className="mt-3 text-xs text-gray-500">
            <p>
              PaperSeeker is connected to a real academic paper database. 
              You can ask questions like "What are recent advances in COVID vaccines?" or 
              "Find papers about climate change mitigation strategies."
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface; 