export async function POST(request) {
  try {
    const requestData = await request.json();
    
    // Get API URL from environment variable or use default
    const apiUrl = process.env.BACKEND_API_URL || 'http://localhost:8000';
    
    // Forward request to backend search endpoint with use_llm=true to get the structured response
    const response = await fetch(`${apiUrl}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: requestData.query,
        top_k: 5,  // Request 10 papers
        use_llm: true
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return Response.json(
        { error: 'Failed to fetch search results', details: errorData },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('Search API error:', error);
    return Response.json({ error: 'Internal server error' }, { status: 500 });
  }
} 