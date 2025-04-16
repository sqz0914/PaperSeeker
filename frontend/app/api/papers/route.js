export async function POST(request) {
  try {
    const paperData = await request.json();
    
    // Validate paper data
    if (!paperData.title || !paperData.abstract) {
      return Response.json(
        { error: 'Title and abstract are required fields' },
        { status: 400 }
      );
    }
    
    // Get API URL from environment variable or use default
    const apiUrl = process.env.BACKEND_API_URL || 'http://localhost:8000';
    
    // Call backend API to add paper
    const response = await fetch(`${apiUrl}/papers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(paperData)
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return Response.json(
        { error: 'Failed to add paper', details: errorData },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('Add paper API error:', error);
    return Response.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function GET(request) {
  try {
    // Get API URL from environment variable or use default
    const apiUrl = process.env.BACKEND_API_URL || 'http://localhost:8000';
    
    // Call backend API to get all papers
    const response = await fetch(`${apiUrl}/papers`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return Response.json(
        { error: 'Failed to fetch papers', details: errorData },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('Get papers API error:', error);
    return Response.json({ error: 'Internal server error' }, { status: 500 });
  }
} 