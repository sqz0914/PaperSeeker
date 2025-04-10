from pydantic import BaseModel

class QueryRequest(BaseModel):
    """
    Model representing a query request from the client.
    Used for simple query interfaces and chat functionality.
    """
    query: str 