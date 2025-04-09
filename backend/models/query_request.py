from pydantic import BaseModel

class QueryRequest(BaseModel):
    """
    Model representing a query request from the client
    """
    query: str 