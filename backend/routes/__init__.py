# Import all routes for export
from . import paper_routes
from . import search_routes
from . import chat_routes
from . import config_routes

# Export route modules
__all__ = [
    'paper_routes',
    'search_routes',
    'chat_routes',
    'config_routes'
] 