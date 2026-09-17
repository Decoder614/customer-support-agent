import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app


class VercelPathFixer:
    """WSGI middleware to ensure correct routing on Vercel serverless deployments.
    
    Prevents 404 errors caused by internal path rewrites forwarding '/api/index.py'
    or '/api/index' as PATH_INFO instead of the client's requested URL path.
    """

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path_info = environ.get("PATH_INFO", "")
        # If Vercel passed the serverless file path as PATH_INFO
        if path_info in ("/api/index.py", "/api/index", "/api", "/api/"):
            # Extract the actual client route from headers if available
            orig_uri = (
                environ.get("HTTP_X_FORWARDED_URI")
                or environ.get("HTTP_X_MATCHED_PATH")
                or "/"
            )
            # Remove query parameters from PATH_INFO
            clean_path = orig_uri.split("?")[0]
            environ["PATH_INFO"] = clean_path if clean_path else "/"

        return self.wsgi_app(environ, start_response)


# Apply the path fixer middleware to Flask WSGI app
app.wsgi_app = VercelPathFixer(app.wsgi_app)
