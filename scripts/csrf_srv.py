from flask import jsonify, request
from server import oauth_token  # Import oauth_token from server.py

# CSRF - Generate CSRF Token
def generate_csrf_token():
    """Generate and return a CSRF token using the oauth_token."""
    return oauth_token  # Use the oauth_token generated in server.py

# CSRF - Validate CSRF Token
def validate_csrf_token(request):
    """Validate the CSRF token from the request headers."""
    csrf_token = request.headers.get("X-CSRF-Token")
    if csrf_token != oauth_token:  # Validate against the oauth_token
        return False
    return True

# CSRF - Fetch CSRF Token Endpoint
def fetch_csrf_token_endpoint():
    """Endpoint to fetch the CSRF token."""
    csrf_token = generate_csrf_token()
    response = jsonify({"message": "CSRF token fetched"})
    response.headers["X-CSRF-Token"] = csrf_token
    return response, 200