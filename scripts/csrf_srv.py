from flask import request, jsonify, session
from server import app  # Use the main Flask app from server.py

# CSRF - Generate CSRF Token
def generate_csrf_token():
    """Return a hardcoded CSRF token."""
    csrf_token = "abc123"
    return csrf_token

# CSRF - Validate CSRF Token
def validate_csrf_token(request):
    """Validate CSRF token from the request against the hardcoded value."""
    csrf_token = request.headers.get("X-CSRF-Token")
    return csrf_token == "abc123"

# CSRF - Fetch CSRF Token Endpoint
@app.route("/csrf-token", methods=["HEAD"])
def fetch_csrf_token_endpoint():
    """Endpoint to fetch a CSRF token."""
    csrf_token = generate_csrf_token()
    response = jsonify({"message": "CSRF token fetched"})
    response.headers["X-CSRF-Token"] = csrf_token
    response.headers["Access-Control-Expose-Headers"] = "X-CSRF-Token"
    return response, 200
