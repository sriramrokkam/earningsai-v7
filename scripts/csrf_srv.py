import secrets
from flask import request, jsonify, session
from server import app  # Use the main Flask app from server.py

# CSRF - Generate CSRF Token
def generate_csrf_token():
    """Generate a secure random CSRF token and store in session."""
    csrf_token = secrets.token_urlsafe(32)
    session["csrf_token"] = csrf_token
    return csrf_token

# CSRF - Validate CSRF Token
def validate_csrf_token(request):
    """Validate CSRF token from the request against the session."""
    csrf_token = request.headers.get("X-CSRF-Token")
    return csrf_token == session.get("csrf_token")

# CSRF - Fetch CSRF Token Endpoint
@app.route("/csrf-token", methods=["HEAD"])
def fetch_csrf_token_endpoint():
    """Endpoint to fetch a CSRF token."""
    csrf_token = generate_csrf_token()
    response = jsonify({"message": "CSRF token fetched"})
    response.headers["X-CSRF-Token"] = csrf_token
    response.headers["Access-Control-Expose-Headers"] = "X-CSRF-Token"
    return response, 200
