import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

# Mock endpoint to simulate CSRF token generation
@app.route('/api/get-csrf-token', methods=['HEAD'])
def get_csrf_token():
    """Simulate CSRF token generation"""
    csrf_token = "mock-csrf-token-12345"  # Mock token for testing
    response = jsonify({"message": "CSRF token fetched"})
    response.headers["X-CSRF-Token"] = csrf_token
    return response, 200

# Endpoint that requires CSRF protection
@app.route('/api/protected-endpoint', methods=['POST'])
def protected_endpoint():
    """Simulate a protected endpoint that requires CSRF token"""
    csrf_token = request.headers.get("X-CSRF-Token")
    if csrf_token != "mock-csrf-token-12345":
        return jsonify({"error": "Invalid CSRF token"}), 403
    return jsonify({"message": "Request successful"}), 200

if __name__ == '__main__':
    # Start the Flask app
    app.run(port=5000, debug=True)