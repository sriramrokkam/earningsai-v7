import os
from pickle import NONE
import time
import logging
from logging.handlers import RotatingFileHandler
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from hdbcli import dbapi  # SAP HANA client library
from db_connection import load_vector_stores
from query_processor import process_query
from embedding_storer import process_and_store_embeddings
from api_client import download_embedding_files, update_completed_files
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details,extract_hana_credentials,extract_aicore_credentials
from xsuaa_srv import get_xsuaa_credentials, verify_jwt_token, require_auth
from fastapi import HTTPException  # Ensure HTTPException is imported for error handling

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()

# Set up logger
logger = logging.getLogger('EarningsAnalysis')
logger.setLevel(logging.INFO)

# Set directories
LOCALPATH = os.getenv('LOCALPATH', os.getcwd())
documents_dir = os.path.join(LOCALPATH, "Documents")
logger.info("Document Directory", documents_dir)
images_dir = os.path.join(LOCALPATH, "Images")
logger.info("Image Library",images_dir)
logs_dir = os.path.join(LOCALPATH, "logs")
logger.info("Log Directory", logs_dir)
os.makedirs(documents_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# Configure logging with rotation
log_file_path = os.path.join(logs_dir, "earnings_analysis.log")
handler = RotatingFileHandler(log_file_path, maxBytes=50 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ---------------------------- XSUAA Authentication Setup ----------------------------
"""
XSUAA authentication is enforced on protected endpoints using the @require_auth decorator.
- The XSUAA credentials are loaded from VCAP_SERVICES and stored in the Flask app context as 'uaa_xsuaa_credentials'.
- The decorator (from xsuaa_srv.py) checks for a Bearer token in the Authorization header and validates it using the credentials.
- If the token is missing, invalid, or lacks the required scope, a 401/403 error is returned.
- Example usage:
    @app.route('/api/chat', methods=['POST'])
    @require_auth
    def chat():
        ...
"""

vcap_services = os.environ.get("VCAP_SERVICES")

uaa_xsuaa_credentials = get_xsuaa_credentials(vcap_services)
logger.info(f"XSUAA Credentials: {uaa_xsuaa_credentials}")
# Store credentials in Flask app context for decorator access
app.uaa_xsuaa_credentials = uaa_xsuaa_credentials

#----------------------------LOAD CF VCAP_SERVICES Variables -----------------------------
# Log start of credential retrieval
logger.info ("***server.py -> GET HANA AND AIC CREDENTIALS FROM DESTINATION SERVICES***")
# # Load VCAP_SERVICES from environment
# vcap_services = os.environ.get("VCAP_SERVICES")
# Extract destination service credentials
destination_service_credentials = get_destination_service_credentials(vcap_services)
logger.info(f"Destination Service Credentials: {destination_service_credentials}")
# Generate OAuth token for destination service
try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
except requests.exceptions.HTTPError as e:
    # Handle HTTP 500 error for invalid client secret
    if e.response is not None and e.response.status_code == 500:
        raise Exception("HTTP 500: Check if the client secret is correct.") from e
    else:
        raise
#-------------------------------- READ HANA DB Configuration -------------------------------------
# Step 2: Get the destination details by passing name and token
dest_HDB = 'EARNINGS_HDB' # make sure this is the correct destination name at btp account.
hana_dest_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    name=dest_HDB,
    token=oauth_token
)
logger.info(f"HANA Destination Details: {hana_dest_details}")
# Step 2.2: Extract HANA connection details
HANA_CONN = GV_HANA_CREDENTIALS = NONE

def initialize_hana_connection():
    """Initialize HANA DB connection using extracted credentials"""
    global HANA_CONN, GV_HANA_CREDENTIALS

    # set the hana connection details
    GV_HANA_CREDENTIALS = extract_hana_credentials(hana_dest_details)
    logger.info(f" HANA Credentials: {GV_HANA_CREDENTIALS}")

    try:
        HANA_CONN = dbapi.connect(
            address=GV_HANA_CREDENTIALS['address'],
            port=GV_HANA_CREDENTIALS['port'],
            user=GV_HANA_CREDENTIALS['user'],
            password=GV_HANA_CREDENTIALS['password'],
            encrypt=True,
            sslValidateCertificate=False
        )
        
        logger.info("Successfully connected to HANA database")
        return True
    except Exception as e:
        logger.error(f"Error initializing HANA connection: {str(e)}")
        return False

# Initialize the HANA Crdentials to Global Variables
initialize_hana_connection()


def store_metadata_in_hana(filename, file_path, file_type, upload_time):
    """Store file metadata in HANA database"""
    try:
        if not HANA_CONN:
            logger.warning("HANA connection not initialized, skipping metadata storage")
            return False

        #cursor = HANA_CONN.cursor()
        #cursor.execute("SET SCHEMA {GV_HANA_CREDENTIALS['schema']}")
        logger.info("Schema set to {GV_HANA_CREDENTIALS['schema']}")

        query = """
            INSERT INTO "FILE_METADATA" (filename, file_path, file_type, upload_time)
            VALUES (?, ?, ?, ?)
        """
        cursor.execute(query, (filename, file_path, file_type, upload_time))
        HANA_CONN.commit()
        logger.info(f"Stored metadata for {filename} in HANA database")
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"Error storing metadata in HANA: {str(e)}")
        return False

#-------------------------------- READ AIC Configuration -------------------------------------

# # Global variables for AIC credentials
AIC_CREDENTIALS = None

def initialize_aic_credentials():
    """Initialize AIC credentials from VCAP_SERVICES"""
    global GV_AIC_CREDENTIALS#, AIC_BASE_URL, AIC_CLIENTID, AIC_CLIENTSECRET, AIC_AUTH_URL, AIC_RESOURCE_GROUP
    
    try:
        dest_AIC = "EARNINGS_AIC"
        aicore_details = fetch_destination_details(
            destination_service_credentials['dest_base_url'],
            dest_AIC,
            oauth_token
        )
        
        logger.info("AIC Destination Details fetched successfully")
        GV_AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
        logger.info(f"Global AIC Credentials: {GV_AIC_CREDENTIALS}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing AIC credentials: {str(e)}")
        return False

# Initialize the AIC Credentials
initialize_aic_credentials()

# ### EOC: SRIRAM ROKKAM 23.05.2025###

# Load vector stores
logger.info("Loading vector stores")
transcript_store, non_transcript_store, excel_non_transcript_store = load_vector_stores(AIC_CREDENTIALS=GV_AIC_CREDENTIALS)
if transcript_store is None or non_transcript_store is None:
    logger.error("Failed to load required vector stores (transcript or non-transcript)")
    transcript_store = non_transcript_store = None
if excel_non_transcript_store is None:
    logger.warning("Failed to load excel_non_transcript_store; proceeding without Excel support")
else:
    logger.info("All vector stores loaded successfully")

# Configuration
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.doc', '.xlsx', '.jpg', '.png', '.jpeg'}
IMAGE_EXTENSIONS = {'.jpg', '.png', '.jpeg'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
UPLOAD_LIMIT = 30
upload_counts = {}

def allowed_file(filename):
    """Check if a file has an allowed extension"""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health_check', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    logger.info("Health check accessed")
    status_info = {
        "status": "Server is running",
        "aic_credentials_loaded": GV_AIC_CREDENTIALS is not None,
        "hana_connected": HANA_CONN is not None
    }
    return jsonify(status_info), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get detailed status information"""
    status = {
        "server_status": "running",
        "vector_stores": {
            "transcript_store": transcript_store is not None,
            "non_transcript_store": non_transcript_store is not None,
            "excel_non_transcript_store": excel_non_transcript_store is not None
        },
        "aic_configuration": {
            "credentials_loaded": GV_AIC_CREDENTIALS is not None,
            "base_url_configured": GV_AIC_CREDENTIALS['aic_base_url'] is not None,
            "auth_url_configured": GV_AIC_CREDENTIALS['aic_auth_url'] is not None
        },
        "hana_configuration": {
            "connected": HANA_CONN is not None,
            "credentials_loaded": GV_HANA_CREDENTIALS is not None,
           # "schema_configured": GV_HANA_CREDENTIALS['schema'] is not None
        }
    }
    return jsonify(status), 200



@app.route('/api/chat', methods=['POST'])
@require_auth
def chat():
    """Process chat queries and return responses"""
    logger.info("Chat endpoint accessed")
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")
        if not data or 'message' not in data:
            logger.warning("Missing 'message' in request")
            return jsonify({"error": "Missing 'message' in request body"}), 400

        user_input = data.get('message', '')
        result = process_query(
            user_input,
            transcript_store=transcript_store,
            non_transcript_store=non_transcript_store,
            excel_non_transcript_store=excel_non_transcript_store if excel_non_transcript_store is not None else None
        )
        return jsonify({"result": result}), 200
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_file():
    """Handle file uploads and store metadata in HANA"""
    try:
        client_ip = request.remote_addr
        current_time = int(time.time())
        hour_ago = current_time - 3600
        upload_counts[client_ip] = [t for t in upload_counts.get(client_ip, []) if t > hour_ago]
        
        if len(upload_counts.get(client_ip, [])) >= UPLOAD_LIMIT:
            logger.warning(f"Upload limit reached for IP {client_ip}")
            return jsonify({"error": "Upload limit reached"}), 429
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"Unsupported file type: {file.filename}")
            return jsonify({"error": "Unsupported file type. Only PDF, TXT, DOCX, DOC, XLSX, JPG, PNG, JPEG allowed."}), 400
        
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        target_dir = images_dir if file_ext in IMAGE_EXTENSIONS else documents_dir
        file_path = os.path.join(target_dir, filename)
        
        if os.path.exists(file_path):
            overwrite = request.form.get('overwrite', 'false').lower() == 'true'
            if not overwrite:
                logger.info(f"File exists: {filename}")
                return jsonify({"exists": True, "message": f"File '{filename}' exists. Overwrite?"}), 200
        
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size} bytes")
            return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE//1024//1024}MB"}), 400
        
        file.save(file_path)
        upload_counts.setdefault(client_ip, []).append(current_time)
        logger.info(f"File uploaded: {filename} to {target_dir}")

        # Store metadata in HANA
        store_metadata_in_hana(
            filename=filename,
            file_path=file_path,
            file_type=file_ext,
            upload_time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
        )

        return jsonify({"message": f"File '{filename}' uploaded to {'Images' if file_ext in IMAGE_EXTENSIONS else 'Documents'} folder"}), 200
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({"error": f"Error uploading: {str(e)}"}), 500

@app.route('/api/generate-embeddings', methods=['POST'])
@require_auth
def generate_embeddings():
    """
    Generate embeddings for downloaded files and update their status to 'Completed'
    """
    try:
        # Step 1: Download files from the API
        logger.info("Starting file download from API")
        download_success, download_count = download_embedding_files(documents_dir, images_dir, IMAGE_EXTENSIONS)
        if not download_success:
            logger.error("Failed to download files from API")
            return jsonify({"error": "Failed to download files from API"}), 500

        # Step 2: Get list of files in directories after download
        uploaded_files = set()
        for directory in [documents_dir, images_dir]:
            files = [f for f in os.listdir(directory) if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
            for f in files:
                uploaded_files.add((os.path.join(directory, f), os.path.getmtime(os.path.join(directory, f))))

        if not uploaded_files:
            logger.warning("No files found after download")
            return jsonify({"error": "No files to process after download"}), 400

        # Step 3: Process and store embeddings
        logger.info(f"Processing {len(uploaded_files)} files for embeddings")
        force_overwrite_files = set()
        process_and_store_embeddings(documents_dir, force_overwrite_files)

        # Step 4: Reload vector stores
        global transcript_store, non_transcript_store, excel_non_transcript_store
        transcript_store, non_transcript_store, excel_non_transcript_store = load_vector_stores()
        logger.info("Embeddings generated successfully")

        # Step 5: Update status for each file
        update_results = update_completed_files(documents_dir, images_dir, ALLOWED_EXTENSIONS)

        # Return status
        logger.info(f"Completed embeddings generation and status updates. Success: {update_results['success']}, Failed: {update_results['failed']}")
        return jsonify({
            "message": "Embeddings generated and file statuses updated",
            "download_success": download_success,
            "files_downloaded": download_count,
            "files_processed": len(uploaded_files),
            "embeddings_generated": True,
            "files_updated": update_results['success'],
            "files_failed": update_results['failed'],
            "updated_files": update_results['updated_files']
        }), 200
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return jsonify({"error": f"Error generating embeddings: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on port {port}")
    for rule in app.url_map.iter_rules():
        logger.info(f"Registered rule: {rule}")
    app.run(host='0.0.0.0', port=port, debug=False)