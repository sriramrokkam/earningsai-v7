"""
API Client module for interacting with the Embedding Files API.
This module handles authentication, downloading files, and updating file statuses.
"""
import os
import logging
import requests
import shutil
from typing import Dict, List, Tuple, Optional, Any
from destination_srv import extract_cap_credentials, fetch_destination_details,get_destination_service_credentials, generate_token

logger = logging.getLogger('EarningsAnalysis: api_client')
logger.setLevel(logging.INFO)

### ------------------- Read VCAP Services from CF --------------------###

vcap_services = os.environ.get("VCAP_SERVICES")
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

#-------------------------------- READ CAP Configuration -------------------------------------
try:
    dest_AIC = "EARNINGS_XSUAA"
    cap_details = fetch_destination_details(
        destination_service_credentials['dest_base_url'],
        dest_AIC,
        oauth_token
    )
    logger.info("CAP Destination Details fetched successfully")
    CAP_CREDENTIALS = extract_cap_credentials(cap_details)
    logger.info(f"CAP Credentials: {CAP_CREDENTIALS}")

except Exception as e:
    logger.error(f"Error initializing CAP credentials: {str(e)}")

EMBEDDING_API_BASE_URL = CAP_CREDENTIALS['cap_base_url'] + "/odata/v4/earning-upload-srv/EmbeddingFiles"
AUTH_URL = CAP_CREDENTIALS['cap_auth_url']
CLIENT_ID = CAP_CREDENTIALS['cap_clientid']
CLIENT_SECRET = CAP_CREDENTIALS['cap_clientsecret']

# **** End of Code ***#

# EMBEDDING_API_BASE_URL = "https://standard-chartered-bank-core-foundational-12982zqn-gena4b53cb41.cfapps.ap11.hana.ondemand.com/odata/v4/earning-upload-srv/EmbeddingFiles"
# AUTH_URL = "https://core-foundational-12982zqn.authentication.ap11.hana.ondemand.com/oauth/token"
# CLIENT_ID = "sb-earning-upload!t5156"
# CLIENT_SECRET = "2db20f6f-b4cd-4f9a-b27f-a8f6e15a3c23$CLy31CAgGjo9EpJOQB-HOegl8fxEypDWetJDHPB0Bac="


# Configure logging
logger = logging.getLogger('EarningsAnalysis.APIClient')

def get_auth_token() -> Optional[str]:
    """
    Get OAuth token for API calls
    
    Returns:
        str or None: Access token if successful, None otherwise
    """
    try:
        logger.info("Requesting OAuth token")
        r = requests.post(AUTH_URL, data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }, headers={"Content-Type": "application/x-www-form-urlencoded"})
        
        if r.status_code != 200:
            logger.error(f"Authentication failed: Status {r.status_code}, Response: {r.text}")
            return None
        
        token = r.json().get("access_token")
        if not token:
            logger.error("No access token received")
            return None
        
        logger.info("OAuth token acquired")
        return token
    except Exception as e:
        logger.error(f"Error getting auth token: {str(e)}")
        return None

def update_file_status(file_id: str, status: str = "Completed") -> bool:
    """
    Update the status of a file in the EmbeddingFiles API
    
    Args:
        file_id (str): ID of the file to update
        status (str, optional): New status. Defaults to "Completed".
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        token = get_auth_token()
        if not token:
            return False
        
        # Build the URL for the specific file
        file_url = f"{EMBEDDING_API_BASE_URL}('{file_id}')"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Data for the PATCH request
        data = {
            "status": status
        }
        
        # Make the PATCH request
        logger.info(f"Updating file {file_id} status to {status}")
        response = requests.patch(file_url, headers=headers, json=data)
        
        if response.status_code in [200, 201, 204]:
            logger.info(f"Successfully updated file {file_id} status to {status}")
            return True
        else:
            logger.error(f"Failed to update file {file_id} status: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error updating file status: {str(e)}")
        return False

def get_file_mappings() -> Dict[str, str]:
    """
    Get mapping of filenames to file IDs from the API
    
    Returns:
        dict: Dictionary mapping filenames to file IDs
    """
    try:
        token = get_auth_token()
        if not token:
            return {}
        
        headers = {"Authorization": f"Bearer {token}"}
        file_list_url = f"{EMBEDDING_API_BASE_URL}?$filter=status eq 'Submitted'"
        
        logger.info(f"Fetching file list from {file_list_url}")
        response = requests.get(file_list_url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch file list: Status {response.status_code}, Response: {response.text}")
            return {}
        
        files = response.json().get('value', [])
        
        # Create a mapping of filename to file ID
        file_mappings = {}
        for file in files:
            file_name = file.get('fileName')
            file_id = file.get('ID')
            if file_name and file_id:
                file_mappings[file_name] = file_id
        
        logger.info(f"Found {len(file_mappings)} Submitted files")
        return file_mappings
    except Exception as e:
        logger.error(f"Error getting file mappings: {str(e)}")
        return {}

def download_embedding_files(documents_dir: str, images_dir: str, image_extensions: set) -> List[str]:
    """
    Download Submitted files from the API to local directories
    
    Args:
        documents_dir (str): Path to store document files
        images_dir (str): Path to store image files
        image_extensions (set): Set of file extensions considered as images
    
    Returns:
        list: List of downloaded file paths
    """
    try:
        # Get authentication token
        token = get_auth_token()
        if not token:
            logger.error("Failed to get authentication token")
            return []
        
        # Fetch Submitted files
        headers = {"Authorization": f"Bearer {token}"}
        file_list_url = f"{EMBEDDING_API_BASE_URL}?$filter=status eq 'Approved'"
        logger.info(f"Fetching file list from {file_list_url}")
        r = requests.get(file_list_url, headers=headers)
        
        if r.status_code != 200:
            logger.error(f"Failed to fetch file list: Status {r.status_code}, Response: {r.text}")
            return []
        
        files = r.json().get('value', [])
        logger.info(f"Found {len(files)} Submitted files")
        
        if not files:
            logger.warning("No Submitted files found")
            return []
        
        # Track successful downloads
        successful_downloads = 0
        total_files = len(files)
        downloaded_file_paths = []
        
        # Download each file individually
        for file in files:
            file_name = file.get('fileName', 'unknown_file')
            file_id = file.get('ID')
            if not file_id:
                logger.warning(f"Skipping file with no ID: {file_name}")
                continue
            
            # Quote the file_id for OData string key
            file_url = f"{EMBEDDING_API_BASE_URL}('{file_id}')/content"
            logger.info(f"Downloading file: {file_name} from {file_url}")
            file_response = requests.get(file_url, headers=headers)
            
            if file_response.status_code != 200:
                logger.error(f"Failed to download {file_name}: Status {file_response.status_code}, Response: {file_response.text}")
                continue
            
            file_content = file_response.content
            
            # Save all files to documents_dir
            file_path = os.path.join(documents_dir, file_name)
            with open(file_path, 'wb') as f:
                f.write(file_content)
            logger.info(f"Downloaded: {file_path}")
            successful_downloads += 1
            downloaded_file_paths.append(file_path)
            
            # Move image files to images_dir
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension in image_extensions:
                image_path = os.path.join(images_dir, file_name)
                shutil.move(file_path, image_path)
                logger.info(f"Moved to Images: {image_path}")
        
        if successful_downloads == 0 and total_files > 0:
            logger.error("No files were downloaded successfully")
            return []
        
        logger.info(f"All Submitted files processed: {successful_downloads}/{total_files} downloaded successfully")
        logger.debug(f"Returning downloaded file paths: {downloaded_file_paths}")
        return downloaded_file_paths
    
    except Exception as e:
        logger.error(f"Error in download_embedding_files: {str(e)}")
        return []

def update_completed_files(documents_dir: str, images_dir: str, allowed_extensions: set) -> Dict[str, Any]:
    """
    Update status for all files in the given directories to 'Completed'
    
    Args:
        documents_dir (str): Directory containing document files
        images_dir (str): Directory containing image files
        allowed_extensions (set): Set of allowed file extensions
    
    Returns:
        dict: Results including success count, failed count, and updated file names
    """
    try:
        # Get file mappings from API
        file_mappings = get_file_mappings()
        if not file_mappings:
            logger.warning("No file mappings found or couldn't retrieve mappings")
            return {"success": 0, "failed": 0, "updated_files": []}
        
        # Update status for each file
        success_count = 0
        fail_count = 0
        updated_files = []
        
        # Ensure that only files with 'Approved' status are updated to 'Completed'
        for directory in [documents_dir, images_dir]:
            files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in allowed_extensions]
            logger.info(f"Found {len(files)} files in {directory} to update status")
            for filename in files:
                if filename in file_mappings:
                    file_id = file_mappings[filename]
                    logger.info(f"Updating status for {filename} (ID: {file_id})")
                    if update_file_status(file_id, "Completed"):
                        success_count += 1
                        updated_files.append(filename)
                    else:
                        fail_count += 1
                        logger.error(f"Failed to update status for {filename}")
                else:
                    logger.warning(f"No mapping found for file: {filename}")
        
        return {
            "success": success_count,
            "failed": fail_count,
            "updated_files": updated_files
        }
    except Exception as e:
        logger.error(f"Error updating file statuses: {str(e)}")
        return {"success": 0, "failed": 0, "updated_files": [], "error": str(e)}