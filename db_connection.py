import os
import json
import threading
import logging
from gen_ai_hub.proxy.core import proxy_clients
from hdbcli import dbapi
from langchain_community.vectorstores import HanaDB
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from logger_setup import get_logger
from dotenv import load_dotenv
from env_config import TABLE_NAMES, EMBEDDING_MODEL
#from server import GV_AIC_CREDENTIALS

# Load environment variables from .env file
#load_dotenv()
logger = get_logger()  # Initialize logger for this module

# Global variable for HANA DB configuration (dictionary)
HANA_CREDENTIALS = None  # Will hold the full credentials dict

#ORCHESTRATION_SERVICE_URL = os.environ.get('ORCHESTRATION_SERVICE_URL')   

# --- HANA CREDENTIALS FROM DESTINATION SERVICES ---
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials, extract_aicore_credentials

# Load VCAP_SERVICES from environment (Cloud Foundry service bindings)
vcap_services = os.environ.get("VCAP_SERVICES")
logger.info("===>DB_Connections => GET HANA CREDENTIALS FROM DESTINATION SERVICES<===")

# Extract destination service credentials from VCAP_SERVICES
# This will provide the auth URL, client ID/secret, and base URL for destination service

# Parse destination service credentials
destination_service_credentials = get_destination_service_credentials(vcap_services)
logger.info(f"Destination Service Credentials: {destination_service_credentials}")

# Generate OAuth token for destination service using client credentials
try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
    logger.info("OAuth token generated successfully for destination service.")
except Exception as e:
    logger.error(f"Error generating OAuth token: {str(e)}")
    oauth_token = None

# Get the destination details for the HANA DB by passing name and token
if oauth_token:
    dest_HDB = 'EARNINGS_HDB'  # Destination name for HANA DB (update as needed)

    hana_dest_details = fetch_destination_details(
        uri=destination_service_credentials['dest_base_url'],
        name=dest_HDB,
        token=oauth_token
    )
    logger.info(f"HANA Destination Details: {hana_dest_details}")

    # Extract HANA connection details from destination details
    HANA_CREDENTIALS = extract_hana_credentials(hana_dest_details)
    logger.info(f"HANA_CREDENTIALS: {HANA_CREDENTIALS}")

else:
    logger.warning("OAuth token not available; HANA credentials or AIC Credentials not initialized.")

# Custom Connection Pool Implementation
class ConnectionPool:
    def __init__(self, max_connections=20):
        self.max_connections = max_connections
        self.pool = []
        self.lock = threading.Lock()

    def get_connection(self):
        """Fetch a connection from the pool or create a new one if necessary."""
        with self.lock:
            if self.pool:
                logger.debug("Reusing connection from pool")
                return self.pool.pop()
            else:
                logger.debug("Creating a new connection")
                return self._create_connection()

    def release_connection(self, conn):
        """Release a connection back to the pool."""
        with self.lock:
            if conn and len(self.pool) < self.max_connections:
                self.pool.append(conn)
                logger.debug("Connection released back to pool")
            else:
                if conn:
                    conn.close()
                    logger.debug("Connection closed as pool is full or conn is None")
                else:
                    logger.debug("No connection to release")

    def _create_connection(self):
        """Create a new database connection."""
        try:
            # Use HANA_CREDENTIALS dictionary directly
            if not HANA_CREDENTIALS or not all([HANA_CREDENTIALS.get(k) for k in ['address', 'user', 'password', 'port', 'schema']]) or HANA_CREDENTIALS['address'] == 'default-hana-host':
                logger.warning("HANA credentials not properly initialized; skipping connection creation")
                return None
            conn = dbapi.connect(
                address=HANA_CREDENTIALS['address'],
                port=int(HANA_CREDENTIALS['port']),
                user=HANA_CREDENTIALS['user'],
                password=HANA_CREDENTIALS['password']
            )
            cursor = conn.cursor()
            #cursor.execute(f"SET SCHEMA {HANA_CREDENTIALS['schema']}")
            logger.info(f"Schema set to {HANA_CREDENTIALS['schema']}")
            logger.info("Database connection established successfully")
            return conn
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            return None

    def close_all_connections(self):
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.pool:
                try:
                    conn.close()
                    logger.info("Closed connection from pool")
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self.pool.clear()

# Initialize the global connection pool
connection_pool = ConnectionPool(max_connections=20)

def get_db_connection():
    """Fetch a connection from the global connection pool."""
    logger.debug("Fetching connection from pool")
    return connection_pool.get_connection()

def release_db_connection(conn):
    """Release a connection back to the global connection pool."""
    if conn:
        logger.debug("Releasing connection back to pool")
        connection_pool.release_connection(conn)
    else:
        logger.debug("No connection to release")

def close_all_db_connections():
    """Close all connections in the global connection pool."""
    logger.info("Closing all database connections in the pool")
    connection_pool.close_all_connections()

def load_vector_stores(bank_name: str = None, AIC_CREDENTIALS = None):
    """Initialize transcript and non-transcript vector stores, including Excel non-transcripts, optionally filtered by bank name."""
    logger.info(f"Loading vector stores with bank_name filter: {bank_name if bank_name else 'None'}")
    conn = None
    cursor = None
    try:
        # Initialize embedding model with ORCHESTRATION_SERVICE_URL
        try:
            logger.info('Embedding Model', EMBEDDING_MODEL)
            logger.info('AIC Credentials', AIC_CREDENTIALS)
            #SOC: SRIRAM 26.05.25
            from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
            from gen_ai_hub.proxy import GenAIHubProxyClient

            proxy_client = GenAIHubProxyClient(
                                            base_url = AIC_CREDENTIALS['aic_base_url'],
                                            auth_url = AIC_CREDENTIALS['aic_auth_url'],
                                            client_id = AIC_CREDENTIALS['clientid'],
                                            client_secret = AIC_CREDENTIALS['clientsecret'],
                                            resource_group = AIC_CREDENTIALS['resource_group']
                                            )
            embedding_model = init_embedding_model(model_name = EMBEDDING_MODEL, proxy_client=proxy_client)
            logger.info(proxy_client.deployments)
            #EOC: SRIRAM
            
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            logger.warning("Vector stores not initialized due to embedding model failure")
            return None, None, None

        conn = get_db_connection()
        if not conn:
            logger.warning("No database connection available; vector stores not initialized")
            return None, None, None

        # Define table names from config
        transcript_table = TABLE_NAMES['transcript']
        non_transcript_table = TABLE_NAMES['non_transcript']
        excel_non_transcript_table = TABLE_NAMES['excel_non_transcript']

        # Initialize vector stores
        try:
            transcript_store = HanaDB(
                connection=conn,
                embedding=embedding_model,
                table_name=transcript_table,
                content_column="VEC_TEXT",
                metadata_column="VEC_META",
                vector_column="VEC_VECTOR"
            )
            non_transcript_store = HanaDB(
                connection=conn,
                embedding=embedding_model,
                table_name=non_transcript_table,
                content_column="VEC_TEXT",
                metadata_column="VEC_META",
                vector_column="VEC_VECTOR"
            )
            excel_non_transcript_store = HanaDB(
                connection=conn,
                embedding=embedding_model,
                table_name=excel_non_transcript_table,
                content_column="VEC_TEXT",
                metadata_column="VEC_META",
                vector_column="VEC_VECTOR"
            )
            logger.info("Vector stores initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HanaDB vector stores: {str(e)}")
            return None, None, None

        cursor = conn.cursor()
        try:
            # Function to count documents (with optional bank name filter)
            def count_documents(table_name: str, bank_name: str = None) -> int:
                try:
                    if bank_name:
                        query = f"""
                            SELECT COUNT(*) 
                            FROM {table_name} 
                            WHERE JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) LIKE ?
                        """
                        cursor.execute(query, (f'%{bank_name}%',))
                    else:
                        query = f"SELECT COUNT(*) FROM {table_name}"
                        cursor.execute(query)
                    return cursor.fetchone()[0]
                except Exception as e:
                    logger.error(f"Error counting documents in {table_name}: {str(e)}")
                    return 0

            # Count for transcript store
            transcript_count = count_documents(transcript_table, bank_name)
            logger.info(f"Transcript vector store loaded with {transcript_count} documents")
            if transcript_count == 0:
                logger.warning("No documents found in the transcript vector store!")

            # Count for non-transcript store (PDFs)
            non_transcript_count = count_documents(non_transcript_table, bank_name)
            logger.info(f"Non-transcript vector store (PDFs) loaded with {non_transcript_count} documents")
            if non_transcript_count == 0:
                logger.warning("No documents found in the non-transcript vector store (PDFs)!")

            # Count for Excel non-transcript store
            excel_non_transcript_count = count_documents(excel_non_transcript_table, bank_name)
            logger.info(f"Excel non-transcript vector store loaded with {excel_non_transcript_count} documents")
            if excel_non_transcript_count == 0:
                logger.warning("No documents found in the Excel non-transcript vector store!")

            return transcript_store, non_transcript_store, excel_non_transcript_store
        finally:
            if cursor:
                cursor.close()
    except Exception as e:
        logger.error(f"Failed to initialize vector stores: {str(e)}")
        return None, None, None
    finally:
        if conn:
            release_db_connection(conn)

# Close all connections when the application exits
import atexit
atexit.register(close_all_db_connections)