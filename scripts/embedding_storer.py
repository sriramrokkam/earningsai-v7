from db_connection import get_db_connection, release_db_connection
from langchain_community.vectorstores import HanaDB
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from logger_setup import get_logger
import os
import hashlib
from pdf_processor import process_all_pdfs
from excel_processor import process_all_excel
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
from env_config import TABLE_NAMES, EMBEDDING_MODEL
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details,extract_aicore_credentials
import json
import requests


logger = get_logger()


#$$$ SOC: 28.05.25 -- Initialize AIC Credentials --- $$$#
logger.info ("====> embeddings_storer.py -> AIC CREDENTIALS <====")

# Load VCAP_SERVICES from environment
vcap_services = os.environ.get("VCAP_SERVICES")

# Extract destination service credentials
destination_service_credentials = get_destination_service_credentials(vcap_services)

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


#-------------------------------- READ AIC Configuration -------------------------------------

# variables for AIC credentials
AIC_CREDENTIALS = None

# Get AIC details from Dest Services
dest_AIC = "EARNINGS_AIC"
aicore_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    dest_AIC,
    oauth_token
)
    
# Extract AIC Details
AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
logger.info("AIC Credential", AIC_CREDENTIALS)

#$$$ EOC: 28.05.25 -- Initialize AIC Credentials --- $$$#




def get_existing_file_info_from_db():
    """Retrieve unique file info from three HANA tables."""
    logger.info("Fetching existing file info from all tables")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = f"""
        SELECT 
            DISTINCT JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file,
            JSON_VALUE(VEC_META, '$.content_hash' RETURNING NVARCHAR(32)) as content_hash
        FROM {TABLE_NAMES['transcript']}
        WHERE VEC_META IS NOT NULL
        UNION ALL
        SELECT 
            DISTINCT JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file,
            JSON_VALUE(VEC_META, '$.content_hash' RETURNING NVARCHAR(32)) as content_hash
        FROM {TABLE_NAMES['non_transcript']}
        WHERE VEC_META IS NOT NULL
        UNION ALL
        SELECT 
            DISTINCT JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file,
            JSON_VALUE(VEC_META, '$.content_hash' RETURNING NVARCHAR(32)) as content_hash
        FROM {TABLE_NAMES['excel_non_transcript']}
        WHERE VEC_META IS NOT NULL
        """
        cursor.execute(query)
        file_info = {row[0]: row[1] for row in cursor.fetchall()}
        logger.info(f"Found {len(file_info)} unique files across all tables")
        return file_info
    except Exception as e:
        logger.error(f"Error fetching file info from database: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)

def compute_file_hash(file_path):
    """Calculate MD5 hash of a file's content."""
    logger.debug(f"Computing hash for file: {file_path}")
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    file_hash = hash_md5.hexdigest()
    logger.debug(f"Hash computed: {file_hash}")
    return file_hash

def store_embeddings(vector_store, texts, embeddings, metadatas):
    """Store embeddings and metadata in the HANA database."""
    logger.info(f"Storing {len(embeddings)} embeddings in HANA DB table {vector_store.table_name}")

    # Commenting out the original dictionary-based logic
    # def final_clean_metadata(meta):
    #     meta = dict(meta)
    #     sf = meta.get("source_file", "unknown")
    #     logger.debug(f"Initial source_file value: {sf} (type: {type(sf)})")
    #     if isinstance(sf, dict):
    #         logger.error(f"Final cleaning: source_file is dict, converting to JSON string: {sf}")
    #         meta["source_file"] = json.dumps(sf)
    #     elif not isinstance(sf, str):
    #         logger.error(f"Final cleaning: source_file is not string, converting to str: {sf} (type: {type(sf)})")
    #         meta["source_file"] = str(sf)
    #     # Ensure all metadata fields are strings where required
    #     if "content_hash" in meta and not isinstance(meta["content_hash"], str):
    #         meta["content_hash"] = str(meta["content_hash"])
    #     if "page" in meta and not isinstance(meta["page"], int):
    #         try:
    #             meta["page"] = int(meta["page"])
    #         except Exception:
    #             meta["page"] = 0
    #     logger.debug(f"Cleaned metadata: {meta}")
    #     return meta

    # New flat structure logic
    def validate_metadata_tuple(meta_tuple):
        source_file, content_hash, page = meta_tuple
        if not isinstance(source_file, str):
            logger.error(f"source_file is not a string, converting: {source_file} (type: {type(source_file)})")
            source_file = str(source_file)
        if not isinstance(content_hash, str):
            logger.error(f"content_hash is not a string, converting: {content_hash} (type: {type(content_hash)})")
            content_hash = str(content_hash)
        if not isinstance(page, int):
            try:
                page = int(page)
            except Exception:
                logger.error(f"page is not an integer, defaulting to 0: {page} (type: {type(page)})")
                page = 0
        return (source_file, content_hash, page)

    filtered_metadatas = []
    filtered_texts = []
    filtered_embeddings = []
    for i, meta in enumerate(metadatas):
        try:
            logger.debug(f"Processing metadata at index {i}: {meta}")
            # Assuming meta is now a tuple (source_file, content_hash, page)
            clean_meta = validate_metadata_tuple(meta)
            filtered_metadatas.append(clean_meta)
            filtered_texts.append(texts[i])
            filtered_embeddings.append(embeddings[i])
        except Exception as e:
            logger.error(f"Skipping metadata at index {i} due to validation error: {e}. Metadata: {meta}")
            continue
    if not filtered_texts:
        logger.warning("No valid embeddings to store after validation. Skipping DB insert.")
        return
    try:
        logger.debug(f"Filtered texts: {filtered_texts}")
        logger.debug(f"Filtered embeddings: {filtered_embeddings}")
        logger.debug(f"Filtered metadatas: {filtered_metadatas}")
        vector_store.add_texts(
            texts=filtered_texts,
            embeddings=filtered_embeddings,
            metadatas=filtered_metadatas
        )
        logger.info(f"✅ Successfully stored {len(filtered_embeddings)} embeddings in {vector_store.table_name}")
    except Exception as e:
        logger.error(f"❌ Error storing embeddings in {vector_store.table_name}: {e}")
        # Do not raise, allow process to continue

def delete_embeddings_for_file(table_name, source_file):
    """Remove embeddings for a specific file from a table."""
    logger.info(f"Deleting embeddings for {source_file} from {table_name}")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = f"""
        DELETE FROM {table_name}
        WHERE JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) = ?
        """
        cursor.execute(query, (source_file,))
        deleted_count = cursor.rowcount
        conn.commit()
        logger.info(f"🗑️ Deleted {deleted_count} embeddings for {source_file} from {table_name}")
        return deleted_count
    except Exception as e:
        logger.error(f"❌ Error deleting embeddings for {source_file} from {table_name}: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)

def remove_duplicates(table_name):
    """Eliminate duplicate entries in a specified table."""
    logger.info(f"Removing duplicate entries from vector store table: {table_name}")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        select_query = f"""
        SELECT 
            VEC_TEXT, 
            VEC_VECTOR, 
            VEC_META,
            JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file,
            JSON_VALUE(VEC_META, '$.page' RETURNING NVARCHAR(5000)) as page,
            JSON_VALUE(VEC_META, '$.content_hash' RETURNING NVARCHAR(32)) as content_hash
        FROM {table_name}
        """
        cursor.execute(select_query)
        all_records = cursor.fetchall()
        logger.info(f"Retrieved {len(all_records)} total records from {table_name}")
        if not all_records:
            logger.info(f"No records to deduplicate in {table_name}")
            return 0
        unique_records = {}
        for record in all_records:
            vec_text, vec_vector, vec_meta, source_file, page, content_hash = record
            source_file = source_file or "unknown"
            page = int(page) if page and page.isdigit() else 0
            content_hash = content_hash or "unknown"
            key = (str(vec_text), source_file, page, content_hash)
            if key not in unique_records:
                unique_records[key] = (vec_text, vec_vector, vec_meta)
        logger.info(f"Truncating table {table_name}")
        cursor.execute(f"TRUNCATE TABLE {table_name}")
        logger.info(f"Inserting {len(unique_records)} unique records into {table_name}")
        insert_query = f"""
        INSERT INTO {table_name} (VEC_TEXT, VEC_VECTOR, VEC_META)
        VALUES (?, ?, ?)
        """
        for vec_text, vec_vector, vec_meta in unique_records.values():
            cursor.execute(insert_query, (vec_text, vec_vector, vec_meta))
        records_removed = len(all_records) - len(unique_records)
        conn.commit()
        logger.info(f"🧹 Removed {records_removed} duplicates from {table_name}")
        return records_removed
    except Exception as e:
        logger.error(f"❌ Error removing duplicates from {table_name}: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)

def process_and_store_embeddings(directory_path, force_overwrite_files=None, model_name=EMBEDDING_MODEL):
    """Process files from a single directory and store embeddings based on file type."""
    logger.info(f"Processing files from {directory_path} with model: {model_name}")
    if force_overwrite_files is None:
        force_overwrite_files = set()
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist")
        return
    existing_file_info = get_existing_file_info_from_db()
    # Separate PDF and Excel files
    pdf_files_info = {}
    pdf_files_to_process = set()
    excel_files_info = {}
    excel_files_to_process = set()
    for f in os.listdir(directory_path):
        if not isinstance(f, str):
            logger.warning(f"Skipping non-string filename in directory: {f} (type: {type(f)})")
            continue
        file_path = os.path.join(directory_path, f)
        current_hash = compute_file_hash(file_path)
        if f.lower().endswith('.pdf'):
            pdf_files_info[f] = current_hash
            if f not in existing_file_info:
                logger.info(f"New PDF file detected: {f}")
                pdf_files_to_process.add(f)
            elif existing_file_info[f] != current_hash:
                logger.info(f"Content changed for PDF {f}: old hash {existing_file_info[f]}, new hash {current_hash}")
                pdf_files_to_process.add(f)
            elif f in force_overwrite_files:
                logger.info(f"Forced overwrite requested for PDF {f}")
                pdf_files_to_process.add(f)
        elif f.lower().endswith(('.xlsx', '.xls')):
            excel_files_info[f] = current_hash
            if f not in existing_file_info:
                logger.info(f"New Excel file detected: {f}")
                excel_files_to_process.add(f)
            elif existing_file_info[f] != current_hash:
                logger.info(f"Content changed for Excel {f}: old hash {existing_file_info[f]}, new hash {current_hash}")
                excel_files_to_process.add(f)
            elif f in force_overwrite_files:
                logger.info(f"Forced overwrite requested for Excel {f}")
                excel_files_to_process.add(f)
    logger.info(f"Found {len(pdf_files_info)} PDF files, {len(pdf_files_to_process)} need processing")
    logger.info(f"Found {len(excel_files_info)} Excel files, {len(excel_files_to_process)} need processing")
    if not pdf_files_to_process and not excel_files_to_process:
        logger.info("No new or changed files to process")
        return
    # Process PDFs in parallel
    def process_pdf_task():
        transcript_embeddings, non_transcript_embeddings = process_all_pdfs(directory_path, model_name)
        # Debug: log type of source_file metadata
        for doc, emb in transcript_embeddings + non_transcript_embeddings:
            sf = doc.metadata.get("source_file")
            if not isinstance(sf, str):
                logger.error(f"source_file metadata is not a string: {sf} (type: {type(sf)})")
        filtered_transcript_embeddings = [
            (doc, emb) for doc, emb in transcript_embeddings 
            if isinstance(doc.metadata.get("source_file"), str) and doc.metadata.get("source_file") in pdf_files_to_process
        ]
        filtered_non_transcript_embeddings = [
            (doc, emb) for doc, emb in non_transcript_embeddings 
            if isinstance(doc.metadata.get("source_file"), str) and doc.metadata.get("source_file") in pdf_files_to_process
        ]
        logger.info(f"Filtered to {len(filtered_transcript_embeddings)} PDF transcript embeddings and "
                    f"{len(filtered_non_transcript_embeddings)} PDF non-transcript embeddings for processing")
        return filtered_transcript_embeddings, filtered_non_transcript_embeddings

    # Process Excel files in parallel
    def process_excel_task():
        all_excel_embeddings = process_all_excel(directory_path, model_name)
        filtered_excel_embeddings = [(doc, emb) for doc, emb in all_excel_embeddings 
                                    if doc.metadata.get("source_file") in excel_files_to_process]
        logger.info(f"Filtered to {len(filtered_excel_embeddings)} Excel embeddings for processing")
        return filtered_excel_embeddings

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_pdf = executor.submit(process_pdf_task)
        future_excel = executor.submit(process_excel_task)
        filtered_transcript_embeddings, filtered_non_transcript_embeddings = future_pdf.result()
        filtered_excel_embeddings = future_excel.result()

    # SOC: SRIRAM 28.05.2025 -- Proxy Initialization
    #embedding_model = init_embedding_model(model_name)
    from gen_ai_hub.proxy import GenAIHubProxyClient
    logger.info("Embedding_Storer: AIC", {AIC_CREDENTIALS})

    proxy_client = GenAIHubProxyClient(
                                base_url = AIC_CREDENTIALS['aic_base_url'],
                                auth_url = AIC_CREDENTIALS['aic_auth_url'],
                                client_id = AIC_CREDENTIALS['clientid'],
                                client_secret = AIC_CREDENTIALS['clientsecret'],
                                resource_group = AIC_CREDENTIALS['resource_group']
                                )

    embedding_model = init_embedding_model(model_name = EMBEDDING_MODEL, proxy_client=proxy_client)

    # EOC: SRIRAM 28.05.2025 -- Proxy Initialization
    # embedding_model = init_embedding_model(model_name)
    conn = get_db_connection()
    # Store PDF transcript embeddings
    pdf_transcript_table = TABLE_NAMES['transcript']
    if filtered_transcript_embeddings:
        transcript_store = HanaDB(
            connection=conn,
            embedding=embedding_model,
            table_name=pdf_transcript_table,
            content_column="VEC_TEXT",
            metadata_column="VEC_META",
            vector_column="VEC_VECTOR"
        )
        for source_file in pdf_files_to_process:
            if source_file in existing_file_info or source_file in force_overwrite_files:
                delete_embeddings_for_file(pdf_transcript_table, source_file)
        transcript_texts = [doc.page_content for doc, _ in filtered_transcript_embeddings]
        transcript_embeds = [embedding for _, embedding in filtered_transcript_embeddings]
        # Clean metadata: ensure 'source_file' is always a string
        def clean_metadata(meta):
            sf = meta.get("source_file")
            if isinstance(sf, dict):
                logger.error(f"Fixing source_file metadata from dict to string: {sf}")
                meta = dict(meta)
                meta["source_file"] = json.dumps(sf)  # Use JSON string for dicts
            elif not isinstance(sf, str):
                meta = dict(meta)
                meta["source_file"] = str(sf)
            return meta

        transcript_metadatas = [
            clean_metadata({**doc.metadata, "content_hash": pdf_files_info.get(doc.metadata.get("source_file"))})
            for doc, _ in filtered_transcript_embeddings
        ]
        # Add this debug log to catch any dicts in transcript_metadatas
        for meta in transcript_metadatas:
            if isinstance(meta.get("source_file"), dict):
                logger.error(f"FATAL: source_file is still a dict in transcript_metadatas: {meta}")
        non_transcript_metadatas = [
            clean_metadata({**doc.metadata, "content_hash": pdf_files_info.get(doc.metadata.get("source_file"))})
            for doc, _ in filtered_non_transcript_embeddings
        ]
        for meta in non_transcript_metadatas:
            if isinstance(meta.get("source_file"), dict):
                logger.error(f"FATAL: source_file is still a dict in non_transcript_metadatas: {meta}")
        excel_non_transcript_metadatas = [
            clean_metadata({**doc.metadata, "content_hash": excel_files_info.get(doc.metadata.get("source_file"))})
            for doc, _ in filtered_excel_embeddings
        ]
        for meta in excel_non_transcript_metadatas:
            if isinstance(meta.get("source_file"), dict):
                logger.error(f"FATAL: source_file is still a dict in excel_non_transcript_metadatas: {meta}")
        if transcript_texts:

            #$$$$ SR: SOC 1:
            logger.info(f"Logpoint 1: Transcripts embedding type: {type(transcript_embeds[0])}, value: {transcript_embeds[0]}")
            #$$$ SR: EOC 1

            store_embeddings(transcript_store, transcript_texts, transcript_embeds, transcript_metadatas)
    # Store PDF non-transcript embeddings
    pdf_non_transcript_table = TABLE_NAMES['non_transcript']
    if filtered_non_transcript_embeddings:
        non_transcript_store = HanaDB(
            connection=conn,
            embedding=embedding_model,
            table_name=pdf_non_transcript_table,
            content_column="VEC_TEXT",
            metadata_column="VEC_META",
            vector_column="VEC_VECTOR"
        )
        for source_file in pdf_files_to_process:
            if source_file in existing_file_info or source_file in force_overwrite_files:
                delete_embeddings_for_file(pdf_non_transcript_table, source_file)
        non_transcript_texts = [doc.page_content for doc, _ in filtered_non_transcript_embeddings]
        non_transcript_embeds = [embedding for _, embedding in filtered_non_transcript_embeddings]
        non_transcript_metadatas = [
            clean_metadata({**doc.metadata, "content_hash": pdf_files_info.get(doc.metadata.get("source_file"))})
            for doc, _ in filtered_non_transcript_embeddings
        ]

        #$$$$ SR: SOC LP2:
        logger.info(f"logpoint 2: non transcript type: {type(non_transcript_embeds[0])}, value: {non_transcript_embeds[0]}")
        logger.info(f"logpoint 2: value: {non_transcript_texts}")
        #$$$ SR: EOC LP2

        if non_transcript_texts:
            store_embeddings(non_transcript_store, non_transcript_texts, non_transcript_embeds, non_transcript_metadatas)
    # Store Excel embeddings in non-transcript table only
    excel_non_transcript_table = TABLE_NAMES['excel_non_transcript']
    if filtered_excel_embeddings:
        excel_non_transcript_store = HanaDB(
            connection=conn,
            embedding=embedding_model,
            table_name=excel_non_transcript_table,
            content_column="VEC_TEXT",
            metadata_column="VEC_META",
            vector_column="VEC_VECTOR"
        )
        for source_file in excel_files_to_process:
            if source_file in existing_file_info or source_file in force_overwrite_files:
                delete_embeddings_for_file(excel_non_transcript_table, source_file)
        excel_non_transcript_texts = [doc.page_content for doc, _ in filtered_excel_embeddings]
        excel_non_transcript_embeds = [embedding for _, embedding in filtered_excel_embeddings]
        excel_non_transcript_metadatas = [
            clean_metadata({**doc.metadata, "content_hash": excel_files_info.get(doc.metadata.get("source_file"))})
            for doc, _ in filtered_excel_embeddings
        ]
        if excel_non_transcript_texts:
            store_embeddings(excel_non_transcript_store, excel_non_transcript_texts, excel_non_transcript_embeds, excel_non_transcript_metadatas)
    release_db_connection(conn)
    logger.info("Starting duplicate removal process")
    pdf_transcript_removed = remove_duplicates(pdf_transcript_table)
    pdf_non_transcript_removed = remove_duplicates(pdf_non_transcript_table)
    excel_non_transcript_removed = remove_duplicates(excel_non_transcript_table)
    total_removed = pdf_transcript_removed + pdf_non_transcript_removed + excel_non_transcript_removed
    logger.info(f"Total duplicates removed: {total_removed}")