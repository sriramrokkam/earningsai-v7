def process_and_store_embeddings(directory_path, force_overwrite_files=None, model_name=EMBEDDING_MODEL):
    """Process files from a single directory and store embeddings based on file type."""
    logger.info(f"Processing files from {directory_path} with model: {model_name}")
    
    if force_overwrite_files is None:
        force_overwrite_files = set()
    
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist")
        return
    
    try:
        existing_file_info = get_existing_file_info_from_db()
        
        # Separate PDF and Excel files
        pdf_files_info = {}
        pdf_files_to_process = set()
        excel_files_info = {}
        excel_files_to_process = set()
        
        # FIX 1: Add file existence check and error handling
        for f in os.listdir(directory_path):
            file_path = os.path.join(directory_path, f)
            
            # Skip if not a file (could be directory)
            if not os.path.isfile(file_path):
                continue
                
            try:
                current_hash = compute_file_hash(file_path)
            except Exception as e:
                logger.error(f"Error computing hash for {f}: {e}")
                continue
            
            if f.lower().endswith('.pdf'):
                pdf_files_info[f] = current_hash
                if f not in existing_file_info:
                    logger.info(f"New PDF file detected: {f}")
                    pdf_files_to_process.add(f)
                elif existing_file_info.get(f) != current_hash:  # FIX 2: Use .get() to avoid KeyError
                    logger.info(f"Content changed for PDF {f}: old hash {existing_file_info.get(f)}, new hash {current_hash}")
                    pdf_files_to_process.add(f)
                elif f in force_overwrite_files:
                    logger.info(f"Forced overwrite requested for PDF {f}")
                    pdf_files_to_process.add(f)
                    
            elif f.lower().endswith(('.xlsx', '.xls')):
                excel_files_info[f] = current_hash
                if f not in existing_file_info:
                    logger.info(f"New Excel file detected: {f}")
                    excel_files_to_process.add(f)
                elif existing_file_info.get(f) != current_hash:  # FIX 2: Use .get() to avoid KeyError
                    logger.info(f"Content changed for Excel {f}: old hash {existing_file_info.get(f)}, new hash {current_hash}")
                    excel_files_to_process.add(f)
                elif f in force_overwrite_files:
                    logger.info(f"Forced overwrite requested for Excel {f}")
                    excel_files_to_process.add(f)
        
        logger.info(f"Found {len(pdf_files_info)} PDF files, {len(pdf_files_to_process)} need processing")
        logger.info(f"Found {len(excel_files_info)} Excel files, {len(excel_files_to_process)} need processing")
        
        if not pdf_files_to_process and not excel_files_to_process:
            logger.info("No new or changed files to process")
            return
        
        # FIX 3: Add validation for files_to_process sets
        logger.info(f"PDF files to process: {list(pdf_files_to_process)}")
        logger.info(f"Excel files to process: {list(excel_files_to_process)}")
        
        # Process PDFs in parallel
        def process_pdf_task():
            try:
                transcript_embeddings, non_transcript_embeddings = process_all_pdfs(directory_path, model_name)
                
                # FIX 4: Add debugging and validation for metadata
                logger.info(f"Total transcript embeddings before filtering: {len(transcript_embeddings)}")
                logger.info(f"Total non-transcript embeddings before filtering: {len(non_transcript_embeddings)}")
                
                # Debug metadata structure
                if transcript_embeddings:
                    sample_doc, sample_emb = transcript_embeddings[0]
                    logger.info(f"Sample transcript metadata type: {type(sample_doc.metadata)}")
                    logger.info(f"Sample transcript metadata keys: {list(sample_doc.metadata.keys()) if isinstance(sample_doc.metadata, dict) else 'Not a dict'}")
                    logger.info(f"Sample transcript source_file: {sample_doc.metadata.get('source_file', 'NOT_FOUND')}")
                
                # FIX 5: Safer filtering with proper error handling
                filtered_transcript_embeddings = []
                for doc, emb in transcript_embeddings:
                    try:
                        # Ensure metadata is a dictionary
                        if not isinstance(doc.metadata, dict):
                            logger.warning(f"Document metadata is not a dict: {type(doc.metadata)}")
                            continue
                            
                        source_file = doc.metadata.get("source_file")
                        if source_file is None:
                            logger.warning(f"Document missing source_file in metadata: {doc.metadata}")
                            continue
                            
                        # Convert source_file to string if it's not already
                        source_file = str(source_file)
                        
                        # Extract just filename if it's a full path
                        source_filename = os.path.basename(source_file)
                        
                        if source_filename in pdf_files_to_process:
                            filtered_transcript_embeddings.append((doc, emb))
                            
                    except Exception as e:
                        logger.error(f"Error filtering transcript embedding: {e}")
                        continue
                
                # Same safe filtering for non-transcript embeddings
                filtered_non_transcript_embeddings = []
                for doc, emb in non_transcript_embeddings:
                    try:
                        if not isinstance(doc.metadata, dict):
                            logger.warning(f"Document metadata is not a dict: {type(doc.metadata)}")
                            continue
                            
                        source_file = doc.metadata.get("source_file")
                        if source_file is None:
                            logger.warning(f"Document missing source_file in metadata: {doc.metadata}")
                            continue
                            
                        source_file = str(source_file)
                        source_filename = os.path.basename(source_file)
                        
                        if source_filename in pdf_files_to_process:
                            filtered_non_transcript_embeddings.append((doc, emb))
                            
                    except Exception as e:
                        logger.error(f"Error filtering non-transcript embedding: {e}")
                        continue
                
                logger.info(f"Filtered to {len(filtered_transcript_embeddings)} PDF transcript embeddings and "
                           f"{len(filtered_non_transcript_embeddings)} PDF non-transcript embeddings for processing")
                
                return filtered_transcript_embeddings, filtered_non_transcript_embeddings
                
            except Exception as e:
                logger.error(f"Error in process_pdf_task: {e}")
                import traceback
                logger.error(f"PDF processing traceback: {traceback.format_exc()}")
                raise
        
        # FIX 6: Add similar safe processing for Excel files
        def process_excel_task():
            try:
                all_excel_embeddings = process_all_excel(directory_path, model_name)
                
                logger.info(f"Total Excel embeddings before filtering: {len(all_excel_embeddings)}")
                
                # Debug Excel metadata
                if all_excel_embeddings:
                    sample_doc, sample_emb = all_excel_embeddings[0]
                    logger.info(f"Sample Excel metadata type: {type(sample_doc.metadata)}")
                    logger.info(f"Sample Excel metadata: {sample_doc.metadata}")
                
                filtered_excel_embeddings = []
                for doc, emb in all_excel_embeddings:
                    try:
                        if not isinstance(doc.metadata, dict):
                            logger.warning(f"Excel document metadata is not a dict: {type(doc.metadata)}")
                            continue
                            
                        source_file = doc.metadata.get("source_file")
                        if source_file is None:
                            logger.warning(f"Excel document missing source_file in metadata: {doc.metadata}")
                            continue
                            
                        source_file = str(source_file)
                        source_filename = os.path.basename(source_file)
                        
                        if source_filename in excel_files_to_process:
                            filtered_excel_embeddings.append((doc, emb))
                            
                    except Exception as e:
                        logger.error(f"Error filtering Excel embedding: {e}")
                        continue
                
                logger.info(f"Filtered to {len(filtered_excel_embeddings)} Excel embeddings for processing")
                return filtered_excel_embeddings
                
            except Exception as e:
                logger.error(f"Error in process_excel_task: {e}")
                import traceback
                logger.error(f"Excel processing traceback: {traceback.format_exc()}")
                raise
        
        # Continue with rest of the function...
        # (The parallel execution and storage parts would go here)
        
    except Exception as e:
        logger.error(f"Error in process_and_store_embeddings: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

# DEBUGGING FUNCTION - Add this to help identify the exact issue
def debug_embeddings_structure(embeddings_list, list_name):
    """Debug function to analyze embeddings structure"""
    logger.info(f"=== Debugging {list_name} ===")
    logger.info(f"Total items: {len(embeddings_list)}")
    
    if embeddings_list:
        for i, item in enumerate(embeddings_list[:3]):  # Check first 3 items
            logger.info(f"Item {i}:")
            logger.info(f"  Type: {type(item)}")
            
            if isinstance(item, tuple) and len(item) == 2:
                doc, emb = item
                logger.info(f"  Document type: {type(doc)}")
                logger.info(f"  Embedding type: {type(emb)}")
                
                if hasattr(doc, 'metadata'):
                    logger.info(f"  Metadata type: {type(doc.metadata)}")
                    logger.info(f"  Metadata content: {doc.metadata}")
                    
                    # Check if any metadata values are dicts (potential issue)
                    if isinstance(doc.metadata, dict):
                        for key, value in doc.metadata.items():
                            if isinstance(value, dict):
                                logger.warning(f"  Found nested dict in metadata['{key}']: {value}")
                else:
                    logger.warning(f"  Document has no metadata attribute")
            else:
                logger.warning(f"  Item is not a (doc, embedding) tuple: {item}")

# ADDITIONAL HELPER FUNCTION
def clean_metadata_for_hashing(metadata):
    """Clean metadata to avoid unhashable type issues"""
    if not isinstance(metadata, dict):
        return {"error": "metadata_not_dict"}
    
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            # Convert nested dicts to strings
            cleaned[key] = str(value)
        elif isinstance(value, list):
            # Convert lists to strings
            cleaned[key] = str(value)
        elif value is None:
            cleaned[key] = "None"
        else:
            cleaned[key] = str(value)
    
    return cleaned