from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import execute_final_analysis, data_formatter, extract_data_requirements, extract_topics, run_orchestration
from text_processor import format_documents, parse_query
from coda_analyzer import generate_coda_prompt
from logger_setup import get_logger
from langchain_community.vectorstores import HanaDB
from image_processor import process_images
import os

logger = get_logger()

def generate_summary_template(context: str, query: str, analysis_type: str = "general", transcript_context: str = None) -> str:
    """Create a concise summary template based on analysis type."""
    templates = {
        "financial": "Financial summary for '{query}': Use {context} for metrics; reject transcript data. Use {transcript_context} for quotes only. Note if data insufficient.",
        "trend": "Outlook for '{query}': Use {context} for trends; reject transcript data. Use {transcript_context} for quotes with speaker details. State if unclear.",
        "general": "Key points for '{query}': Use {context} only; no quotes unless specified.",
        "topics": "Top 5 topics for '{query}': Use {transcript_context} (QnA only) to extract top 5 topics with percentage weightage (summing to 100%) and keywords in HTML format as an ordered list. Exclude financial metrics, summaries, or external data.",
        "quotes": "Top 5 management quotes for '{query}': Use {transcript_context} (transcript only) to extract up to 5 quotes from company management (e.g., CEO, CFO) with speaker details in HTML format as an ordered list. Exclude financial metrics or external data.",
        "callouts": "Major callouts for '{query}': Use {context} for bullet-point drivers. Note if unclear.",
        "consensus": "Consensus for '{query}': Use {context} for drivers. Note limitations if insufficient.",
        "Stock": "Stock analysis for '{query}': Use {context} for price and volume trends; include technical indicators. Use {transcript_context} for management commentary only. Note if data insufficient."
    }
    template = templates.get(analysis_type.lower(), templates["general"])
    return template.format(
        query=query, 
        context=context, 
        transcript_context=transcript_context or "No transcript context"
    )

def fetch_context(retriever, query: str, k: int = 30) -> str:
    """Retrieve documents with minimal memory usage."""
    try:
        docs = retriever.invoke(query)
        context = format_documents(docs) if docs else "No documents found."
        logger.debug(f"Fetched context for query '{query[:50]}...': {context[:100]}... (length: {len(context)})")
        return context
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return f"Error: {str(e)}"

def process_analysis_type(
    analysis_type: str, user_query: str, transcript_store, non_transcript_store, transcript_context: str
) -> Tuple[str, str, str]:
    """Process an individual analysis type."""
    try:
        logger.debug(f"Processing analysis type '{analysis_type}' with user_query: '{user_query[:50]}...' and transcript_context: {transcript_context[:100]}... (length: {len(transcript_context)})")
        if analysis_type == "topics":
            # Validate transcript_context for topics
            if not transcript_context or transcript_context.startswith("Error") or len(transcript_context.strip()) < 50:
                logger.warning(f"Invalid or insufficient transcript_context for topics: {transcript_context[:100]}...")
                task = "<ol><li>Topic: None, Weight: 100%, Keywords: none</li></ol>"
            else:
                task = extract_topics(transcript_context)
                # Validate task output
                if not task.startswith("<ol>") or not task.endswith("</ol>"):
                    logger.warning(f"Invalid topics output: {task[:100]}...")
                    task = "<ol><li>Topic: None, Weight: 100%, Keywords: none</li></ol>"
            context = transcript_context
            logger.debug(f"Topics task result: {task[:100]}...")
        elif analysis_type == "quotes":
            # Create prompt for quotes extraction, using the raw user query
            prompt = f"""<prompt> You are tasked with extracting up to 5 quotes from company management (e.g., CEO, CFO, executives) in the provided transcript text, specifically relevant to the user's request: '{user_query}'. Each quote must include the speaker's name and role (if available). Output the result STRICTLY in HTML format as an ordered list (<ol>), with each list item (<li>) formatted exactly as: "<quote>" - <speaker> (<role>).

            Strict Requirements:
            - Analyze ONLY the provided transcript text. Do NOT use external information or infer beyond the text.
            - Extract quotes ONLY from company management (e.g., CEO, CFO, other executives), ignoring analysts or others.
            - Quotes MUST be relevant to the user's request '{user_query}'. If the request specifies a focus (e.g., 'revenue growth', 'cloud services'), prioritize quotes addressing that focus.
            - If the user request is vague or general (e.g., 'management quotes'), select quotes related to company strategy, operations, or outlook.
            - EXCLUDE any financial metrics (e.g., revenue, profit, stock price, earnings), summaries, introductions, notifications, or caveats.
            - If fewer than 5 relevant quotes are available, list only those found.
            - If no relevant management quotes are identified, return: <ol><li>Quote: None, Speaker: None (None)</li></ol>
            - Ensure quotes are concise (1-2 sentences) and directly address the user's request or general strategy/operations if unspecified.
            - Output ONLY the HTML ordered list (<ol>...</ol>) with no additional text, comments, or explanations.
            - Ensure valid HTML syntax.
            - DBS Bank and Deutsche Banking Services are distinct entities and should not be regarded as the same. 

            Example Output for user request 'management quotes for cloud services':
            <ol>
                <li>"We are expanding our cloud services to new markets." - John Doe (CEO)</li>
                <li>"Our cloud infrastructure is driving efficiency." - Jane Smith (CFO)</li>
                <li>"Cloud innovation is a core focus." - Alice Brown (COO)</li>
            </ol>

            Example Output for general user request 'management quotes':
            <ol>
                <li>"We are investing in AI to drive growth." - John Doe (CEO)</li>
                <li>"Sustainability is key to our operations." - Jane Smith (CFO)</li>
                <li>"Customer engagement is our priority." - Alice Brown (COO)</li>
            </ol>

            </prompt> <text> {transcript_context} </text>"""
            if not transcript_context or transcript_context.startswith("Error"):
                logger.warning("No valid transcript_context for quotes")
                task = "<ol><li>Quote: None, Speaker: None (None)</li></ol>"
            else:
                task = run_orchestration(prompt, error_context="quote extraction")
            context = transcript_context
            logger.debug(f"Quotes task result: {task[:100]}...")
        else:
            # Other analysis types use non-transcript store
            store = non_transcript_store
            retriever = store.as_retriever(search_kwargs={"k": 30, "score_threshold": 0.6}, search_type="similarity_score_threshold") 
            context = fetch_context(retriever, user_query)
            task = generate_summary_template(context, user_query, analysis_type, transcript_context)
        
        return analysis_type, task, context
    except Exception as e:
        logger.error(f"Processing {analysis_type} failed: {str(e)}")
        return analysis_type, f"Error: {str(e)}", f"Error: {str(e)}"

def process_excel_only_query(query: str, excel_non_transcript_store: HanaDB) -> str:
    """Process Excel query efficiently."""
    try:
        retriever = excel_non_transcript_store.as_retriever(search_kwargs={"k": 30, "score_threshold": 0.8}, search_type="similarity_score_threshold") 
        context = fetch_context(retriever, query)
        return context if context and not context.startswith("Error") else "No Excel data"
    except Exception as e:
        logger.error(f"Excel query failed: {str(e)}")
        return "No Excel data"

def process_query(
    query: str,
    transcript_store: Optional[HanaDB] = None,
    non_transcript_store: Optional[HanaDB] = None,
    excel_non_transcript_store: Optional[HanaDB] = None
) -> str:
    """Handle query processing with parallel retrieval, image processing, and Excel query execution."""
    if not query:
        logger.error("No query provided")
        return "Error: No query provided."
    
    if not all([transcript_store, non_transcript_store, excel_non_transcript_store]):
        logger.error("Missing vector stores")
        return "Error: All vector stores required."

    try:
        # Parse query and generate analysis types
        clean_query, analysis_types = parse_query(query)
        logger.debug(f"Processing query: {clean_query}, analysis types: {analysis_types}")

        # Fetch transcript context (required for quotes and topics)
        transcript_context = fetch_context(transcript_store.as_retriever(search_kwargs={"k": 30, "score_threshold": 0.6}, search_type="similarity_score_threshold"), clean_query)

        # Special case: Only quotes requested
        if analysis_types == ["quotes"]:
            logger.debug("Processing quotes-only request")
            _, task, _ = process_analysis_type(
                "quotes", query, transcript_store, non_transcript_store, transcript_context
            )
            # Format the quotes result directly as HTML
            formatted_result = f"<h2>Management Quotes</h2>{task}"
            return formatted_result

        # Special case: Only topics requested
        if analysis_types == ["topics"]:
            logger.debug("Processing topics-only request")
            _, task, _ = process_analysis_type(
                "topics", query, transcript_store, non_transcript_store, transcript_context
            )
            # Format the topics result directly as HTML
            formatted_result = f"<h2>Key Topics</h2>{task}"
            return formatted_result

        # General case: Process all analysis types
        coda_prompt = generate_coda_prompt(query)
        data_requirements = extract_data_requirements(coda_prompt)
        refined_query = f"{clean_query} {data_requirements}"

        # Check for stock-related query
        stock_keywords = ["stock", "share price", "stock analysis", "stock insights"]
        is_stock_query = any(keyword.lower() in query.lower() for keyword in stock_keywords)

        # Parallel processing of image processing, analysis types, and Excel query
        image_results = []
        tasks = {}
        contexts = {}
        excel_result = "No Excel data"

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}

            # Submit image processing task if stock query
            if is_stock_query:
                base_path = os.getenv('LOCALPATH', '')
                folder_path = os.path.join(base_path, "Images")
                user_prompt = query
                if not os.path.exists(folder_path):
                    logger.error(f"Image folder does not exist: {folder_path}")
                    image_results = [{"image_path": "N/A", "analysis": f"Error: Image folder {folder_path} does not exist"}]
                else:
                    futures[executor.submit(process_images, folder_path, user_prompt)] = "image"

            # Submit analysis type tasks
            for atype in analysis_types:
                futures[executor.submit(
                    process_analysis_type, 
                    atype, 
                    query,  # Use raw query for quotes and topics
                    transcript_store, 
                    non_transcript_store, 
                    transcript_context
                )] = f"analysis_{atype}"

            # Submit Excel query task
            futures[executor.submit(process_excel_only_query, refined_query, excel_non_transcript_store)] = "excel"

            # Collect results as they complete
            for future in as_completed(futures):
                task_type = futures[future]
                try:
                    if task_type == "image":
                        image_results = future.result()
                        logger.info(f"Processed {len(image_results)} images for stock query")
                    elif task_type.startswith("analysis_"):
                        atype, task, context = future.result()
                        tasks[atype] = task
                        contexts[atype] = context
                    elif task_type == "excel":
                        excel_result = future.result()
                except Exception as e:
                    if task_type == "image":
                        logger.error(f"Image processing failed: {str(e)}")
                        image_results = [{"image_path": "N/A", "analysis": f"Error processing images: {str(e)}"}]
                    elif task_type.startswith("analysis_"):
                        logger.error(f"Processing {task_type} failed: {str(e)}")
                        atype = task_type.split("_")[1]
                        tasks[atype] = f"Error: {str(e)}"
                        contexts[atype] = f"Error: {str(e)}"
                    elif task_type == "excel":
                        logger.error(f"Excel query failed: {str(e)}")
                        excel_result = f"Error: {str(e)}"

        # Create integrated prompt
        integrated_prompt = f"""
        Tasks: {', '.join(tasks.values())}
        CODA Data: {data_requirements}
        Contexts: {str(contexts)}
        Transcript Context: {transcript_context}
        Query: {query}
        Requirements:
        - Do not hallucinate. If there is no data for Specified Bank, STRICTLY state 'Data not Available'.
        - Integrate tasks with CODA analysis
        - Use bullets for financials, quotes, callouts. Support with Driver Details.
        - Quotes from Transcript Context only, STRICTLY with Speaker Details
        - Financials, callouts, consensus from Contexts only
        - State 'Data not available' if no non-transcript data
        - Format cohesively with confidence metrics
        - DBS Bank and Deutsche Banking Services are distinct entities and should not be regarded as the same.
        """

        # Execute and format final response
        final_analysis = execute_final_analysis(integrated_prompt)
        response = data_formatter(final_analysis, excel_result, image_results if is_stock_query else None)
        return response or "No response due to insufficient data."

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return f"Error: {str(e)}"