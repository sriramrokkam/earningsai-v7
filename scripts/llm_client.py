import os
import json
import requests
from dotenv import load_dotenv
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.orchestration.models.azure_content_filter import AzureContentFilter 
from logger_setup import get_logger
from typing import Optional, List, Dict
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details,extract_aicore_credentials
#from db_connection import ORCHESTRATION_SERVICE_URL


# Load environment variables
#load_dotenv()

# Setup logger
logger = get_logger()

# Global variables for AIC credentials and orchestration service

#----------------------------LOAD CF VCAP_SERVICES Variables -----------------------------
# Log start of credential retrieval
logger.info ("====> llm_client.py -> GET HANA AND AIC CREDENTIALS <====")

# Load VCAP_SERVICES from environment
vcap_services = os.environ.get("VCAP_SERVICES")

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
    
logger.info("AIC Destination Details fetched successfully")
# Extract AIC Details
AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
#**** Construct the Orchestration url client
from gen_ai_hub.proxy import GenAIHubProxyClient
logger.info("Orchestration URL : initialization")
proxy_client = GenAIHubProxyClient(
                                base_url = AIC_CREDENTIALS['aic_base_url'],
                                auth_url = AIC_CREDENTIALS['aic_auth_url'],
                                client_id = AIC_CREDENTIALS['clientid'],
                                client_secret = AIC_CREDENTIALS['clientsecret'],
                                resource_group = AIC_CREDENTIALS['resource_group']
                                )
ORCHESTRATION_SERVICE_URL = AIC_CREDENTIALS['ORCHESTRATION_SERVICE_URL']
ORCHESTRATION_SERVICE = OrchestrationService(api_url=ORCHESTRATION_SERVICE_URL, proxy_client=proxy_client)  
logger.info(f" AIC Credentials: {AIC_CREDENTIALS}")


# Define Azure Content Filter thresholds
CONTENT_FILTER = AzureContentFilter(hate=6, sexual=4, self_harm=0, violence=4)

# Model configuration
MODEL_CONFIG = LLM(
    name="anthropic--claude-3.5-sonnet",
    parameters={
        'temperature': 0.5,
        'max_tokens': 200000,
        'top_p': 0.9
    }
)

def run_orchestration(prompt, error_context="orchestration"):
    """Run orchestration service with content filtering."""
    try:
        if ORCHESTRATION_SERVICE is None:
            raise ValueError("OrchestrationService not initialized")
        
        template = Template(messages=[UserMessage("{{ ?extraction_prompt }}")])
        config = OrchestrationConfig(template=template, llm=MODEL_CONFIG)
        config.input_filter = CONTENT_FILTER
        config.output_filter = CONTENT_FILTER
        
        logger.debug(f"Running {error_context} with prompt: {prompt[:100]}...")
        response = ORCHESTRATION_SERVICE.run(
            config=config,
            template_values=[TemplateValue("extraction_prompt", prompt)]
        )
        
        result = response.orchestration_result.choices[0].message.content
        logger.debug(f"Completed {error_context} with result: {result[:100]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error in {error_context}: {str(e)}", exc_info=True)
        raise Exception(f"Error in {error_context}: {str(e)}")

def execute_coda_analysis(coda_prompt):
    """Execute CODA analysis."""
    return run_orchestration(coda_prompt, error_context="CODA analysis")

def extract_data_requirements(coda_result):
    """Extract data requirements from CODA analysis."""
    prompt = f"<prompt> extract the data requirements portion of the following text </prompt> <text> {coda_result} </text>"
    return run_orchestration(prompt, error_context="data requirements extraction")

def execute_final_analysis(final_prompt):
    """Execute final analysis."""
    return run_orchestration(final_prompt, error_context="final analysis")

def extract_analysis_steps(coda_result):
    """Extract analysis steps from CODA analysis."""
    prompt = f"<prompt> extract STRICTLY the required analysis and data requirements portion of the following text </prompt> <text> {coda_result} </text>"
    return run_orchestration(prompt, error_context="analysis steps extraction")

def extract_topics(transcript_text: str) -> str:
    """Extract top 5 topics with weights and keywords in HTML format."""
    prompt = f"""<prompt> You are tasked with extracting the top 5 topics discussed in the provided transcript text (e.g., earnings call QnA). For each topic, provide a percentage weightage (summing exactly to 100% across all topics) and 3-5 associated keywords (comma-separated). Output the result STRICTLY in HTML format as an ordered list (<ol>), with each list item (<li>) formatted exactly as: "Topic: <topic_name>, Weight: <percentage>%, Keywords: <keyword1>,<keyword2>,<keyword3>,...".

    Strict Requirements:
    - Analyze ONLY the provided transcript text. Do NOT use external information or infer beyond the text.
    - Identify EXACTLY 5 topics unless fewer distinct topics are present, in which case list only those available and adjust percentages to sum to 100%.
    - If no topics can be identified (e.g., text is too short or unclear), return: <ol><li>Topic: None, Weight: 100%, Keywords: none</li></ol>
    - Topics should reflect key discussion themes (e.g., strategy, operations, innovation), NOT financial metrics (e.g., revenue, profit, stock price, earnings).
    - EXCLUDE summaries, introductions, notifications, caveats, or speaker names in topic descriptions.
    - Ensure percentages are integers and sum to 100%.
    - Keywords should be specific, relevant terms from the transcript (e.g., 'cloud computing', 'AI', 'sustainability').
    - Output ONLY the HTML ordered list (<ol>...</ol>) with no additional text, comments, or explanations.
    - Ensure valid HTML syntax.
    - If the transcript is noisy or fragmented, prioritize coherent themes based on frequency and context.

    Example Output:
    <ol>
        <li>Topic: Cloud Computing, Weight: 30%, Keywords: cloud, infrastructure, services, expansion</li>
        <li>Topic: AI Development, Weight: 25%, Keywords: artificial intelligence, machine learning, algorithms</li>
        <li>Topic: Customer Engagement, Weight: 20%, Keywords: user experience, feedback, retention</li>
        <li>Topic: Product Innovation, Weight: 15%, Keywords: new features, development, launches</li>
        <li>Topic: Sustainability, Weight: 10%, Keywords: eco-friendly, green tech, renewable</li>
    </ol>

    </prompt> <text> {transcript_text} </text>"""
    return run_orchestration(prompt, error_context="data topic extraction")

def data_formatter(final_result: str, excel_final_result: str, Image_Result: Optional[List[Dict[str, str]]] = None) -> str:
    """Format final response as HTML with executive summary, main content, stock analysis, and Excel data."""
    stock_section = ""
    if Image_Result:
        stock_section = "<h2>Stock Analysis</h2><div>"
        for result in Image_Result:
            analysis = result.get("analysis", "No stock analysis available")
            stock_section += f"<p>{analysis}</p>"
        stock_section += "</div>"

    prompt = f"""<prompt> Format the final response as HTML content with highlights, bold, and consistent styling. 
    Include up to four sections: 
    1. An Executive Summary at the top, written as a single, cohesive paragraph of full sentences, integrating key drivers and precise numbers. The Executive Summary must not contain bullet points, lists, or list-like structures; it should be a narrative flow. Ensure number formatting is consistent, using bn for Billion, mn for Million, YoY for Year over Year, and QoQ for Quarter over Quarter.
    2. The full main content from the input include Excel data as well STRICTLY NO SEPERATE EXCEL SECTION, formatted with any bullet points on new lines, preserving all original details, quotes, and structure.
    3. A Stock Analysis section if provided, containing only the analysis text formatted as paragraphs.
    Ensure number formatting is consistent, using bn for Billion, mn for Million, YoY for Year over Year, and QoQ for Quarter over Quarter. 
    Keep all quotes unchanged and do not summarize or alter the content beyond formatting. 
    Output only the formatted HTML content, STRICTLY excluding any introductory text such as 'Here is an HTML-formatted summary ', 'Here is the formatted content..' or 'Here is my analysis of..' or 'Here is an HTML-formatted..'. 
    Consolidate the results and exclude CODA Analysis. 
    </prompt> <text> {final_result} </text> <stock_section> {stock_section} </stock_section> <excel_section> {excel_final_result} </excel_section>"""   
    return run_orchestration(prompt, error_context="data formatting")