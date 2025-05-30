import os
import json
from dotenv import load_dotenv

# Table names for HANA DB
TABLE_NAMES = {
    "transcript": "GENAI_EARNINGS_Q125_EMBEDDED_TRANSCRIPTS",
    "non_transcript": "GENAI_EARNINGS_Q125_EMBEDDED_NON_TRANSCRIPTS",
    "excel_non_transcript": "GENAI_EARNINGS_Q125_EMBEDDED_EXCEL_NON_TRANSCRIPTS"
}

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-large"
HANA_DB_API ="3cb8ff87-b67f-4b68-8106-e297566641ef.hana.prod-ap11.hanacloud.ondemand.com"

# Bedrock model configuration
MODEL_ID = "anthropic--claude-3.5-sonnet"

# Known banks as JSON string with code-name pairs
KNOWN_BANKS_JSON = '''
{
    "JPMC": "JP Morgan",
    "MS": "Morgan Stanley",
    "GS": "Goldman Sachs",
    "C": "Citi",
    "BAC": "Bank of America",
    "BNP": "BNP Paribas",
    "DBK": "Deutsche Bank",
    "HSBC": "HSBC",
    "BBVA": "Banco Bilbao Vizcaya Argentaria",
    "BCS": "Barclays",
    "SAN": "Banco Santander",
    "UBSG": "UBS Group",
    "ING": "ING Bank",
    "SCB": "Standard Chartered",
    "DBS": "DBS"
}
'''

# Supported image extensions
IMAGE_EXTENSIONS = ["jpeg", "png"]

def get_known_banks() -> dict:
    """
    Load and parse the KNOWN_BANKS_JSON into a dictionary.
    
    Returns:
        dict: Dictionary with bank codes as keys and names as values.
    """
    try:
        return json.loads(KNOWN_BANKS_JSON)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse KNOWN_BANKS_JSON: {str(e)}")

def load_config():
    """Load environment variables from .env file."""
    load_dotenv()

def get_documents_dir_path():
    """Get default documents directory path."""
    return os.path.join(load_config()['local_path'], "Documents")