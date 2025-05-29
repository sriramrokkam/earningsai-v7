import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(log_dir='logs', log_file='earnings_analysis.log', level=logging.INFO):
    """
    Set up and configure a logger instance with rotating file handler.
    
    Args:
        log_dir (str): Directory to store log files
        log_file (str): Name of the log file
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Full path to log file
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger('earnings_analysis')
    logger.setLevel(level)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create rotating file handler (max 10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10*1024*1024, backupCount=5
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logger(log_dir = "logs")

def get_logger():
    """
    Get the global logger instance.
    
    Returns:
        logging.Logger: The configured logger instance
    """
    return logger
