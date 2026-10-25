import os
import logging
from logging.handlers import RotatingFileHandler
import datetime
import functools
import json
import time
import psutil

def setup_logging(log_dir='logs'):
    """
    Set up logging configuration with different log files for various components.
    
    Args:
        log_dir: Directory to store log files (default: 'logs')
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Common log format
    log_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create different loggers for each component
    loggers = {
        'satellite': {
            'file': 'satellite_processing.log',
            'level': logging.INFO,
        },
        'preprocessing': {
            'file': 'data_preprocessing.log',
            'level': logging.INFO,
        },
        'training': {
            'file': 'model_training.log',
            'level': logging.INFO,
        },
        'evaluation': {
            'file': 'model_evaluation.log',
            'level': logging.INFO,
        }
    }
    
    # Configure each logger
    configured_loggers = {}
    for name, config in loggers.items():
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(config['level'])
        
        # Create rotating file handler (10MB per file, max 5 files)
        handler = RotatingFileHandler(
            os.path.join(log_dir, config['file']),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(log_format)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Store configured logger
        configured_loggers[name] = logger
    
    return configured_loggers

def log_execution_time(func):
    """
    Decorator to log function execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Get the logger from the class instance if it exists
            logger = args[0].logger if hasattr(args[0], 'logger') else logging.getLogger()
            
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            logger = args[0].logger if hasattr(args[0], 'logger') else logging.getLogger()
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def log_error(func):
    """
    Decorator to log function errors with detailed information.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the logger from the class instance if it exists
            logger = args[0].logger if hasattr(args[0], 'logger') else logging.getLogger()
            
            error_info = {
                'function': func.__name__,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.datetime.now().isoformat(),
                'args': str(args[1:]),  # Exclude self
                'kwargs': str(kwargs)
            }
            
            logger.error(f"Error details: {json.dumps(error_info, indent=2)}")
            raise
    return wrapper

def log_memory_usage(func):
    """
    Decorator to log memory usage before and after function execution.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = end_memory - start_memory
            
            # Get the logger from the class instance if it exists
            logger = args[0].logger if hasattr(args[0], 'logger') else logging.getLogger()
            
            logger.info(
                f"{func.__name__} memory usage: "
                f"Start: {start_memory:.2f}MB, "
                f"End: {end_memory:.2f}MB, "
                f"Difference: {memory_diff:+.2f}MB"
            )
            return result
        except Exception as e:
            logger = args[0].logger if hasattr(args[0], 'logger') else logging.getLogger()
            logger.error(f"Error in {func.__name__} while monitoring memory: {str(e)}")
            raise
    return wrapper