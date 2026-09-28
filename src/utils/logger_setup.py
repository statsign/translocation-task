import logging
import logging.config
from pathlib import Path
from datetime import datetime
import sys
from src.utils.path_utils import get_project_root

PROJECT_ROOT = get_project_root()
LOGS_DIR = PROJECT_ROOT / "logs"

def setup_logging(experiment_name: str = None):
    """
    Project logging settings

    """

    if experiment_name:
        log_dir = LOGS_DIR / experiment_name
    else:
        log_dir = LOGS_DIR / datetime.now().strftime("%Y-%m-%d")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    main_log = log_dir / "app.log"
    error_log = log_dir / "errors.log"
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'console': {
                'format': '%(message)s'
                },
        },
        'handlers': {
            'file_main': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'filename': str(main_log),
                'maxBytes': 10_000_000,  # 10 MB
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'file_errors': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'standard',
                'filename': str(error_log),
                'maxBytes': 5_000_000,
                'backupCount': 5,
                'encoding': 'utf-8'
            },
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'console',
                'stream': sys.stdout  
            },
        },
        'root': {
            'level': 'INFO',
            'handlers': ['file_main', 'file_errors', 'console']
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Main log: {main_log}")
    
    return log_dir