import logging, traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        log_file = f"logs/{name}.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, e: Exception):
        msn = str({
            "success": False,
            "type": type(e).__name__,   # exception name
            "message": str(e),          # message error
            "traceback": traceback.format_exc()  # full traceback
        })
        self.logger.error(msn)
    
    def debug(self, message: str):
        self.logger.debug(message)