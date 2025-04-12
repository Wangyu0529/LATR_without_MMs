"""
Custom logger implementation inspired by mmcv
"""
import os
import sys
import logging
from logging import FileHandler, StreamHandler
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union, IO

LogLevel = Union[int, str]
class MMLogger:
    """MM-style logger with multiple handlers"""
    
    _INSTANCE = None
    
    def __init__(self,
                 name: str = 'main',
                 log_file: Optional[str] = None,
                 log_level: LogLevel = 'INFO',
                 file_mode: str = 'a',
                 max_bytes: int = 50 * 1024 * 1024,  # 50MB
                 backup_count: int = 5,
                 format_str: str = None,
                 datefmt: str = None):
        """
        Args:
            name: Logger name
            log_file: Path to log file
            log_level: Logging level (int or str)
            file_mode: File write mode
            max_bytes: Max file size before rotation
            backup_count: Number of backup files to keep
            format_str: Custom log format
            datefmt: Custom datetime format
        """
        self.logger = logging.getLogger(name)
        self._setup_logger(log_file, log_level, file_mode, 
                          max_bytes, backup_count,
                          format_str, datefmt)
    
    def _setup_logger(self,
                     log_file: Optional[str],
                     log_level: LogLevel,
                     file_mode: str,
                     max_bytes: int,
                     backup_count: int,
                     format_str: Optional[str],
                     datefmt: Optional[str]):
        # Clear existing handlers
        self.logger.handlers = []
        
        # Set log level
        self.logger.setLevel(log_level)
        
        # Create formatter
        format_str = format_str or (
            '[%(asctime)s] %(levelname)s - %(name)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
        formatter = logging.Formatter(format_str, datefmt)
        
        # Console handler with color
        console = ColorHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = self._get_file_handler(
                log_file, max_bytes, backup_count, file_mode, formatter
            )
            self.logger.addHandler(file_handler)
            
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _get_file_handler(self,
                         log_file: str,
                         max_bytes: int,
                         backup_count: int,
                         file_mode: str,
                         formatter: logging.Formatter):
        if max_bytes > 0:
            handler = RotatingFileHandler(
                log_file, mode=file_mode,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            handler = TimedRotatingFileHandler(
                log_file, when='D',
                backupCount=backup_count,
                encoding='utf-8'
            )
        handler.setFormatter(formatter)
        return handler
    
    @classmethod
    def get_logger(cls, name: str, **kwargs) -> logging.Logger:
        """Get logger instance (singleton pattern)"""
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(name, **kwargs)
        return cls._INSTANCE.logger
    
    def set_level(self, level: LogLevel):
        """Set global log level"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

class ColorHandler(StreamHandler):
    """Colorized console output"""
    
    COLOR_MAP = {
        'DEBUG': '\033[37m',     # White
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
    }
    RESET_SEQ = '\033[0m'
    
    def emit(self, record):
        try:
            msg = self.format(record)
            color = self.COLOR_MAP.get(record.levelname, '')
            msg = f"{color}{msg}{self.RESET_SEQ}"
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Shortcut functions
def init_logger(name: str = 'main', **kwargs) -> logging.Logger:
    """Initialize and get logger instance"""
    return MMLogger.get_logger(name, **kwargs)

def get_logger(name: str = 'main') -> logging.Logger:
    """Get existing logger instance"""
    return logging.getLogger(name)

def set_log_level(level: LogLevel):
    """Set global log level"""
    logger = get_logger()
    logger.setLevel(level)