"""
Logging Utility

This module provides centralized logging configuration for the DataTobiz
brand monitoring system with proper formatting and log management.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class BrandMonitoringLogger:
    """Centralized logger configuration for the brand monitoring system."""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup_logging(
        cls,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_colors: bool = True
    ):
        """
        Setup centralized logging configuration.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
            enable_console: Whether to enable console logging
            enable_colors: Whether to enable colored console output
        """
        if cls._initialized:
            return
        
        # Create logs directory if it doesn't exist
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if enable_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                console_formatter = ColoredFormatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # File gets all levels
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for the specified name.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]

def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return BrandMonitoringLogger.get_logger(name)

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = "logs/brand_monitoring.log",
    **kwargs
):
    """
    Convenience function to setup logging.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
        **kwargs: Additional arguments for BrandMonitoringLogger.setup_logging
    """
    BrandMonitoringLogger.setup_logging(
        log_level=log_level,
        log_file=log_file,
        **kwargs
    )

# Initialize default logging configuration
def initialize_default_logging():
    """Initialize default logging configuration for the project."""
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        log_level = settings.workflow.log_level
    except:
        log_level = "INFO"
    
    setup_logging(
        log_level=log_level,
        log_file="logs/brand_monitoring.log",
        enable_console=True,
        enable_colors=True
    )

# Auto-initialize logging when module is imported
try:
    initialize_default_logging()
except:
    # Fallback to basic logging if settings aren't available
    setup_logging(log_level="INFO", log_file=None)