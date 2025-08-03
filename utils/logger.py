"""
Centralized logging system for vAlgo Trading System
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

def _get_project_root() -> Path:
    """Find project root directory"""
    current_path = Path(__file__).parent
    
    # Look for indicators of project root
    for _ in range(5):  # Check up to 5 levels up
        if any((current_path / indicator).exists() for indicator in 
               ['README.md', '.git', 'main_backtest.py', 'requirements.txt']):
            return current_path
        current_path = current_path.parent
        if current_path == current_path.parent:  # Reached filesystem root
            break
    
    # Fallback to current directory
    return Path.cwd()

def _get_env_config() -> Dict[str, str]:
    """Get environment configuration safely"""
    try:
        # Try to import env_loader if available
        from utils.env_loader import get_all_config
        config = get_all_config()
        if config:
            return {
                'log_file': str(config.get('log_file', 'logs/vAlgo.log')),
                'log_level': str(config.get('log_level', 'INFO')),
                'log_format': str(config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            }
        else:
            return {
                'log_file': 'logs/vAlgo.log',
                'log_level': 'INFO',
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
    except ImportError:
        # Fallback to direct environment variables
        return {
            'log_file': os.getenv('LOG_FILE', 'logs/vAlgo.log') or 'logs/vAlgo.log',
            'log_level': os.getenv('LOG_LEVEL', 'INFO') or 'INFO',
            'log_format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s') or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }

def setup_logger(name: str = "vAlgo", log_file: Optional[str] = None, log_level: Optional[str] = None, 
                daily_rotation: bool = True) -> logging.Logger:
    """
    Set up a logger with consistent formatting and daily rotation
    
    Args:
        name: Logger name
        log_file: Log file path (default from env or logs/vAlgo.log)
        log_level: Log level (default from env or INFO)
        daily_rotation: Enable daily log rotation (default True)
    
    Returns:
        Configured logger instance
    """
    
    try:
        # Get configuration
        env_config = _get_env_config()
        
        if log_file is None:
            log_file = env_config['log_file']
        
        if log_level is None:
            log_level = env_config['log_level']
        
        # Make log file path absolute relative to project root
        if not os.path.isabs(log_file):
            project_root = _get_project_root()
            log_file = str(project_root / log_file)
        
        # Apply daily rotation if enabled
        if daily_rotation:
            # Extract directory and base filename
            log_path = Path(log_file)
            log_dir = log_path.parent
            base_name = log_path.stem
            extension = log_path.suffix or '.log'
            
            # Create daily log filename: vAlgoLog_YYYYMMDD.log
            today = datetime.now().strftime('%Y%m%d')
            log_file = str(log_dir / f"{base_name}_{today}{extension}")
        
        # Ensure log directory exists
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # Fallback to current directory if can't create log dir
            log_file = f"{name}.log"
            print(f"Warning: Could not create log directory, using {log_file}: {e}")
        
        # Create logger
        logger = logging.getLogger(name)
        
        # Validate log level
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            level = logging.INFO
            print(f"Warning: Invalid log level '{log_level}', using INFO")
        
        logger.setLevel(level)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatter
        log_format = env_config['log_format']
        formatter = logging.Formatter(log_format)
        
        # File handler
        try:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            if daily_rotation:
                print(f"Daily rotation enabled: logging to {log_file}")
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create file handler: {e}")
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    except Exception as e:
        # Emergency fallback - basic logger
        print(f"Error setting up logger: {e}")
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        return logger

def get_logger(name: str = "vAlgo") -> logging.Logger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name)

# Create default logger
default_logger = get_logger()

# Convenience functions
def log_info(message: str, logger_name: str = "vAlgo"):
    """Log info message"""
    get_logger(logger_name).info(message)

def log_warning(message: str, logger_name: str = "vAlgo"):
    """Log warning message"""
    get_logger(logger_name).warning(message)

def log_error(message: str, logger_name: str = "vAlgo"):
    """Log error message"""
    get_logger(logger_name).error(message)

def log_debug(message: str, logger_name: str = "vAlgo"):
    """Log debug message"""
    get_logger(logger_name).debug(message)

def log_trade(symbol: str, action: str, quantity: int, price: float, 
              logger_name: str = "vAlgo.trades"):
    """
    Log trading activity
    
    Args:
        symbol: Trading symbol
        action: BUY/SELL
        quantity: Number of shares/contracts
        price: Execution price
        logger_name: Logger name for trades
    """
    trade_logger = get_logger(logger_name)
    trade_logger.info(f"TRADE: {action} {quantity} {symbol} @ {price}")

def log_performance(metric: str, value: float, logger_name: str = "vAlgo.performance"):
    """
    Log performance metrics
    
    Args:
        metric: Metric name (e.g., 'pnl', 'sharpe_ratio')
        value: Metric value
        logger_name: Logger name for performance
    """
    perf_logger = get_logger(logger_name)
    perf_logger.info(f"PERFORMANCE: {metric} = {value}")

def log_strategy_signal(symbol: str, signal: str, conditions: dict, 
                       logger_name: str = "vAlgo.signals"):
    """
    Log strategy signals
    
    Args:
        symbol: Trading symbol
        signal: Signal type (ENTRY/EXIT)
        conditions: Conditions that triggered the signal
        logger_name: Logger name for signals
    """
    signal_logger = get_logger(logger_name)
    conditions_str = ", ".join([f"{k}={v}" for k, v in conditions.items()])
    signal_logger.info(f"SIGNAL: {symbol} {signal} [{conditions_str}]")