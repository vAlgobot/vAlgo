"""
Selective Centralized Logging System for vAlgo VectorBT Backtesting
================================================================

Creates a fresh log file for every execution while keeping major component prints
visible to the user. Detailed information is logged to files for debugging.

Features:
- Fresh log files for each execution (auto-cleanup of old logs)
- Major component prints remain visible (bot performance, portfolio analysis)
- Detailed logs written to files in logs/ folder
- Multiple log levels for different components
- Performance metrics logging

Author: vAlgo Development Team
Created: July 29, 2025
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

class SelectiveLogger:
    """
    Selective logging system that maintains console prints for major components
    while logging detailed information to files.
    """
    
    def __init__(self, session_name: str = "vbt_backtesting"):
        """
        Initialize selective logger with fresh log files.
        
        Args:
            session_name: Name for this logging session
        """
        self.session_name = session_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create fresh log files for this execution
        self.main_log_file = LOGS_DIR / f"{session_name}_{self.timestamp}.log"
        self.performance_log_file = LOGS_DIR / f"{session_name}_performance_{self.timestamp}.log"
        self.debug_log_file = LOGS_DIR / f"{session_name}_debug_{self.timestamp}.log"
        
        # Clean up old log files (keep only last 5 executions)
        self._cleanup_old_logs()
        
        # Setup loggers
        self._setup_loggers()
        
        print(f"🔍 Selective Logging System initialized")
        print(f"   📝 Main log: {self.main_log_file.name}")
        print(f"   📊 Performance log: {self.performance_log_file.name}")
        print(f"   🐛 Debug log: {self.debug_log_file.name}")
    
    def _cleanup_old_logs(self):
        """Remove old log files, keeping only the 5 most recent."""
        try:
            # Find all log files for this session
            log_pattern = f"{self.session_name}_*.log"
            old_logs = list(LOGS_DIR.glob(log_pattern))
            
            # Sort by modification time (newest first)
            old_logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove files beyond the 5 most recent
            if len(old_logs) > 5:
                for old_log in old_logs[5:]:
                    try:
                        old_log.unlink()
                        print(f"🗑️ Cleaned up old log: {old_log.name}")
                    except Exception as e:
                        pass  # Ignore cleanup errors
                        
        except Exception as e:
            pass  # Ignore cleanup errors
    
    def _setup_loggers(self):
        """Setup different loggers for different purposes."""
        
        # Main application logger
        self.main_logger = logging.getLogger(f"{self.session_name}.main")
        self.main_logger.setLevel(logging.INFO)
        
        # Performance logger
        self.performance_logger = logging.getLogger(f"{self.session_name}.performance")
        self.performance_logger.setLevel(logging.INFO)
        
        # Debug logger
        self.debug_logger = logging.getLogger(f"{self.session_name}.debug")
        self.debug_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        for logger in [self.main_logger, self.performance_logger, self.debug_logger]:
            logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Main log file handler
        main_handler = logging.FileHandler(self.main_log_file)
        main_handler.setFormatter(detailed_formatter)
        self.main_logger.addHandler(main_handler)
        
        # Performance log file handler
        perf_handler = logging.FileHandler(self.performance_log_file)
        perf_handler.setFormatter(simple_formatter)
        self.performance_logger.addHandler(perf_handler)
        
        # Debug log file handler
        debug_handler = logging.FileHandler(self.debug_log_file)
        debug_handler.setFormatter(detailed_formatter)
        self.debug_logger.addHandler(debug_handler)
    
    def log_major_component(self, message: str, component: str = "SYSTEM"):
        """
        Log major component information (shown in console + logged to file).
        
        Args:
            message: Message to log
            component: Component name (SYSTEM, BOT_PERFORMANCE, PORTFOLIO_ANALYSIS, etc.)
        """
        # Show in console (as requested by user)
        print(f"🚀 {component}: {message}")
        
        # Also log to file
        self.main_logger.info(f"[{component}] {message}")
    
    def log_performance(self, metrics: Dict[str, Any], component: str = "PERFORMANCE"):
        """
        Log performance metrics (shown in console + logged to file).
        
        Args:
            metrics: Performance metrics dictionary
            component: Component name
        """
        # Show in console (as requested by user)
        print(f"📊 {component} METRICS:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Log to performance file
        self.performance_logger.info(f"[{component}] {json.dumps(metrics, indent=2)}")
    
    def log_portfolio_analysis(self, analysis: Dict[str, Any], strategy: str):
        """
        Log portfolio analysis (shown in console + logged to file).
        
        Args:
            analysis: Portfolio analysis results
            strategy: Strategy name
        """
        # Show in console (as requested by user)  
        print(f"💼 PORTFOLIO ANALYSIS - {strategy}:")
        for key, value in analysis.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"   {key}: {value}")
        
        # Log detailed analysis to file
        self.main_logger.info(f"[PORTFOLIO_ANALYSIS] Strategy: {strategy}")
        self.main_logger.info(f"[PORTFOLIO_ANALYSIS] {json.dumps(analysis, indent=2, default=str)}")
    
    def log_bot_performance(self, performance: Dict[str, Any]):
        """
        Log bot performance metrics (shown in console + logged to file).
        
        Args:
            performance: Bot performance metrics
        """
        # Show in console (as requested by user)
        print(f"🤖 BOT PERFORMANCE:")
        for key, value in performance.items():
            print(f"   {key}: {value}")
        
        # Log to performance file
        self.performance_logger.info(f"[BOT_PERFORMANCE] {json.dumps(performance, indent=2)}")
    
    def log_detailed(self, message: str, level: str = "INFO", component: str = "SYSTEM"):
        """
        Log detailed information (only to file, not console).
        
        Args:
            message: Detailed message
            level: Log level (INFO, DEBUG, WARNING, ERROR)
            component: Component name
        """
        # Only log to file (no console output)
        if level == "DEBUG":
            self.debug_logger.debug(f"[{component}] {message}")
        elif level == "WARNING":
            self.main_logger.warning(f"[{component}] {message}")
        elif level == "ERROR":
            self.main_logger.error(f"[{component}] {message}")
        else:
            self.main_logger.info(f"[{component}] {message}")
    
    def log_trade_details(self, trade_data: Dict[str, Any]):
        """
        Log detailed trade information (only to file).
        
        Args:
            trade_data: Trade details dictionary
        """
        self.debug_logger.debug(f"[TRADE_DETAILS] {json.dumps(trade_data, indent=2, default=str)}")
    
    def log_signal_details(self, signal_data: Dict[str, Any]):
        """
        Log detailed signal information (only to file).
        
        Args:
            signal_data: Signal details dictionary
        """
        self.debug_logger.debug(f"[SIGNAL_DETAILS] {json.dumps(signal_data, indent=2, default=str)}")
    
    def log_execution_summary(self, summary: Dict[str, Any]):
        """
        Log execution summary (shown in console + logged to file).
        
        Args:
            summary: Execution summary dictionary
        """
        # Show in console
        print(f"📋 EXECUTION SUMMARY:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Log to main file
        self.main_logger.info(f"[EXECUTION_SUMMARY] {json.dumps(summary, indent=2, default=str)}")


# Global logger instance
_global_logger: Optional[SelectiveLogger] = None

def get_selective_logger(session_name: str = "vbt_backtesting") -> SelectiveLogger:
    """
    Get global selective logger instance.
    
    Args:
        session_name: Session name for logging
        
    Returns:
        SelectiveLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = SelectiveLogger(session_name)
    return _global_logger

def reset_selective_logger():
    """Reset global logger for new execution."""
    global _global_logger
    _global_logger = None


if __name__ == "__main__":
    # Test selective logger
    logger = SelectiveLogger("test_session")
    
    # Test major component logging (shows in console + logs to file)
    logger.log_major_component("System initialization complete", "SYSTEM")
    logger.log_bot_performance({"signals_per_second": 1000, "total_trades": 25})
    logger.log_portfolio_analysis({"total_return": 15.5, "sharpe_ratio": 1.2}, "TrendFollow")
    
    # Test detailed logging (only to files)
    logger.log_detailed("Detailed calculation step 1", "DEBUG", "SIGNAL_GENERATOR")
    logger.log_detailed("Warning: Missing indicator data", "WARNING", "INDICATOR_ENGINE")
    
    # Test trade and signal details (only to files)
    logger.log_trade_details({"entry_price": 23000, "exit_price": 23100, "pnl": 7500})
    logger.log_signal_details({"entry_condition": True, "rsi": 25.5, "sma_cross": True})
    
    print("✅ Selective logger test completed. Check logs/ directory for output files.")