"""
Processor Base Class for Ultimate Efficiency Engine Components
=============================================================

Base class providing shared functionality for all Ultimate Efficiency Engine processors.
Implements dependency injection pattern and common utilities.

Author: vAlgo Development Team
Created: July 29, 2025
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.json_config_loader import JSONConfigLoader, ConfigurationError

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    print("❌ VectorBT not available for processor components")

# Selective logger - direct import  
from utils.selective_logger import get_selective_logger


class ProcessorBase:
    """
    Base class for all Ultimate Efficiency Engine processors.
    
    Provides shared functionality including:
    - Configuration access
    - Data loader access  
    - Logging interface
    - Common utilities
    - Error handling patterns
    """
    
    def __init__(
        self, 
        config_loader: JSONConfigLoader,
        data_loader: Any,  # EfficientDataLoader
        logger: Any  # SelectiveLogger or fallback
    ):
        """
        Initialize processor with dependency injection.
        
        Args:
            config_loader: Configuration loader instance
            data_loader: EfficientDataLoader instance
            logger: Selective logger instance
        """
        self.config_loader = config_loader
        self.data_loader = data_loader  
        self.logger = logger
        
        # Quick access to configurations
        self.main_config = config_loader.main_config
        self.strategies_config = config_loader.strategies_config
        
        # Performance tracking
        self.performance_stats = {}
        
        # Options are enabled by default - this is an options trading system
        
        self.processor_name = self.__class__.__name__
        
    def log_major_component(self, message: str, component: str = None) -> None:
        """Log major component activity."""
        component = component or self.processor_name.upper()
        self.logger.log_major_component(message, component)
    
    def log_detailed(self, message: str, level: str = "INFO", component: str = None) -> None:
        """Log detailed information."""
        component = component or self.processor_name.upper()
        self.logger.log_detailed(message, level, component)
    
    def log_performance(self, metrics: Dict[str, Any], component: str = None) -> None:
        """Log performance metrics."""
        component = component or f"{self.processor_name.upper()}_PERFORMANCE"
        self.logger.log_performance(metrics, component)
    
    def validate_required_data(self, data: Any, data_name: str) -> None:
        """Validate that required data is present."""
        if data is None or (hasattr(data, 'empty') and data.empty):
            raise ConfigurationError(f"{data_name} is required but not provided")
    
    def validate_options_enabled(self) -> None:
        """Validate that options trading is available - simplified for options-only system."""
        if not hasattr(self.data_loader, 'options_table'):
            raise ConfigurationError("Options data not available - system requires options database")
    
    def get_strategy_configuration(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        if not strategy_name or strategy_name not in self.strategies_config:
            raise ConfigurationError(f"Strategy '{strategy_name}' not found in configuration")
        return self.strategies_config[strategy_name]
    
    def get_case_configuration(self, strategy_name: str, case_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy case."""
        strategy_config = self.get_strategy_configuration(strategy_name)
        cases = strategy_config.get('cases', {})
        
        if case_name not in cases:
            raise ConfigurationError(f"Case '{case_name}' not found in strategy '{strategy_name}'")
        
        return cases[case_name]
    
    def measure_execution_time(self, operation_name: str = None):
        """Context manager for measuring execution time."""
        return ExecutionTimer(operation_name or "operation", self)


class ExecutionTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, operation_name: str, processor: ProcessorBase):
        self.operation_name = operation_name
        self.processor = processor
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        self.processor.performance_stats[f"{self.operation_name}_time"] = execution_time
        self.processor.log_detailed(
            f"{self.operation_name} completed in {execution_time:.4f}s", 
            "INFO", 
            "PERFORMANCE"
        )