#!/usr/bin/env python3
"""
Strategy Configuration Helper for vAlgo Trading System
=====================================================

Provides dynamic access to strategy-specific configuration from Excel config file.
Handles signal types, position sizes, and other strategy-specific parameters.

Features:
- Dynamic signal type lookup by strategy name
- Position size configuration per strategy
- Breakout confirmation settings
- Integration with existing config cache system

Author: vAlgo Development Team
Created: July 22, 2025
Version: 1.0.0 (Production)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger


class StrategyConfigHelper:
    """
    Utility class for accessing strategy-specific configuration.
    
    Provides dynamic lookup of signal types, position sizes and other
    strategy parameters from the Strategy_Config sheet in config.xlsx.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize strategy configuration helper.
        
        Args:
            config_path: Path to Excel config file (optional)
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path or "config/config.xlsx"
        self._strategy_cache = None
        self._load_strategy_config()
    
    def _load_strategy_config(self) -> None:
        """Load strategy configuration from Excel config file."""
        try:
            # Use direct ConfigLoader (established working approach)
            from utils.config_loader import ConfigLoader
            config_loader = ConfigLoader(self.config_path)
            
            # Load config successfully
            success = config_loader.load_config()
            if not success:
                raise ValueError(f"Failed to load config from '{self.config_path}'")
            
            # Access the config data (ConfigLoader stores sheets with lowercase keys)
            config = config_loader.config_data
            
            # Look for 'strategy_config' (lowercase) - this is how ConfigLoader stores it
            if 'strategy_config' not in config:
                # Enhanced error with all available keys
                available_keys = list(config.keys()) if isinstance(config, dict) else []
                raise ValueError(
                    f"Strategy_Config sheet not found in config dictionary. "
                    f"Available keys: {available_keys}. "
                    f"ConfigLoader stores sheet names in lowercase - looking for 'strategy_config'."
                )
            
            self._strategy_cache = config['strategy_config']
            self.logger.info(f"Loaded strategy config: {len(self._strategy_cache)} strategies")
            
        except Exception as e:
            self.logger.error(f"Failed to load strategy config: {e}")
            self.logger.error(f"Config path: {self.config_path}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            
            # STRICT: No fallback - throw clear error
            raise ValueError(
                f"Strategy_Config sheet loading failed from '{self.config_path}'. "
                f"Excel file must contain 'Strategy_Config' sheet with Strategy_Name and configuration columns. "
                f"Original error: {e}"
            )
    
    def get_signal_type_for_strategy(self, strategy_name: str) -> Optional[str]:
        """
        Get signal type (CALL/PUT) for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Signal type ('CALL' or 'PUT'), or None if not found
        """
        try:
            if self._strategy_cache is None:
                raise ValueError("Strategy config was not loaded. Check Excel file access.")
            
            if self._strategy_cache.empty:
                raise ValueError("Strategy_Config sheet is empty. Check Excel file contains data.")
            
            # Find row for the strategy
            strategy_rows = self._strategy_cache[self._strategy_cache['Strategy_Name'] == strategy_name]
            
            if strategy_rows.empty:
                self.logger.warning(f"Strategy '{strategy_name}' not found in Strategy_Config")
                return None
            
            # Try multiple possible column names for signal type
            signal_type_columns = ['signal_type', 'Signal_Type', 'option_type', 'Option_Type', 'Signal_Type']
            signal_type = None
            
            for col in signal_type_columns:
                if col in strategy_rows.columns:
                    value = strategy_rows[col].iloc[0]
                    if not pd.isna(value) and str(value).strip():
                        signal_type = str(value).strip().upper()
                        break
            
            if signal_type:
                # Normalize signal type values
                if signal_type in ['CALL', 'CE', 'C', 'LONG', 'BUY']:
                    signal_type = 'CALL'
                elif signal_type in ['PUT', 'PE', 'P', 'SHORT', 'SELL']:
                    signal_type = 'PUT'
                
                self.logger.debug(f"Signal type for {strategy_name}: {signal_type}")
                return signal_type
            else:
                self.logger.warning(f"Signal type not configured for strategy '{strategy_name}'")
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting signal type for strategy '{strategy_name}': {e}")
            return None
    
    def get_position_size_for_strategy(self, strategy_name: str) -> Optional[int]:
        """
        Get position size for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Position size as integer, or None if not found
        """
        try:
            if self._strategy_cache is None:
                raise ValueError("Strategy config was not loaded. Check Excel file access.")
            
            if self._strategy_cache.empty:
                raise ValueError("Strategy_Config sheet is empty. Check Excel file contains data.")
            
            # Find row for the strategy
            strategy_rows = self._strategy_cache[self._strategy_cache['Strategy_Name'] == strategy_name]
            
            if strategy_rows.empty:
                self.logger.warning(f"Strategy '{strategy_name}' not found in Strategy_Config")
                return None
            
            # Try multiple possible column names for position size
            position_size_columns = ['Position_Size', 'position_size', 'Position_size', 'PositionSize']
            position_size = None
            
            for col in position_size_columns:
                if col in strategy_rows.columns:
                    value = strategy_rows[col].iloc[0]
                    if not pd.isna(value):
                        position_size = int(float(value))
                        break
            
            if position_size is not None:
                self.logger.debug(f"Position size for {strategy_name}: {position_size}")
                return position_size
            else:
                self.logger.warning(f"Position size not configured for strategy '{strategy_name}'")
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting position size for strategy '{strategy_name}': {e}")
            return None
    
    def get_breakout_confirmation_for_strategy(self, strategy_name: str) -> bool:
        """
        Get breakout confirmation setting for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            True if breakout confirmation is enabled, False otherwise
        """
        try:
            if self._strategy_cache is None or self._strategy_cache.empty:
                self.logger.warning("Strategy config not available")
                return False
            
            # Find row for the strategy
            strategy_rows = self._strategy_cache[self._strategy_cache['Strategy_Name'] == strategy_name]
            
            if strategy_rows.empty:
                self.logger.warning(f"Strategy '{strategy_name}' not found in Strategy_Config")
                return False
            
            # Try multiple possible column names for breakout confirmation
            breakout_columns = ['breakout_confirmation', 'Breakout_Confirmation', 'breakout', 'Breakout']
            breakout_confirmation = False
            
            for col in breakout_columns:
                if col in strategy_rows.columns:
                    value = strategy_rows[col].iloc[0]
                    if not pd.isna(value):
                        # Handle various boolean representations
                        if isinstance(value, bool):
                            breakout_confirmation = value
                        elif isinstance(value, str):
                            breakout_confirmation = value.lower() in ['true', 'yes', '1', 'on', 'enabled']
                        elif isinstance(value, (int, float)):
                            breakout_confirmation = bool(value)
                        break
            
            self.logger.debug(f"Breakout confirmation for {strategy_name}: {breakout_confirmation}")
            return breakout_confirmation
            
        except Exception as e:
            self.logger.error(f"Error getting breakout confirmation for strategy '{strategy_name}': {e}")
            return False
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get complete strategy information.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with all strategy parameters
        """
        try:
            if self._strategy_cache is None or self._strategy_cache.empty:
                self.logger.warning("Strategy config not available")
                return {}
            
            # Find row for the strategy
            strategy_rows = self._strategy_cache[self._strategy_cache['Strategy_Name'] == strategy_name]
            
            if strategy_rows.empty:
                self.logger.warning(f"Strategy '{strategy_name}' not found in Strategy_Config")
                return {}
            
            # Convert first matching row to dictionary
            strategy_info = strategy_rows.iloc[0].to_dict()
            
            # Clean up NaN values
            cleaned_info = {k: v for k, v in strategy_info.items() if not pd.isna(v)}
            
            self.logger.debug(f"Strategy info for {strategy_name}: {cleaned_info}")
            return cleaned_info
            
        except Exception as e:
            self.logger.error(f"Error getting strategy info for '{strategy_name}': {e}")
            return {}
    
    def get_all_strategy_names(self) -> List[str]:
        """
        Get list of all configured strategy names.
        
        Returns:
            List of strategy names
        """
        try:
            if self._strategy_cache is None or self._strategy_cache.empty:
                self.logger.warning("Strategy config not available")
                return []
            
            # Get all strategy names, filter out NaN values
            strategy_names = self._strategy_cache['Strategy_Name'].dropna().tolist()
            strategy_names = [str(name) for name in strategy_names]
            
            self.logger.debug(f"All strategy names: {strategy_names}")
            return strategy_names
            
        except Exception as e:
            self.logger.error(f"Error getting all strategy names: {e}")
            return []
    
    def validate_strategy(self, strategy_name: str) -> bool:
        """
        Validate if a strategy is configured.
        
        Args:
            strategy_name: Strategy name to validate
            
        Returns:
            True if strategy is configured, False otherwise
        """
        try:
            if self._strategy_cache is None or self._strategy_cache.empty:
                return False
            
            # Check if strategy exists
            strategy_exists = (self._strategy_cache['Strategy_Name'] == strategy_name).any()
            
            if strategy_exists:
                self.logger.debug(f"Strategy '{strategy_name}' is valid")
            else:
                self.logger.warning(f"Strategy '{strategy_name}' not found in configuration")
            
            return strategy_exists
            
        except Exception as e:
            self.logger.error(f"Error validating strategy '{strategy_name}': {e}")
            return False


# Convenience functions for easy import and usage
def get_signal_type_for_strategy(strategy_name: str, config_path: Optional[str] = None) -> Optional[str]:
    """Convenience function to get signal type for a strategy."""
    strategy_helper = StrategyConfigHelper(config_path)
    return strategy_helper.get_signal_type_for_strategy(strategy_name)


def get_position_size_for_strategy(strategy_name: str, config_path: Optional[str] = None) -> Optional[int]:
    """Convenience function to get position size for a strategy."""
    strategy_helper = StrategyConfigHelper(config_path)
    return strategy_helper.get_position_size_for_strategy(strategy_name)


def get_breakout_confirmation_for_strategy(strategy_name: str, config_path: Optional[str] = None) -> bool:
    """Convenience function to get breakout confirmation setting for a strategy."""
    strategy_helper = StrategyConfigHelper(config_path)
    return strategy_helper.get_breakout_confirmation_for_strategy(strategy_name)


# Example usage and testing
if __name__ == "__main__":
    # Test the StrategyConfigHelper class
    strategy_helper = StrategyConfigHelper()
    
    # Test getting signal type
    signal_type = strategy_helper.get_signal_type_for_strategy('EMA_RSI_Scalping')
    print(f"EMA_RSI_Scalping signal type: {signal_type}")
    
    # Test getting position size
    position_size = strategy_helper.get_position_size_for_strategy('EMA_RSI_Scalping')
    print(f"EMA_RSI_Scalping position size: {position_size}")
    
    # Test getting breakout confirmation
    breakout_confirmation = strategy_helper.get_breakout_confirmation_for_strategy('EMA_RSI_Scalping')
    print(f"EMA_RSI_Scalping breakout confirmation: {breakout_confirmation}")
    
    # Test getting all strategy names
    all_strategies = strategy_helper.get_all_strategy_names()
    print(f"All strategies: {all_strategies}")