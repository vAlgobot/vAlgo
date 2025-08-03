#!/usr/bin/env python3
"""
Instrument Configuration Utility for vAlgo Trading System
========================================================

Provides dynamic access to instrument-specific configuration from Excel config file.
Handles lot sizes, strike steps, and other symbol-specific parameters.

Features:
- Dynamic lot size lookup by symbol
- Strike step configuration per instrument  
- Default symbol management
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
from utils.config_cache import get_cached_config


class InstrumentConfig:
    """
    Utility class for accessing instrument-specific configuration.
    
    Provides dynamic lookup of lot sizes, strike steps and other
    instrument parameters from the Instruments sheet in config.xlsx.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize instrument configuration utility.
        
        Args:
            config_path: Path to Excel config file (optional)
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path or "config/config.xlsx"
        self._instruments_cache = None
        self._load_instruments_config()
    
    def _load_instruments_config(self) -> None:
        """Load instruments configuration from Excel config file."""
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
            
            # Look for 'instruments' (lowercase) - this is how ConfigLoader stores it
            if 'instruments' not in config:
                # Enhanced error with all available keys
                available_keys = list(config.keys()) if isinstance(config, dict) else []
                raise ValueError(
                    f"Instruments sheet not found in config dictionary. "
                    f"Available keys: {available_keys}. "
                    f"ConfigLoader stores sheet names in lowercase - looking for 'instruments'."
                )
            
            self._instruments_cache = config['instruments']
            self.logger.info(f"Loaded instruments config: {len(self._instruments_cache)} instruments")
            
        except Exception as e:
            self.logger.error(f"Failed to load instruments config: {e}")
            self.logger.error(f"Config path: {self.config_path}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            
            # STRICT: No fallback - throw clear error
            raise ValueError(
                f"Instruments sheet loading failed from '{self.config_path}'. "
                f"Excel file must contain 'Instruments' sheet with Symbol and Lot_Size columns. "
                f"Original error: {e}"
            )
    
    def get_lot_size_for_symbol(self, symbol: str) -> Optional[int]:
        """
        Get lot size for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY', 'BANKNIFTY')
            
        Returns:
            Lot size as integer, or None if not found
        """
        try:
            if self._instruments_cache is None:
                raise ValueError("Instruments config was not loaded. Check Excel file access.")
            
            if self._instruments_cache.empty:
                raise ValueError("Instruments sheet is empty. Check Excel file contains data.")
            
            # Find row for the symbol
            symbol_rows = self._instruments_cache[self._instruments_cache['Symbol'] == symbol]
            
            if symbol_rows.empty:
                self.logger.warning(f"Symbol '{symbol}' not found in Instruments config")
                return None
            
            # Get lot size from first matching row
            lot_size = symbol_rows['Lot_Size'].iloc[0]
            
            if pd.isna(lot_size):
                self.logger.warning(f"Lot_Size not configured for symbol '{symbol}'")
                return None
            
            lot_size_int = int(lot_size)
            self.logger.debug(f"Lot size for {symbol}: {lot_size_int}")
            return lot_size_int
            
        except Exception as e:
            self.logger.error(f"Error getting lot size for symbol '{symbol}': {e}")
            return None
    
    def get_strike_step_for_symbol(self, symbol: str) -> Optional[int]:
        """
        Get strike step for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY', 'BANKNIFTY')
            
        Returns:
            Strike step as integer, or None if not found
        """
        try:
            if self._instruments_cache is None:
                raise ValueError("Instruments config was not loaded. Check Excel file access.")
            
            if self._instruments_cache.empty:
                raise ValueError("Instruments sheet is empty. Check Excel file contains data.")
            
            # Find row for the symbol
            symbol_rows = self._instruments_cache[self._instruments_cache['Symbol'] == symbol]
            
            if symbol_rows.empty:
                self.logger.warning(f"Symbol '{symbol}' not found in Instruments config")
                return None
            
            # Get strike step from first matching row
            strike_step = symbol_rows['Strike_Step'].iloc[0]
            
            if pd.isna(strike_step):
                self.logger.warning(f"Strike_Step not configured for symbol '{symbol}'")
                return None
            
            strike_step_int = int(strike_step)
            self.logger.debug(f"Strike step for {symbol}: {strike_step_int}")
            return strike_step_int
            
        except Exception as e:
            self.logger.error(f"Error getting strike step for symbol '{symbol}': {e}")
            return None
    
    def get_default_symbol(self) -> Optional[str]:
        """
        Get the default trading symbol (first symbol in instruments config).
        
        Returns:
            Default symbol name or None if no instruments configured
        """
        try:
            if self._instruments_cache is None:
                raise ValueError("Instruments config was not loaded. Check Excel file access.")
            
            if self._instruments_cache.empty:
                raise ValueError("Instruments sheet is empty. Check Excel file contains data.")
            
            # Get first symbol from instruments
            first_symbol = self._instruments_cache['Symbol'].iloc[0]
            
            if pd.isna(first_symbol):
                self.logger.warning("No valid symbols found in Instruments config")
                return None
            
            default_symbol = str(first_symbol)
            self.logger.debug(f"Default symbol: {default_symbol}")
            return default_symbol
            
        except Exception as e:
            self.logger.error(f"Error getting default symbol: {e}")
            return None
    
    def get_all_symbols(self) -> List[str]:
        """
        Get list of all configured symbols.
        
        Returns:
            List of symbol names
        """
        try:
            if self._instruments_cache is None or self._instruments_cache.empty:
                self.logger.warning("Instruments config not available")
                return []
            
            # Get all symbols, filter out NaN values
            symbols = self._instruments_cache['Symbol'].dropna().tolist()
            symbols = [str(symbol) for symbol in symbols]
            
            self.logger.debug(f"All symbols: {symbols}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting all symbols: {e}")
            return []
    
    def get_instrument_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get complete instrument information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with all instrument parameters
        """
        try:
            if self._instruments_cache is None or self._instruments_cache.empty:
                self.logger.warning("Instruments config not available")
                return {}
            
            # Find row for the symbol
            symbol_rows = self._instruments_cache[self._instruments_cache['Symbol'] == symbol]
            
            if symbol_rows.empty:
                self.logger.warning(f"Symbol '{symbol}' not found in Instruments config")
                return {}
            
            # Convert first matching row to dictionary
            instrument_info = symbol_rows.iloc[0].to_dict()
            
            # Clean up NaN values
            cleaned_info = {k: v for k, v in instrument_info.items() if not pd.isna(v)}
            
            self.logger.debug(f"Instrument info for {symbol}: {cleaned_info}")
            return cleaned_info
            
        except Exception as e:
            self.logger.error(f"Error getting instrument info for symbol '{symbol}': {e}")
            return {}
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is configured in instruments.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if symbol is configured, False otherwise
        """
        try:
            if self._instruments_cache is None or self._instruments_cache.empty:
                return False
            
            # Check if symbol exists
            symbol_exists = (self._instruments_cache['Symbol'] == symbol).any()
            
            if symbol_exists:
                self.logger.debug(f"Symbol '{symbol}' is valid")
            else:
                self.logger.warning(f"Symbol '{symbol}' not found in configuration")
            
            return symbol_exists
            
        except Exception as e:
            self.logger.error(f"Error validating symbol '{symbol}': {e}")
            return False


# Convenience functions for easy import and usage
def get_lot_size_for_symbol(symbol: str, config_path: Optional[str] = None) -> Optional[int]:
    """Convenience function to get lot size for a symbol."""
    instrument_config = InstrumentConfig(config_path)
    return instrument_config.get_lot_size_for_symbol(symbol)


def get_strike_step_for_symbol(symbol: str, config_path: Optional[str] = None) -> Optional[int]:
    """Convenience function to get strike step for a symbol."""
    instrument_config = InstrumentConfig(config_path)
    return instrument_config.get_strike_step_for_symbol(symbol)


def get_default_symbol(config_path: Optional[str] = None) -> Optional[str]:
    """Convenience function to get default symbol."""
    instrument_config = InstrumentConfig(config_path)
    return instrument_config.get_default_symbol()


# Example usage and testing
if __name__ == "__main__":
    # Test the InstrumentConfig class
    instrument_config = InstrumentConfig()
    
    # Test getting lot size
    nifty_lot_size = instrument_config.get_lot_size_for_symbol('NIFTY')
    print(f"NIFTY lot size: {nifty_lot_size}")
    
    # Test getting strike step
    nifty_strike_step = instrument_config.get_strike_step_for_symbol('NIFTY')
    print(f"NIFTY strike step: {nifty_strike_step}")
    
    # Test getting default symbol
    default_symbol = instrument_config.get_default_symbol()
    print(f"Default symbol: {default_symbol}")
    
    # Test getting all symbols
    all_symbols = instrument_config.get_all_symbols()
    print(f"All symbols: {all_symbols}")