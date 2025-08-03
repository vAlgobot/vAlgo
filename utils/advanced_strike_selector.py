#!/usr/bin/env python3
"""
Advanced Strike Selector for vAlgo Trading System
===============================================

Professional strike selection engine for options trading with ATM/ITM/OTM
identification, dynamic selection based on market conditions, and config-driven parameters.

Features:
- ATM strike identification with precision matching
- ITM1-5 and OTM1-5 strike selection algorithms
- Real-time strike selection for live trading
- Config-driven parameters and customization
- Multi-expiry support with DTE filtering
- Integration with OptionsDatabase for fast queries
- Advanced moneyness calculations and Greeks consideration

Author: vAlgo Development Team
Created: July 10, 2025
Version: 1.0.0 (Production)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from data_manager.options_database import OptionsDatabase, create_options_database
from utils.config_cache import get_cached_config


class StrikeType(Enum):
    """Strike type enumeration for option selection."""
    ATM = "ATM"
    ITM1 = "ITM1"
    ITM2 = "ITM2"
    ITM3 = "ITM3"
    ITM4 = "ITM4"
    ITM5 = "ITM5"
    OTM1 = "OTM1"
    OTM2 = "OTM2"
    OTM3 = "OTM3"
    OTM4 = "OTM4"
    OTM5 = "OTM5"


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "CALL"
    PUT = "PUT"


class StrikeSelectionConfig:
    """Configuration class for strike selection parameters."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize strike selection configuration.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        self.config = config_dict or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default strike selection configuration."""
        return {
            'strike_step': 50,                    # Standard strike step for NIFTY
            'max_dte_filter': 30,                # Maximum days to expiry
            'min_dte_filter': 0,                 # Minimum days to expiry
            'atm_tolerance': 25,                 # ATM strike tolerance (points)
            'min_iv_threshold': 5.0,             # Minimum IV for valid strikes
            'max_iv_threshold': 100.0,           # Maximum IV for valid strikes
            'min_delta_call': 0.05,              # Minimum delta for call options
            'max_delta_call': 0.95,              # Maximum delta for call options
            'min_delta_put': -0.95,              # Minimum delta for put options
            'max_delta_put': -0.05,              # Maximum delta for put options
            'min_premium': 1.0,                  # Minimum premium value
            'max_premium': 5000.0,               # Maximum premium value
            'prefer_liquid_strikes': True,       # Prefer more liquid strikes
            'use_greeks_validation': True,       # Use Greeks for strike validation
            'prioritize_recent_data': True,      # Prefer more recent data points
            'selection_method': 'closest_match'  # 'closest_match' or 'delta_based'
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration values."""
        self.config.update(updates)


class AdvancedStrikeSelector:
    """
    Professional strike selector with advanced moneyness calculations
    and real-time strike identification capabilities.
    """
    
    def __init__(self, options_db: Optional[OptionsDatabase] = None,
                 config_path: Optional[str] = None):
        """
        Initialize Advanced Strike Selector.
        
        Args:
            options_db: Optional OptionsDatabase instance
            config_path: Optional path to configuration file
        """
        self.logger = get_logger(__name__)
        self.options_db = options_db or create_options_database()
        self.config_path = config_path or "config/config.xlsx"
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize strike mapping cache
        self._strike_cache = {}
        self._cache_expiry = None
        
        self.logger.info("AdvancedStrikeSelector initialized successfully")
    
    def _load_configuration(self) -> StrikeSelectionConfig:
        """Load strike selection configuration from file and cache."""
        try:
            # Try to load from config cache
            config_cache = get_cached_config(self.config_path)
            config_loader = config_cache.get_config_loader()
            
            # Get options-specific configuration
            options_config = config_loader.get_sl_tp_config()  # Reuse SL/TP config structure
            
            # Extract strike selection parameters
            strike_config = {}
            for key, value in options_config.items():
                if 'strike' in key.lower() or 'option' in key.lower():
                    clean_key = key.replace('default_', '').replace('options_', '')
                    strike_config[clean_key] = value
            
            # Add any additional config parameters
            strike_config.update({
                'strike_step': options_config.get('default_strike_step', 50),
                'max_dte_filter': options_config.get('default_max_dte', 30),
                'atm_tolerance': options_config.get('default_atm_tolerance', 25)
            })
            
            config = StrikeSelectionConfig(strike_config)
            self.logger.info(f"Loaded strike selection config: strike_step={config.get('strike_step')}")
            
            return config
            
        except Exception as e:
            self.logger.warning(f"Could not load strike selection config: {e}, using defaults")
            return StrikeSelectionConfig()
    
    def select_strike(self, underlying_price: float, strike_type: StrikeType,
                     option_type: OptionType, timestamp: datetime,
                     expiry_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Select strike based on underlying price and strike type.
        
        Args:
            underlying_price: Current underlying asset price
            strike_type: Type of strike to select (ATM, ITM1-5, OTM1-5)
            option_type: CALL or PUT option
            timestamp: Timestamp for the selection
            expiry_date: Optional specific expiry date
            
        Returns:
            Dict with strike information or None if not found
        """
        try:
            self.logger.debug(f"Selecting {strike_type.value} {option_type.value} strike "
                            f"for price {underlying_price} at {timestamp}")
            
            # Get available strikes for the timestamp
            available_strikes = self._get_available_strikes(timestamp, expiry_date)
            
            if available_strikes.empty:
                self.logger.debug(f"No available strikes found for {timestamp}")
                return None
            
            # Filter strikes based on quality criteria
            filtered_strikes = self._filter_strikes_by_quality(available_strikes, option_type)
            
            if filtered_strikes.empty:
                self.logger.debug(f"No valid strikes after quality filtering")
                return None
            
            # Find ATM strike first (reference point)
            atm_strike = self._find_atm_strike(underlying_price, filtered_strikes)
            
            if atm_strike is None:
                self.logger.debug(f"Could not find ATM strike for price {underlying_price}")
                return None
            
            # Select target strike based on strike type
            target_strike = self._select_target_strike(
                atm_strike, strike_type, option_type, filtered_strikes
            )
            
            if target_strike is None:
                self.logger.debug(f"Could not find {strike_type.value} strike")
                return None
            
            # Get detailed strike information
            strike_info = self._get_strike_details(
                target_strike, option_type, filtered_strikes, underlying_price
            )
            
            self.logger.debug(f"Selected strike: {target_strike} for {strike_type.value}")
            
            return strike_info
            
        except Exception as e:
            self.logger.error(f"Error selecting strike: {e}")
            return None
    
    def _get_available_strikes(self, timestamp: datetime,
                              expiry_date: Optional[str] = None) -> pd.DataFrame:
        """Get available strikes for a specific timestamp."""
        try:
            # If no specific expiry provided, find the nearest expiry
            if expiry_date is None:
                expiry_date = self._find_nearest_expiry(timestamp)
            
            # Get strikes from database
            strikes_data = self.options_db.get_strikes_for_timestamp(
                timestamp, expiry_date
            )
            
            return strikes_data
            
        except Exception as e:
            self.logger.error(f"Error getting available strikes: {e}")
            return pd.DataFrame()
    
    def _find_nearest_expiry(self, timestamp: datetime) -> Optional[str]:
        """Find the nearest expiry date for the given timestamp."""
        try:
            # Query database for available expiry dates using DuckDB-compatible syntax
            query = f"""
                SELECT DISTINCT expiry_date, 
                       ABS(date_diff('day', ?, expiry_date)) as days_diff
                FROM {self.options_db.options_table}
                WHERE timestamp <= ? 
                AND expiry_date >= ?
                ORDER BY days_diff
                LIMIT 1
            """
            
            # Convert timestamp to proper format for DuckDB
            timestamp_date = timestamp.date()
            
            result = self.options_db.connection.execute(
                query, [timestamp_date, timestamp, timestamp_date]
            ).fetchone()
            
            if result:
                # Handle both string and date object returns
                expiry_date = result[0]
                if hasattr(expiry_date, 'strftime'):
                    return expiry_date.strftime('%Y-%m-%d')
                else:
                    return str(expiry_date)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding nearest expiry: {e}")
            # Fallback: try a simpler query without date arithmetic
            try:
                fallback_query = f"""
                    SELECT DISTINCT expiry_date
                    FROM {self.options_db.options_table}
                    WHERE timestamp <= ? 
                    AND expiry_date >= ?
                    ORDER BY expiry_date
                    LIMIT 1
                """
                
                result = self.options_db.connection.execute(
                    fallback_query, [timestamp, timestamp.date()]
                ).fetchone()
                
                if result:
                    expiry_date = result[0]
                    if hasattr(expiry_date, 'strftime'):
                        return expiry_date.strftime('%Y-%m-%d')
                    else:
                        return str(expiry_date)
                        
            except Exception as fallback_error:
                self.logger.error(f"Fallback expiry query also failed: {fallback_error}")
            
            return None
    
    def _filter_strikes_by_quality(self, strikes_data: pd.DataFrame,
                                  option_type: OptionType) -> pd.DataFrame:
        """Filter strikes based on data quality criteria."""
        try:
            if strikes_data.empty:
                return strikes_data
            
            # Select relevant columns based on option type
            if option_type == OptionType.CALL:
                price_col = 'call_ltp'
                iv_col = 'call_iv'
                delta_col = 'call_delta'
            else:
                price_col = 'put_ltp'
                iv_col = 'put_iv'
                delta_col = 'put_delta'
            
            # Apply quality filters
            config = self.config
            
            # Filter by premium range
            price_filter = (
                (strikes_data[price_col] >= config.get('min_premium', 1.0)) &
                (strikes_data[price_col] <= config.get('max_premium', 5000.0)) &
                (strikes_data[price_col].notna())
            )
            
            # Filter by IV range
            iv_filter = (
                (strikes_data[iv_col] >= config.get('min_iv_threshold', 5.0)) &
                (strikes_data[iv_col] <= config.get('max_iv_threshold', 100.0)) &
                (strikes_data[iv_col].notna())
            )
            
            # Filter by Delta range
            if option_type == OptionType.CALL:
                delta_filter = (
                    (strikes_data[delta_col] >= config.get('min_delta_call', 0.05)) &
                    (strikes_data[delta_col] <= config.get('max_delta_call', 0.95)) &
                    (strikes_data[delta_col].notna())
                )
            else:
                delta_filter = (
                    (strikes_data[delta_col] >= config.get('min_delta_put', -0.95)) &
                    (strikes_data[delta_col] <= config.get('max_delta_put', -0.05)) &
                    (strikes_data[delta_col].notna())
                )
            
            # Combine all filters
            valid_filter = price_filter & iv_filter
            if config.get('use_greeks_validation', True):
                valid_filter = valid_filter & delta_filter
            
            filtered_data = strikes_data[valid_filter].copy()
            
            self.logger.debug(f"Filtered strikes: {len(filtered_data)} valid from {len(strikes_data)} total")
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error filtering strikes by quality: {e}")
            return strikes_data
    
    def _find_atm_strike(self, underlying_price: float,
                        strikes_data: pd.DataFrame) -> Optional[int]:
        """Find the At-The-Money (ATM) strike."""
        try:
            if strikes_data.empty:
                return None
            
            # Calculate distance from underlying price
            strikes_data = strikes_data.copy()
            strikes_data['distance'] = abs(strikes_data['strike'] - underlying_price)
            
            # Find the closest strike
            atm_idx = strikes_data['distance'].idxmin()
            atm_strike = strikes_data.loc[atm_idx, 'strike']
            
            # Validate ATM strike is within tolerance
            tolerance = self.config.get('atm_tolerance', 25)
            if strikes_data.loc[atm_idx, 'distance'] <= tolerance:
                return int(atm_strike)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ATM strike: {e}")
            return None
    
    def _select_target_strike(self, atm_strike: int, strike_type: StrikeType,
                             option_type: OptionType,
                             strikes_data: pd.DataFrame) -> Optional[int]:
        """Select target strike based on strike type and option type."""
        try:
            strike_step = self.config.get('strike_step', 50)
            
            # Handle ATM case
            if strike_type == StrikeType.ATM:
                return atm_strike
            
            # Calculate strike offset based on type
            offset_mapping = {
                StrikeType.ITM1: 1, StrikeType.ITM2: 2, StrikeType.ITM3: 3,
                StrikeType.ITM4: 4, StrikeType.ITM5: 5,
                StrikeType.OTM1: 1, StrikeType.OTM2: 2, StrikeType.OTM3: 3,
                StrikeType.OTM4: 4, StrikeType.OTM5: 5
            }
            
            offset = offset_mapping.get(strike_type, 0)
            if offset == 0:
                return atm_strike
            
            # Calculate target strike based on option type and moneyness
            if (strike_type.value.startswith('ITM') and option_type == OptionType.CALL) or \
               (strike_type.value.startswith('OTM') and option_type == OptionType.PUT):
                # For CALL ITM or PUT OTM, strikes are below underlying
                target_strike = atm_strike - (offset * strike_step)
            else:
                # For CALL OTM or PUT ITM, strikes are above underlying
                target_strike = atm_strike + (offset * strike_step)
            
            # Verify target strike exists in available data
            available_strikes = strikes_data['strike'].unique()
            
            if target_strike in available_strikes:
                return int(target_strike)
            
            # Find closest available strike if exact target not found
            distances = abs(available_strikes - target_strike)
            closest_idx = np.argmin(distances)
            closest_strike = available_strikes[closest_idx]
            
            # Check if closest strike is within reasonable range
            if abs(closest_strike - target_strike) <= strike_step:
                return int(closest_strike)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error selecting target strike: {e}")
            return None
    
    def _get_strike_details(self, strike: int, option_type: OptionType,
                           strikes_data: pd.DataFrame,
                           underlying_price: float) -> Dict[str, Any]:
        """Get detailed information for selected strike."""
        try:
            strike_row = strikes_data[strikes_data['strike'] == strike].iloc[0]
            
            # Select columns based on option type
            if option_type == OptionType.CALL:
                premium = strike_row['call_ltp']
                iv = strike_row['call_iv']
                delta = strike_row['call_delta']
            else:
                premium = strike_row['put_ltp']
                iv = strike_row['put_iv']
                delta = strike_row['put_delta']
            
            # Calculate moneyness
            moneyness = self._calculate_moneyness(strike, underlying_price, option_type)
            
            # Determine strike classification
            strike_classification = self._classify_strike(strike, underlying_price, option_type)
            
            strike_details = {
                'strike': strike,
                'option_type': option_type.value,
                'premium': float(premium),
                'iv': float(iv),
                'delta': float(delta),
                'underlying_price': underlying_price,
                'moneyness': moneyness,
                'classification': strike_classification,
                'expiry_date': strike_row['expiry_date'],
                'dte': int(strike_row['dte']),
                'timestamp': strike_row['timestamp'],
                'intrinsic_value': self._calculate_intrinsic_value(
                    strike, underlying_price, option_type
                ),
                'time_value': float(premium) - self._calculate_intrinsic_value(
                    strike, underlying_price, option_type
                )
            }
            
            return strike_details
            
        except Exception as e:
            self.logger.error(f"Error getting strike details: {e}")
            return {'error': str(e)}
    
    def _calculate_moneyness(self, strike: int, underlying_price: float,
                            option_type: OptionType) -> str:
        """Calculate option moneyness classification."""
        try:
            if option_type == OptionType.CALL:
                if underlying_price > strike:
                    return "ITM"
                elif underlying_price < strike:
                    return "OTM"
                else:
                    return "ATM"
            else:  # PUT
                if underlying_price < strike:
                    return "ITM"
                elif underlying_price > strike:
                    return "OTM"
                else:
                    return "ATM"
                    
        except Exception:
            return "UNKNOWN"
    
    def _classify_strike(self, strike: int, underlying_price: float,
                        option_type: OptionType) -> str:
        """Classify strike based on distance from underlying."""
        try:
            strike_step = self.config.get('strike_step', 50)
            distance = abs(strike - underlying_price)
            steps_away = round(distance / strike_step)
            
            if steps_away == 0:
                return "ATM"
            
            moneyness = self._calculate_moneyness(strike, underlying_price, option_type)
            
            if steps_away <= 5:
                return f"{moneyness}{steps_away}"
            else:
                return f"{moneyness}_FAR"
                
        except Exception:
            return "UNKNOWN"
    
    def _calculate_intrinsic_value(self, strike: int, underlying_price: float,
                                  option_type: OptionType) -> float:
        """Calculate intrinsic value of the option."""
        try:
            if option_type == OptionType.CALL:
                return max(0, underlying_price - strike)
            else:  # PUT
                return max(0, strike - underlying_price)
                
        except Exception:
            return 0.0
    
    def select_multiple_strikes(self, underlying_price: float,
                               strike_types: List[StrikeType],
                               option_type: OptionType, timestamp: datetime,
                               expiry_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Select multiple strikes at once for efficiency.
        
        Args:
            underlying_price: Current underlying asset price
            strike_types: List of strike types to select
            option_type: CALL or PUT option
            timestamp: Timestamp for the selection
            expiry_date: Optional specific expiry date
            
        Returns:
            Dict mapping strike types to strike information
        """
        try:
            results = {}
            
            # Get available strikes once for all selections
            available_strikes = self._get_available_strikes(timestamp, expiry_date)
            
            if available_strikes.empty:
                return results
            
            # Filter strikes once
            filtered_strikes = self._filter_strikes_by_quality(available_strikes, option_type)
            
            if filtered_strikes.empty:
                return results
            
            # Find ATM strike once
            atm_strike = self._find_atm_strike(underlying_price, filtered_strikes)
            
            if atm_strike is None:
                return results
            
            # Select each requested strike type
            for strike_type in strike_types:
                try:
                    target_strike = self._select_target_strike(
                        atm_strike, strike_type, option_type, filtered_strikes
                    )
                    
                    if target_strike is not None:
                        strike_info = self._get_strike_details(
                            target_strike, option_type, filtered_strikes, underlying_price
                        )
                        results[strike_type.value] = strike_info
                        
                except Exception as e:
                    self.logger.warning(f"Error selecting {strike_type.value}: {e}")
                    continue
            
            self.logger.info(f"Selected {len(results)} strikes from {len(strike_types)} requested")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error selecting multiple strikes: {e}")
            return {}
    
    def get_strike_chain(self, underlying_price: float, timestamp: datetime,
                        option_type: OptionType, range_strikes: int = 10,
                        expiry_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get a chain of strikes around the underlying price.
        
        Args:
            underlying_price: Current underlying asset price
            timestamp: Timestamp for the selection
            option_type: CALL or PUT option
            range_strikes: Number of strikes above and below ATM to include
            expiry_date: Optional specific expiry date
            
        Returns:
            DataFrame with strike chain information
        """
        try:
            # Get available strikes
            available_strikes = self._get_available_strikes(timestamp, expiry_date)
            
            if available_strikes.empty:
                return pd.DataFrame()
            
            # Filter strikes
            filtered_strikes = self._filter_strikes_by_quality(available_strikes, option_type)
            
            if filtered_strikes.empty:
                return pd.DataFrame()
            
            # Find ATM strike
            atm_strike = self._find_atm_strike(underlying_price, filtered_strikes)
            
            if atm_strike is None:
                return pd.DataFrame()
            
            # Define strike range
            strike_step = self.config.get('strike_step', 50)
            min_strike = atm_strike - (range_strikes * strike_step)
            max_strike = atm_strike + (range_strikes * strike_step)
            
            # Filter strikes in range
            range_filter = (
                (filtered_strikes['strike'] >= min_strike) &
                (filtered_strikes['strike'] <= max_strike)
            )
            
            chain_data = filtered_strikes[range_filter].copy()
            
            # Add additional calculated fields
            if option_type == OptionType.CALL:
                chain_data['premium'] = chain_data['call_ltp']
                chain_data['iv'] = chain_data['call_iv']
                chain_data['delta'] = chain_data['call_delta']
            else:
                chain_data['premium'] = chain_data['put_ltp']
                chain_data['iv'] = chain_data['put_iv']
                chain_data['delta'] = chain_data['put_delta']
            
            # Calculate moneyness for each strike
            chain_data['moneyness'] = chain_data['strike'].apply(
                lambda x: self._calculate_moneyness(x, underlying_price, option_type)
            )
            
            # Calculate intrinsic and time value
            chain_data['intrinsic_value'] = chain_data['strike'].apply(
                lambda x: self._calculate_intrinsic_value(x, underlying_price, option_type)
            )
            chain_data['time_value'] = chain_data['premium'] - chain_data['intrinsic_value']
            
            # Sort by strike
            chain_data = chain_data.sort_values('strike').reset_index(drop=True)
            
            return chain_data
            
        except Exception as e:
            self.logger.error(f"Error getting strike chain: {e}")
            return pd.DataFrame()


# Convenience functions
def select_option_strike(underlying_price: float, strike_type: str,
                        option_type: str, timestamp: datetime,
                        options_db: Optional[OptionsDatabase] = None) -> Optional[Dict[str, Any]]:
    """
    Convenience function to select a single option strike.
    
    Args:
        underlying_price: Current underlying asset price
        strike_type: Strike type string (ATM, ITM1-5, OTM1-5)
        option_type: Option type string (CALL, PUT)
        timestamp: Timestamp for the selection
        options_db: Optional OptionsDatabase instance
        
    Returns:
        Strike information dictionary or None
    """
    try:
        selector = AdvancedStrikeSelector(options_db)
        
        # Convert string enums
        strike_enum = StrikeType(strike_type.upper())
        option_enum = OptionType(option_type.upper())
        
        return selector.select_strike(
            underlying_price, strike_enum, option_enum, timestamp
        )
        
    except Exception as e:
        logging.error(f"Error in convenience function: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Example usage of the Advanced Strike Selector
    try:
        # Create selector
        selector = AdvancedStrikeSelector()
        
        # Test parameters
        test_underlying_price = 21650.0
        test_timestamp = datetime(2024, 1, 23, 9, 30)
        
        print(f"Testing strike selection for underlying price: {test_underlying_price}")
        print(f"Timestamp: {test_timestamp}")
        
        # Test single strike selection
        strike_types_to_test = [
            StrikeType.ATM, StrikeType.ITM1, StrikeType.ITM2,
            StrikeType.OTM1, StrikeType.OTM2
        ]
        
        for option_type in [OptionType.CALL, OptionType.PUT]:
            print(f"\n--- {option_type.value} Options ---")
            
            for strike_type in strike_types_to_test:
                result = selector.select_strike(
                    test_underlying_price, strike_type, option_type, test_timestamp
                )
                
                if result:
                    print(f"{strike_type.value}: Strike {result['strike']}, "
                          f"Premium {result['premium']:.2f}, "
                          f"IV {result['iv']:.1f}%, "
                          f"Delta {result['delta']:.3f}")
                else:
                    print(f"{strike_type.value}: Not found")
        
        # Test multiple strike selection
        print(f"\n--- Multiple Strike Selection ---")
        multiple_results = selector.select_multiple_strikes(
            test_underlying_price,
            [StrikeType.ITM2, StrikeType.ATM, StrikeType.OTM2],
            OptionType.CALL,
            test_timestamp
        )
        
        for strike_type, info in multiple_results.items():
            print(f"{strike_type}: Strike {info['strike']}, Premium {info['premium']:.2f}")
        
        # Test strike chain
        print(f"\n--- Strike Chain (CALL options) ---")
        chain = selector.get_strike_chain(
            test_underlying_price, test_timestamp, OptionType.CALL, range_strikes=3
        )
        
        if not chain.empty:
            print(f"Found {len(chain)} strikes in chain:")
            for _, row in chain.iterrows():
                print(f"  Strike {row['strike']}: Premium {row['premium']:.2f}, "
                      f"Moneyness {row['moneyness']}")
        else:
            print("No strikes found in chain")
            
    except Exception as e:
        print(f"Error in example usage: {e}")