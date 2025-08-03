"""
Option Chain Database Integrator for vAlgo Trading System
Framework for future option chain data integration
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger


class OptionChainIntegrator:
    """
    Integration layer for option chain data from database
    Provides framework for future option chain database integration
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize Option Chain Integrator
        
        Args:
            database_path: Path to the database containing option chain data
        """
        self.logger = get_logger(__name__)
        self.database_path = database_path
        self.option_chain_available = False
        self.supported_symbols = []
        
        # Initialize database connection
        self._initialize_database()
        
        self.logger.info(f"OptionChainIntegrator initialized - Data available: {self.option_chain_available}")
    
    def _initialize_database(self) -> None:
        """Initialize database connection and check for option chain tables"""
        try:
            # Future implementation: Initialize database connection
            # Check for option chain tables and data availability
            
            # For now, set to False as option chain data is not yet available
            self.option_chain_available = False
            
            # When option chain data is available, this will be implemented:
            # - Connect to database
            # - Check for option chain tables
            # - Verify data integrity
            # - Load supported symbols
            
            if self.option_chain_available:
                self.logger.info("Option chain data detected and validated")
                self._load_supported_symbols()
            else:
                self.logger.info("Option chain data not available - using dummy mode")
                
        except Exception as e:
            self.logger.error(f"Error initializing option chain database: {e}")
            self.option_chain_available = False
    
    def _load_supported_symbols(self) -> None:
        """Load list of symbols with available option chain data"""
        try:
            # Future implementation: Query database for available symbols
            # self.supported_symbols = database.get_symbols_with_option_chain()
            
            self.supported_symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']  # Placeholder
            self.logger.info(f"Loaded {len(self.supported_symbols)} symbols with option chain data")
            
        except Exception as e:
            self.logger.error(f"Error loading supported symbols: {e}")
            self.supported_symbols = []
    
    def is_option_chain_available(self, symbol: str) -> bool:
        """
        Check if option chain data is available for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if option chain data is available
        """
        return self.option_chain_available and symbol in self.supported_symbols
    
    def get_option_premium(self, symbol: str, strike_price: float, expiry_date: datetime,
                          option_type: str, timestamp: datetime) -> Optional[float]:
        """
        Get option premium for specific parameters
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY')
            strike_price: Strike price of the option
            expiry_date: Expiry date of the option
            option_type: 'CALL' or 'PUT'
            timestamp: Timestamp for premium lookup
            
        Returns:
            Option premium or None if not available
        """
        try:
            if not self.is_option_chain_available(symbol):
                return None
            
            # Future implementation: Query option chain database
            # premium = database.get_option_premium(
            #     symbol=symbol,
            #     strike=strike_price,
            #     expiry=expiry_date,
            #     option_type=option_type,
            #     timestamp=timestamp
            # )
            
            # Placeholder implementation
            self.logger.debug(f"Option premium lookup: {symbol} {strike_price} {option_type} {expiry_date}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting option premium: {e}")
            return None
    
    def get_strike_prices(self, symbol: str, expiry_date: datetime, 
                         spot_price: float, option_type: str) -> List[float]:
        """
        Get available strike prices for an option
        
        Args:
            symbol: Trading symbol
            expiry_date: Expiry date
            spot_price: Current spot price
            option_type: 'CALL' or 'PUT'
            
        Returns:
            List of available strike prices
        """
        try:
            if not self.is_option_chain_available(symbol):
                return []
            
            # Future implementation: Query available strikes from database
            # strikes = database.get_available_strikes(
            #     symbol=symbol,
            #     expiry=expiry_date,
            #     option_type=option_type
            # )
            
            # Placeholder implementation - generate strikes around spot price
            strike_interval = 50 if symbol == 'NIFTY' else 100
            strikes = []
            
            for i in range(-10, 11):  # 10 strikes above and below spot
                strike = round(spot_price + (i * strike_interval), 0)
                strikes.append(strike)
            
            return strikes
            
        except Exception as e:
            self.logger.error(f"Error getting strike prices: {e}")
            return []
    
    def get_expiry_dates(self, symbol: str, current_date: datetime) -> List[datetime]:
        """
        Get available expiry dates for an option
        
        Args:
            symbol: Trading symbol
            current_date: Current date
            
        Returns:
            List of available expiry dates
        """
        try:
            if not self.is_option_chain_available(symbol):
                return []
            
            # Future implementation: Query available expiries from database
            # expiries = database.get_available_expiries(symbol=symbol, date=current_date)
            
            # Placeholder implementation - generate weekly/monthly expiries
            expiries = []
            
            # Next 4 weekly expiries (Thursdays)
            current_date = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            days_until_thursday = (3 - current_date.weekday()) % 7
            next_thursday = current_date + timedelta(days=days_until_thursday)
            
            for i in range(4):
                expiry = next_thursday + timedelta(weeks=i)
                expiries.append(expiry)
            
            return expiries
            
        except Exception as e:
            self.logger.error(f"Error getting expiry dates: {e}")
            return []
    
    def calculate_option_greeks(self, symbol: str, strike_price: float, expiry_date: datetime,
                              option_type: str, spot_price: float, timestamp: datetime) -> Dict[str, float]:
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega)
        
        Args:
            symbol: Trading symbol
            strike_price: Strike price
            expiry_date: Expiry date
            option_type: 'CALL' or 'PUT'
            spot_price: Current spot price
            timestamp: Timestamp for calculation
            
        Returns:
            Dictionary with Greeks values
        """
        try:
            if not self.is_option_chain_available(symbol):
                return {}
            
            # Future implementation: Calculate or fetch Greeks from database
            # This would involve Black-Scholes calculations or database lookups
            
            # Placeholder implementation
            greeks = {
                'delta': 0.5,
                'gamma': 0.1,
                'theta': -0.05,
                'vega': 0.2,
                'rho': 0.1
            }
            
            return greeks
            
        except Exception as e:
            self.logger.error(f"Error calculating option Greeks: {e}")
            return {}
    
    def get_option_chain_snapshot(self, symbol: str, expiry_date: datetime,
                                 timestamp: datetime) -> Dict[str, Any]:
        """
        Get complete option chain snapshot for a symbol and expiry
        
        Args:
            symbol: Trading symbol
            expiry_date: Expiry date
            timestamp: Timestamp for snapshot
            
        Returns:
            Option chain snapshot data
        """
        try:
            if not self.is_option_chain_available(symbol):
                return {}
            
            # Future implementation: Get complete option chain from database
            # chain = database.get_option_chain_snapshot(
            #     symbol=symbol,
            #     expiry=expiry_date,
            #     timestamp=timestamp
            # )
            
            # Placeholder implementation
            snapshot = {
                'symbol': symbol,
                'expiry_date': expiry_date.isoformat(),
                'timestamp': timestamp.isoformat(),
                'calls': {},
                'puts': {},
                'underlying_price': 0,
                'total_call_oi': 0,
                'total_put_oi': 0,
                'pcr': 0,
            }
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error getting option chain snapshot: {e}")
            return {}
    
    def find_optimal_strike(self, symbol: str, expiry_date: datetime, option_type: str,
                           spot_price: float, strategy: str = 'ATM') -> Optional[float]:
        """
        Find optimal strike price based on strategy
        
        Args:
            symbol: Trading symbol
            expiry_date: Expiry date
            option_type: 'CALL' or 'PUT'
            spot_price: Current spot price
            strategy: Strike selection strategy ('ATM', 'ITM', 'OTM')
            
        Returns:
            Optimal strike price or None
        """
        try:
            available_strikes = self.get_strike_prices(symbol, expiry_date, spot_price, option_type)
            
            if not available_strikes:
                return None
            
            if strategy == 'ATM':
                # Find strike closest to spot price
                return min(available_strikes, key=lambda x: abs(x - spot_price))
            elif strategy == 'ITM':
                # Find in-the-money strikes
                if option_type == 'CALL':
                    itm_strikes = [s for s in available_strikes if s < spot_price]
                    return max(itm_strikes) if itm_strikes else None
                else:  # PUT
                    itm_strikes = [s for s in available_strikes if s > spot_price]
                    return min(itm_strikes) if itm_strikes else None
            elif strategy == 'OTM':
                # Find out-of-the-money strikes
                if option_type == 'CALL':
                    otm_strikes = [s for s in available_strikes if s > spot_price]
                    return min(otm_strikes) if otm_strikes else None
                else:  # PUT
                    otm_strikes = [s for s in available_strikes if s < spot_price]
                    return max(otm_strikes) if otm_strikes else None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding optimal strike: {e}")
            return None
    
    def get_database_status(self) -> Dict[str, Any]:
        """
        Get database status and statistics
        
        Returns:
            Database status information
        """
        try:
            status = {
                'option_chain_available': self.option_chain_available,
                'database_path': self.database_path,
                'supported_symbols': self.supported_symbols,
                'total_symbols': len(self.supported_symbols),
                'last_updated': None,  # Future implementation
                'data_quality': 'Unknown',  # Future implementation
                'coverage_percentage': 0,  # Future implementation
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting database status: {e}")
            return {}


def create_option_chain_integrator(database_path: Optional[str] = None) -> OptionChainIntegrator:
    """
    Factory function to create OptionChainIntegrator
    
    Args:
        database_path: Path to option chain database
        
    Returns:
        OptionChainIntegrator instance
    """
    return OptionChainIntegrator(database_path)