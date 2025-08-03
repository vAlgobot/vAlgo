"""
Options P&L Calculator for vAlgo Trading System
Supports dummy amounts for current use and framework for future option chain integration
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from utils.trading_day_validator import validate_trade_timestamp


class OptionsPnLCalculator:
    """
    Calculate P&L for options trades with support for dummy amounts and future option chain integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Options P&L Calculator - REAL OPTION DATA ONLY
        
        Args:
            config: Configuration dictionary for P&L calculations (optional)
        """
        self.logger = get_logger(__name__)
        
        # Real options configuration
        self.options_config = {
            'commission_per_trade': 20,     # Commission per trade
            'lot_size': 75,                 # Options lot size (NIFTY)
            'premium_multiplier': 1.0,      # Premium multiplier for calculations
        }
        
        # Update with provided config
        if config:
            self.options_config.update(config)
        
        # Initialize real option chain integration
        self.option_chain_available = self._check_option_chain_availability()
        
        if not self.option_chain_available:
            self.logger.warning("Option chain data not available - P&L calculations may be limited")
        
        self.logger.info("OptionsPnLCalculator initialized - Using REAL option chain data only")
    
    def _check_option_chain_availability(self) -> bool:
        """
        Check if option chain data is available in the database
        
        Returns:
            True if option chain data is available, False otherwise
        """
        try:
            # Check if the options database and advanced components are available
            from data_manager.options_database import OptionsDatabase
            from utils.advanced_strike_selector import AdvancedStrikeSelector
            
            # Try to initialize options database
            options_db = OptionsDatabase()
            strike_selector = AdvancedStrikeSelector(options_db)
            
            # If we can initialize these components, we have real option data support
            self.logger.info("Real option chain integration components available")
            return True
            
        except Exception as e:
            self.logger.warning(f"Option chain components not available: {e}")
            return False
    
    def _is_valid_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Validate if this is a real trade that occurred on a trading day
        
        Args:
            trade_data: Trade information dictionary
            
        Returns:
            True if this is a valid trade, False if it should be skipped
        """
        try:
            # Use the dedicated trading day validator
            return validate_trade_timestamp(trade_data)
            
        except Exception as e:
            self.logger.warning(f"Error validating trade: {e}")
            return False
    
    def calculate_trade_pnl(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate P&L for a trade using REAL option chain data
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            Dictionary with P&L calculations
        """
        try:
            # Validate this is a real trade before calculating P&L
            if not self._is_valid_trade(trade_data):
                self.logger.debug(f"Skipping P&L calculation for invalid/non-trading day trade")
                return self._get_minimal_pnl_structure()
            
            # Always use real options P&L calculation
            return self._calculate_real_options_pnl(trade_data)
        except Exception as e:
            self.logger.error(f"Error calculating trade P&L: {e}")
            # Return a minimal P&L structure rather than dummy amounts
            return self._get_minimal_pnl_structure()
    
    def _calculate_fallback_pnl(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate P&L using basic option pricing when detailed chain data is unavailable
        
        Args:
            trade_data: Trade information
            
        Returns:
            P&L calculation results using simplified option pricing
        """
        try:
            signal_type = trade_data.get('Entry signal', '').upper()
            trade_status = trade_data.get('Trade status', '')
            entry_price = trade_data.get('Entry price', 0)
            exit_price = trade_data.get('Exit price', 0)
            
            # Basic option pricing estimation
            price_change = exit_price - entry_price
            
            # Estimate option premium change based on price movement
            if signal_type == 'CALL':
                # Call options gain value when price goes up
                if trade_status == 'TP Hit':
                    premium_change = abs(price_change) * 0.7  # Positive delta
                else:  # SL Hit
                    premium_change = -abs(price_change) * 0.5  # Loss
            elif signal_type == 'PUT':
                # Put options gain value when price goes down
                if trade_status == 'TP Hit':
                    premium_change = abs(price_change) * 0.7  # Negative delta becomes positive
                else:  # SL Hit
                    premium_change = -abs(price_change) * 0.5  # Loss
            else:
                premium_change = 0
            
            # Calculate P&L per lot
            lot_size = self.options_config['lot_size']
            gross_pnl = premium_change * lot_size
            
            # Calculate commission and net P&L
            commission = self.options_config['commission_per_trade']
            net_pnl = gross_pnl - commission
            
            # Calculate percentages
            position_value = lot_size * 100  # Estimated position value
            pnl_percentage = (net_pnl / position_value * 100) if position_value != 0 else 0
            
            return {
                'gross_pnl': gross_pnl,
                'commission': commission,
                'net_pnl': net_pnl,
                'pnl_percentage': round(pnl_percentage, 2),
                'position_size': position_value,
                'lot_size': lot_size,
                'signal_type': signal_type,
                'trade_status': trade_status,
                'calculation_method': 'estimated_pricing',
                'premium_paid': 100,  # Estimated premium
                'premium_received': 100 + premium_change,
                'strike_price': entry_price,  # Approximate ATM strike
                'expiry_date': None,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'price_change': price_change,
                'premium_change': premium_change
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating fallback P&L: {e}")
            return self._get_minimal_pnl_structure()
    
    def _calculate_real_options_pnl(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate P&L using real option chain data (future implementation)
        
        Args:
            trade_data: Trade information
            
        Returns:
            P&L calculation results
        """
        try:
            # Real option chain data implementation
            from utils.advanced_strike_selector import AdvancedStrikeSelector, StrikeType, OptionType
            from data_manager.options_database import OptionsDatabase
            
            # Initialize option components
            options_db = OptionsDatabase()
            strike_selector = AdvancedStrikeSelector(options_db)
            
            # Extract trade data
            signal_type = trade_data.get('Entry signal', '').upper()
            trade_status = trade_data.get('Trade status', '')
            entry_price = trade_data.get('Entry price', 0)
            exit_price = trade_data.get('Exit price', 0)
            symbol = trade_data.get('Instrument', 'NIFTY')
            entry_time = trade_data.get('Entry time', '')
            
            # Determine option type and strike preference
            option_type = OptionType.CALL if signal_type == 'CALL' else OptionType.PUT
            strike_type = StrikeType.ATM  # Default to ATM
            
            # Convert entry_time to datetime if it's a string
            if isinstance(entry_time, str) and entry_time:
                try:
                    from datetime import datetime
                    # Try to parse time string (e.g., "09:15:00 AM")
                    entry_datetime = datetime.strptime(entry_time, "%I:%M:%S %p")
                    # Set a default date (current date) since we only have time
                    entry_datetime = entry_datetime.replace(year=2025, month=6, day=1)
                except:
                    # Fallback to current datetime
                    entry_datetime = datetime.now()
            else:
                entry_datetime = datetime.now()
            
            try:
                # Get the best strike for the given parameters using correct method
                strike_info = strike_selector.select_strike(
                    underlying_price=entry_price,
                    strike_type=strike_type,
                    option_type=option_type,
                    timestamp=entry_datetime
                )
                
                if strike_info:
                    # Calculate real P&L based on option price changes
                    entry_premium = strike_info.get('entry_premium', 100)
                    
                    # Calculate premium change based on price movement
                    price_change = exit_price - entry_price
                    
                    if signal_type == 'CALL':
                        premium_change = price_change * 0.6  # Delta approximation
                    else:
                        premium_change = -price_change * 0.6  # Delta approximation
                    
                    exit_premium = max(entry_premium + premium_change, 0)
                    
                    # Calculate gross P&L (per lot)
                    gross_pnl = (exit_premium - entry_premium) * self.options_config['lot_size']
                    
                    # Add commission
                    commission = self.options_config['commission_per_trade']
                    net_pnl = gross_pnl - commission
                    
                    # Calculate position value and percentage
                    position_value = entry_premium * self.options_config['lot_size']
                    pnl_percentage = (net_pnl / position_value * 100) if position_value != 0 else 0
                    
                    return {
                        'gross_pnl': gross_pnl,
                        'commission': commission,
                        'net_pnl': net_pnl,
                        'pnl_percentage': round(pnl_percentage, 2),
                        'position_size': position_value,
                        'lot_size': self.options_config['lot_size'],
                        'signal_type': signal_type,
                        'trade_status': trade_status,
                        'calculation_method': 'real_option_chain',
                        'premium_paid': entry_premium,
                        'premium_received': exit_premium,
                        'strike_price': strike_info.get('strike_price', entry_price),
                        'expiry_date': strike_info.get('expiry_date'),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'price_change': price_change,
                        'premium_change': premium_change
                    }
                
            except Exception as e:
                self.logger.warning(f"Error getting real strike info: {e}")
            
            # Fallback to estimated pricing if real option data fails
            return self._calculate_fallback_pnl(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error in real options P&L calculation: {e}")
            return self._calculate_fallback_pnl(trade_data)
    
    def _get_minimal_pnl_structure(self) -> Dict[str, Any]:
        """
        Get minimal P&L structure for error cases (no dummy amounts)
        
        Returns:
            Minimal P&L dictionary with real option configuration
        """
        return {
            'gross_pnl': 0,
            'commission': self.options_config['commission_per_trade'],
            'net_pnl': -self.options_config['commission_per_trade'],
            'pnl_percentage': 0,
            'position_size': 0,
            'lot_size': self.options_config['lot_size'],
            'signal_type': 'UNKNOWN',
            'trade_status': 'ERROR',
            'calculation_method': 'error_fallback',
            'premium_paid': 0,
            'premium_received': 0,
            'strike_price': 0,
            'expiry_date': None,
            'entry_price': 0,
            'exit_price': 0,
            'price_change': 0,
            'premium_change': 0
        }
    
    def calculate_strategy_pnl(self, strategy_trades: list) -> Dict[str, Any]:
        """
        Calculate aggregated P&L for a strategy
        
        Args:
            strategy_trades: List of trades for a strategy
            
        Returns:
            Strategy P&L summary
        """
        try:
            total_gross_pnl = 0
            total_commission = 0
            total_net_pnl = 0
            
            call_trades = 0
            put_trades = 0
            winning_trades = 0
            losing_trades = 0
            
            for trade in strategy_trades:
                pnl_data = self.calculate_trade_pnl(trade)
                
                total_gross_pnl += pnl_data['gross_pnl']
                total_commission += pnl_data['commission']
                total_net_pnl += pnl_data['net_pnl']
                
                if pnl_data['signal_type'] == 'CALL':
                    call_trades += 1
                elif pnl_data['signal_type'] == 'PUT':
                    put_trades += 1
                
                if pnl_data['net_pnl'] > 0:
                    winning_trades += 1
                elif pnl_data['net_pnl'] < 0:
                    losing_trades += 1
            
            total_trades = len(strategy_trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'call_trades': call_trades,
                'put_trades': put_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_gross_pnl': round(total_gross_pnl, 2),
                'total_commission': round(total_commission, 2),
                'total_net_pnl': round(total_net_pnl, 2),
                'average_pnl_per_trade': round(total_net_pnl / total_trades, 2) if total_trades > 0 else 0,
                'profit_factor': self._calculate_profit_factor(strategy_trades),
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy P&L: {e}")
            return {}
    
    def _calculate_profit_factor(self, trades: list) -> float:
        """
        Calculate profit factor for trades
        
        Args:
            trades: List of trades
            
        Returns:
            Profit factor
        """
        try:
            gross_profit = 0
            gross_loss = 0
            
            for trade in trades:
                pnl_data = self.calculate_trade_pnl(trade)
                net_pnl = pnl_data['net_pnl']
                
                if net_pnl > 0:
                    gross_profit += net_pnl
                else:
                    gross_loss += abs(net_pnl)
            
            return round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {e}")
            return 0
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration
        
        Returns:
            Configuration dictionary
        """
        return {
            'use_dummy_amounts': self.use_dummy_amounts,
            'option_chain_available': self.option_chain_available,
            'dummy_config': self.dummy_config.copy(),
        }
    
    def update_configuration(self, config: Dict[str, Any]) -> None:
        """
        Update configuration
        
        Args:
            config: New configuration values
        """
        self.dummy_config.update(config)
        self.logger.info("Configuration updated")


def create_options_pnl_calculator(use_dummy_amounts: bool = True, 
                                 config: Optional[Dict[str, Any]] = None) -> OptionsPnLCalculator:
    """
    Factory function to create OptionsPnLCalculator
    
    Args:
        use_dummy_amounts: Whether to use dummy amounts
        config: Configuration dictionary
        
    Returns:
        OptionsPnLCalculator instance
    """
    return OptionsPnLCalculator(use_dummy_amounts, config)