"""
Options Trading Configuration for vAlgo Trading System
Configuration settings for options trading and P&L calculations
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class OptionsConfig:
    """
    Configuration class for options trading parameters
    """
    
    def __init__(self):
        """Initialize options configuration with default values"""
        
        # P&L Configuration
        self.pnl_config = {
            # Dummy P&L amounts (used when option chain data is not available)
            'dummy_amounts': {
                'call_win_amount': 500,         # CALL profit when TP hit
                'call_loss_amount': -200,       # CALL loss when SL hit
                'put_win_amount': 450,          # PUT profit when TP hit
                'put_loss_amount': -180,        # PUT loss when SL hit
                'position_size': 1000,          # Default position size
                'commission_per_trade': 20,     # Commission per trade
                'lot_size': 75,                 # Options lot size (NIFTY)
                'premium_multiplier': 1.0,      # Premium multiplier
            },
            
            # Option chain configuration (for future use)
            'option_chain': {
                'use_real_data': False,         # Switch to True when option chain available
                'database_path': 'data/option_chain.db',
                'supported_symbols': ['NIFTY', 'BANKNIFTY', 'FINNIFTY'],
                'default_expiry_strategy': 'nearest',  # 'nearest', 'weekly', 'monthly'
                'default_strike_strategy': 'ATM',      # 'ATM', 'ITM', 'OTM'
                'strike_interval': {
                    'NIFTY': 50,
                    'BANKNIFTY': 100,
                    'FINNIFTY': 50,
                },
            },
            
            # Risk management
            'risk_management': {
                'max_position_size': 5000,      # Maximum position size
                'max_lots_per_trade': 10,       # Maximum lots per trade
                'max_daily_loss': 2000,         # Maximum daily loss
                'position_size_percentage': 2.0, # Position size as % of capital
            },
        }
        
        # Trading configuration
        self.trading_config = {
            'trading_mode': 'options',          # 'options', 'equity', 'futures'
            'enable_options_trading': True,     # Enable options trading
            'enable_multi_leg_strategies': False, # Multi-leg strategies (future)
            'enable_hedging': False,            # Enable hedging (future)
            'auto_roll_positions': False,       # Auto roll expiring positions (future)
        }
        
        # Reporting configuration
        self.reporting_config = {
            'include_options_metrics': True,   # Include options-specific metrics
            'include_greeks_analysis': False,  # Include Greeks analysis (future)
            'include_volatility_analysis': False, # Volatility analysis (future)
            'generate_option_chain_reports': False, # Option chain reports (future)
            'export_trade_details': True,      # Export detailed trade information
        }
        
        # Performance tracking
        self.performance_config = {
            'track_individual_legs': False,    # Track individual option legs (future)
            'calculate_time_decay': False,     # Calculate time decay impact (future)
            'calculate_volatility_impact': False, # Volatility impact (future)
            'benchmark_comparison': False,     # Compare with benchmark (future)
        }
    
    def get_dummy_pnl_config(self) -> Dict[str, Any]:
        """
        Get dummy P&L configuration
        
        Returns:
            Dummy P&L configuration dictionary
        """
        return self.pnl_config['dummy_amounts'].copy()
    
    def get_option_chain_config(self) -> Dict[str, Any]:
        """
        Get option chain configuration
        
        Returns:
            Option chain configuration dictionary
        """
        return self.pnl_config['option_chain'].copy()
    
    def get_risk_management_config(self) -> Dict[str, Any]:
        """
        Get risk management configuration
        
        Returns:
            Risk management configuration dictionary
        """
        return self.pnl_config['risk_management'].copy()
    
    def get_trading_config(self) -> Dict[str, Any]:
        """
        Get trading configuration
        
        Returns:
            Trading configuration dictionary
        """
        return self.trading_config.copy()
    
    def get_reporting_config(self) -> Dict[str, Any]:
        """
        Get reporting configuration
        
        Returns:
            Reporting configuration dictionary
        """
        return self.reporting_config.copy()
    
    def update_dummy_pnl_amounts(self, amounts: Dict[str, Any]) -> None:
        """
        Update dummy P&L amounts
        
        Args:
            amounts: Dictionary with new P&L amounts
        """
        self.pnl_config['dummy_amounts'].update(amounts)
    
    def enable_option_chain_data(self, database_path: Optional[str] = None) -> None:
        """
        Enable option chain data usage
        
        Args:
            database_path: Path to option chain database
        """
        self.pnl_config['option_chain']['use_real_data'] = True
        if database_path:
            self.pnl_config['option_chain']['database_path'] = database_path
    
    def disable_option_chain_data(self) -> None:
        """Disable option chain data usage (use dummy amounts)"""
        self.pnl_config['option_chain']['use_real_data'] = False
    
    def is_option_chain_enabled(self) -> bool:
        """
        Check if option chain data is enabled
        
        Returns:
            True if option chain data is enabled
        """
        return self.pnl_config['option_chain']['use_real_data']
    
    def get_lot_size(self, symbol: str) -> int:
        """
        Get lot size for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Lot size for the symbol
        """
        lot_sizes = {
            'NIFTY': 75,
            'BANKNIFTY': 15,
            'FINNIFTY': 40,
        }
        return lot_sizes.get(symbol, 75)  # Default to NIFTY lot size
    
    def get_strike_interval(self, symbol: str) -> int:
        """
        Get strike interval for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Strike interval for the symbol
        """
        intervals = self.pnl_config['option_chain']['strike_interval']
        return intervals.get(symbol, 50)  # Default to NIFTY interval
    
    def get_commission_structure(self, symbol: str) -> Dict[str, float]:
        """
        Get commission structure for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Commission structure
        """
        # Standard commission structure (can be customized per symbol)
        return {
            'per_trade': 20,
            'per_lot': 0,
            'percentage': 0.05,
            'minimum': 20,
            'maximum': 100,
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration settings
        
        Returns:
            Validation results
        """
        issues = []
        warnings = []
        
        # Validate dummy P&L amounts
        dummy_amounts = self.pnl_config['dummy_amounts']
        if dummy_amounts['call_win_amount'] <= 0:
            issues.append("CALL win amount must be positive")
        if dummy_amounts['call_loss_amount'] >= 0:
            issues.append("CALL loss amount must be negative")
        if dummy_amounts['put_win_amount'] <= 0:
            issues.append("PUT win amount must be positive")
        if dummy_amounts['put_loss_amount'] >= 0:
            issues.append("PUT loss amount must be negative")
        
        # Validate position sizes
        if dummy_amounts['position_size'] <= 0:
            issues.append("Position size must be positive")
        if dummy_amounts['lot_size'] <= 0:
            issues.append("Lot size must be positive")
        
        # Validate risk management
        risk_config = self.pnl_config['risk_management']
        if risk_config['max_position_size'] <= 0:
            issues.append("Maximum position size must be positive")
        if risk_config['max_lots_per_trade'] <= 0:
            issues.append("Maximum lots per trade must be positive")
        
        # Warnings for option chain
        if not self.is_option_chain_enabled():
            warnings.append("Option chain data is disabled - using dummy amounts")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
        }
    
    def get_complete_config(self) -> Dict[str, Any]:
        """
        Get complete configuration
        
        Returns:
            Complete configuration dictionary
        """
        return {
            'pnl_config': self.pnl_config,
            'trading_config': self.trading_config,
            'reporting_config': self.reporting_config,
            'performance_config': self.performance_config,
            'validation': self.validate_configuration(),
        }


# Default configuration instance
default_options_config = OptionsConfig()


def get_options_config() -> OptionsConfig:
    """
    Get default options configuration
    
    Returns:
        OptionsConfig instance
    """
    return default_options_config


def create_custom_options_config(custom_config: Dict[str, Any]) -> OptionsConfig:
    """
    Create custom options configuration
    
    Args:
        custom_config: Custom configuration values
        
    Returns:
        OptionsConfig instance with custom values
    """
    config = OptionsConfig()
    
    # Update with custom values
    if 'dummy_amounts' in custom_config:
        config.update_dummy_pnl_amounts(custom_config['dummy_amounts'])
    
    if 'enable_option_chain' in custom_config:
        if custom_config['enable_option_chain']:
            config.enable_option_chain_data(custom_config.get('database_path'))
        else:
            config.disable_option_chain_data()
    
    return config