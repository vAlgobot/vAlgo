"""
Main Rule Engine for the vAlgo Trading System

Provides a centralized, testable rule evaluation engine that can be used
independently of the main trading system for condition evaluation.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .models import Rule, RuleResult, IndicatorData
from .condition_loader import ConditionLoader
from .rule_parser import RuleParser
from .operator_resolver import OperatorResolver


class RuleEngine:
    """
    Main rule evaluation engine for vAlgo trading strategies
    
    Provides isolated, testable rule evaluation that can work with:
    - CSV indicator data for testing
    - Live indicator data for production
    - Complex multi-condition rules with logic operators
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                 operator_resolver: Optional[OperatorResolver] = None):
        """
        Initialize the rule engine
        
        Args:
            config_path: Path to Excel config file (optional)
            operator_resolver: Custom operator resolver (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.condition_loader = ConditionLoader(config_path)
        self.operator_resolver = operator_resolver or OperatorResolver()
        self.rule_parser = RuleParser(self.operator_resolver)
        
        # Rule caches
        self.entry_rules: Dict[str, Rule] = {}
        self.exit_rules: Dict[str, Rule] = {}
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}
        
        # Load rules on initialization
        self._load_all_rules()
        
        self.logger.info(f"RuleEngine initialized with {len(self.entry_rules)} entry rules, "
                        f"{len(self.exit_rules)} exit rules")
    
    def _load_all_rules(self) -> None:
        """Load all rules from configuration"""
        try:
            self.entry_rules = self.condition_loader.load_entry_conditions()
            self.exit_rules = self.condition_loader.load_exit_conditions()
            self.strategy_configs = self.condition_loader.load_strategy_configs()
            
            self.logger.info("All rules loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading rules: {e}")
            # Initialize empty dictionaries to prevent errors
            self.entry_rules = {}
            self.exit_rules = {}
            self.strategy_configs = {}
    
    def reload_rules(self) -> bool:
        """
        Reload rules from configuration file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Force reload the configuration
            if not self.condition_loader.load_config(force_reload=True):
                return False
            
            # Reload all rules
            self._load_all_rules()
            return True
            
        except Exception as e:
            self.logger.error(f"Error reloading rules: {e}")
            return False
    
    def evaluate_entry_rule(self, rule_name: str, indicator_data: Union[pd.Series, Dict[str, Any], IndicatorData], 
                           timestamp: Optional[datetime] = None) -> RuleResult:
        """
        Evaluate an entry rule against indicator data
        
        Args:
            rule_name: Name of the entry rule to evaluate
            indicator_data: Current indicator values (pandas Series, dict, or IndicatorData)
            timestamp: Timestamp for evaluation (optional)
            
        Returns:
            RuleResult with evaluation details
        """
        # Convert indicator data to IndicatorData object
        if isinstance(indicator_data, pd.Series):
            indicator_obj = IndicatorData.from_pandas_series(indicator_data, timestamp)
        elif isinstance(indicator_data, dict):
            indicator_obj = IndicatorData.from_dict(indicator_data, timestamp)
        elif isinstance(indicator_data, IndicatorData):
            indicator_obj = indicator_data
        else:
            raise ValueError(f"Unsupported indicator data type: {type(indicator_data)}")
        
        # Check if rule exists
        if rule_name not in self.entry_rules:
            result = RuleResult(rule_name=rule_name, triggered=False, timestamp=timestamp)
            result.error_message = f"Entry rule '{rule_name}' not found"
            available_rules = list(self.entry_rules.keys())
            result.evaluation_details['available_rules'] = available_rules
            self.logger.warning(f"Entry rule '{rule_name}' not found. Available: {available_rules}")
            return result
        
        # Get rule and evaluate
        rule = self.entry_rules[rule_name]
        result = self.rule_parser.evaluate_rule(rule, indicator_obj)
        result.timestamp = timestamp or datetime.now()
        
        self.logger.debug(f"Entry rule '{rule_name}' evaluation: {result.triggered}")
        return result
    
    def evaluate_exit_rule(self, rule_name: str, indicator_data: Union[pd.Series, Dict[str, Any], IndicatorData],
                          trade_data: Optional[Dict[str, Any]] = None,
                          timestamp: Optional[datetime] = None) -> RuleResult:
        """
        Evaluate an exit rule against indicator data and trade data
        
        Args:
            rule_name: Name of the exit rule to evaluate
            indicator_data: Current indicator values
            trade_data: Trade data (SL, TP, entry price, etc.)
            timestamp: Timestamp for evaluation (optional)
            
        Returns:
            RuleResult with evaluation details
        """
        # Convert indicator data to IndicatorData object
        if isinstance(indicator_data, pd.Series):
            indicator_obj = IndicatorData.from_pandas_series(indicator_data, timestamp)
        elif isinstance(indicator_data, dict):
            indicator_obj = IndicatorData.from_dict(indicator_data, timestamp)
        elif isinstance(indicator_data, IndicatorData):
            indicator_obj = indicator_data
        else:
            raise ValueError(f"Unsupported indicator data type: {type(indicator_data)}")
        
        # Check if rule exists
        if rule_name not in self.exit_rules:
            result = RuleResult(rule_name=rule_name, triggered=False, timestamp=timestamp)
            result.error_message = f"Exit rule '{rule_name}' not found"
            available_rules = list(self.exit_rules.keys())
            result.evaluation_details['available_rules'] = available_rules
            self.logger.warning(f"Exit rule '{rule_name}' not found. Available: {available_rules}")
            return result
        
        # Get rule and evaluate
        rule = self.exit_rules[rule_name]
        result = self.rule_parser.evaluate_rule(rule, indicator_obj, trade_data)
        result.timestamp = timestamp or datetime.now()
        
        self.logger.debug(f"Exit rule '{rule_name}' evaluation: {result.triggered}")
        return result
    
    def evaluate_strategy_entry(self, strategy_name: str, indicator_data: Union[pd.Series, Dict[str, Any], IndicatorData],
                               timestamp: Optional[datetime] = None) -> List[RuleResult]:
        """
        Evaluate all entry rules for a strategy
        
        Args:
            strategy_name: Name of the strategy
            indicator_data: Current indicator values
            timestamp: Timestamp for evaluation (optional)
            
        Returns:
            List of RuleResult objects for each entry rule
        """
        results = []
        
        if strategy_name not in self.strategy_configs:
            self.logger.warning(f"Strategy '{strategy_name}' not found")
            return results
        
        strategy = self.strategy_configs[strategy_name]
        entry_rules = strategy.get('entry_rules', [])
        
        for rule_name in entry_rules:
            result = self.evaluate_entry_rule(rule_name, indicator_data, timestamp)
            results.append(result)
        
        return results
    
    def evaluate_strategy_exit(self, strategy_name: str, indicator_data: Union[pd.Series, Dict[str, Any], IndicatorData],
                              trade_data: Optional[Dict[str, Any]] = None,
                              timestamp: Optional[datetime] = None) -> List[RuleResult]:
        """
        Evaluate all exit rules for a strategy
        
        Args:
            strategy_name: Name of the strategy
            indicator_data: Current indicator values
            trade_data: Trade data (SL, TP, entry price, etc.)
            timestamp: Timestamp for evaluation (optional)
            
        Returns:
            List of RuleResult objects for each exit rule
        """
        results = []
        
        if strategy_name not in self.strategy_configs:
            self.logger.warning(f"Strategy '{strategy_name}' not found")
            return results
        
        strategy = self.strategy_configs[strategy_name]
        exit_rules = strategy.get('exit_rules', [])
        
        for rule_name in exit_rules:
            result = self.evaluate_exit_rule(rule_name, indicator_data, trade_data, timestamp)
            results.append(result)
        
        return results
    
    def get_rule_expression(self, rule_name: str, rule_type: str = "entry") -> str:
        """
        Get human-readable expression for a rule
        
        Args:
            rule_name: Name of the rule
            rule_type: "entry" or "exit"
            
        Returns:
            Human-readable rule expression
        """
        if rule_type == "entry":
            rule = self.entry_rules.get(rule_name)
        else:
            rule = self.exit_rules.get(rule_name)
        
        if not rule:
            return f"Rule '{rule_name}' not found"
        
        return self.rule_parser.build_rule_expression(rule)
    
    def get_required_indicators(self, rule_name: str, rule_type: str = "entry") -> List[str]:
        """
        Get indicators required by a rule
        
        Args:
            rule_name: Name of the rule
            rule_type: "entry" or "exit"
            
        Returns:
            List of required indicator names
        """
        if rule_type == "entry":
            rule = self.entry_rules.get(rule_name)
        else:
            rule = self.exit_rules.get(rule_name)
        
        if not rule:
            return []
        
        return self.rule_parser.get_required_indicators(rule)
    
    def validate_indicator_data(self, indicator_data: Union[pd.Series, Dict[str, Any]], 
                               rule_name: str, rule_type: str = "entry") -> List[str]:
        """
        Validate that indicator data contains required indicators for a rule
        
        Args:
            indicator_data: Indicator data to validate
            rule_name: Name of the rule
            rule_type: "entry" or "exit"
            
        Returns:
            List of validation issues
        """
        # Get available indicators
        if isinstance(indicator_data, pd.Series):
            available_indicators = list(indicator_data.index)
        elif isinstance(indicator_data, dict):
            available_indicators = list(indicator_data.keys())
        else:
            return [f"Unsupported indicator data type: {type(indicator_data)}"]
        
        # Get rule
        if rule_type == "entry":
            rule = self.entry_rules.get(rule_name)
        else:
            rule = self.exit_rules.get(rule_name)
        
        if not rule:
            return [f"Rule '{rule_name}' not found"]
        
        # Validate rule against available indicators
        return self.rule_parser.validate_rule(rule, available_indicators)
    
    def get_available_rules(self) -> Dict[str, List[str]]:
        """
        Get all available rules by type
        
        Returns:
            Dictionary with 'entry' and 'exit' rule lists
        """
        return {
            'entry': list(self.entry_rules.keys()),
            'exit': list(self.exit_rules.keys())
        }
    
    def get_available_strategies(self) -> List[str]:
        """
        Get all available strategy names
        
        Returns:
            List of strategy names
        """
        return list(self.strategy_configs.keys())
    
    def get_strategy_config(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy configuration or None
        """
        return self.strategy_configs.get(strategy_name)
    
    def validate_all_rules(self, available_indicators: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Validate all loaded rules
        
        Args:
            available_indicators: List of available indicators (optional)
            
        Returns:
            Dictionary mapping rule names to validation issues
        """
        validation_results = {}
        
        # If no indicators provided, use common indicator names
        if available_indicators is None:
            available_indicators = [
                'close', 'open', 'high', 'low', 'volume',
                'ema_9', 'ema_20', 'ema_50', 'ema_200',
                'sma_20', 'sma_50', 'sma_200',
                'rsi_14', 'rsi_21',
                'CPR_R1', 'CPR_R2', 'CPR_S1', 'CPR_S2', 'CPR_Pivot',
                'CurrentCandle_Close', 'CurrentCandle_High', 'CurrentCandle_Low',
                'PreviousCandle_Close', 'PreviousDayCandle_High', 'PreviousDayCandle_Low'
            ]
        
        # Validate entry rules
        for rule_name, rule in self.entry_rules.items():
            issues = self.rule_parser.validate_rule(rule, available_indicators)
            if issues:
                validation_results[f"entry_{rule_name}"] = issues
        
        # Validate exit rules
        for rule_name, rule in self.exit_rules.items():
            issues = self.rule_parser.validate_rule(rule, available_indicators)
            if issues:
                validation_results[f"exit_{rule_name}"] = issues
        
        return validation_results
    
    def process_signal_data_pipeline(self, signal_data_path: Optional[str] = None, 
                                    start_date: Optional[str] = None, 
                                    end_date: Optional[str] = None,
                                    output_path: Optional[str] = None) -> bool:
        """
        Process signal data using the complete Strategy_Config → Entry_Conditions → Exit_Conditions pipeline
        
        Args:
            signal_data_path: Path to signal data CSV (optional, can be read from config)
            start_date: Start date for filtering (optional, can be read from config)
            end_date: End date for filtering (optional, can be read from config)
            output_path: Path for output file (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from .signal_processor import SignalDataProcessor
            
            # Create signal processor
            processor = SignalDataProcessor(self.condition_loader.config_path)
            processor.set_rule_engine(self)
            
            # Override configuration if provided
            if signal_data_path or start_date or end_date:
                # Load base configuration
                if not processor.load_configuration():
                    return False
                
                # Apply overrides
                if signal_data_path:
                    processor.signal_data_path = signal_data_path
                if start_date:
                    processor.start_date = start_date
                if end_date:
                    processor.end_date = end_date
            
            # Process signal data
            if not processor.process_signal_data():
                return False
            
            # Export results if output path provided
            if output_path:
                success = processor.export_results(output_path)
                if not success:
                    self.logger.warning("Failed to export results")
            
            # Log summary statistics
            stats = processor.get_summary_statistics()
            self.logger.info(f"Signal processing completed: {stats}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in signal data pipeline: {e}")
            return False
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the rule engine state
        
        Returns:
            Dictionary with engine information
        """
        return {
            'entry_rules_count': len(self.entry_rules),
            'exit_rules_count': len(self.exit_rules),
            'strategies_count': len(self.strategy_configs),
            'entry_rules': list(self.entry_rules.keys()),
            'exit_rules': list(self.exit_rules.keys()),
            'strategies': list(self.strategy_configs.keys()),
            'supported_operators': self.operator_resolver.get_supported_operators(),
            'config_path': self.condition_loader.config_path
        }


def create_rule_engine(config_path: Optional[str] = None) -> RuleEngine:
    """
    Factory function to create a RuleEngine instance
    
    Args:
        config_path: Path to Excel config file (optional)
        
    Returns:
        Configured RuleEngine instance
    """
    return RuleEngine(config_path=config_path)