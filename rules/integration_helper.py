"""
Integration helper for the vAlgo Rule Engine

Provides backward compatibility and easy integration with existing components.
Allows existing code to gradually migrate to the rule engine without breaking changes.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime

from .engine import RuleEngine
from .models import RuleResult, IndicatorData


class RuleEngineIntegrationHelper:
    """
    Helper class to integrate rule engine with existing vAlgo components
    
    Provides backward-compatible methods that existing code can use
    while gradually migrating to the new rule engine.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize integration helper
        
        Args:
            config_path: Path to Excel config file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.rule_engine = RuleEngine(config_path)
        
        # Cache for backward compatibility methods
        self._entry_conditions_cache = None
        self._exit_conditions_cache = None
        self._strategy_configs_cache = None
    
    def get_entry_conditions_enhanced(self) -> Dict[str, Any]:
        """
        Backward compatibility method for existing ConfigLoader.get_entry_conditions_enhanced()
        
        Returns:
            Dictionary compatible with existing code
        """
        if self._entry_conditions_cache is None:
            available_rules = self.rule_engine.get_available_rules()
            entry_rules = {}
            
            for rule_name in available_rules['entry']:
                # Get rule expression and required indicators
                expression = self.rule_engine.get_rule_expression(rule_name, 'entry')
                indicators = self.rule_engine.get_required_indicators(rule_name, 'entry')
                
                entry_rules[rule_name] = {
                    'status': 'Active',  # Assume active if loaded
                    'expression': expression,
                    'required_indicators': indicators,
                    'total_conditions': len(indicators)  # Simplified
                }
            
            self._entry_conditions_cache = entry_rules
        
        return self._entry_conditions_cache
    
    def get_exit_conditions_enhanced(self) -> Dict[str, Any]:
        """
        Backward compatibility method for existing ConfigLoader.get_exit_conditions_enhanced()
        
        Returns:
            Dictionary compatible with existing code
        """
        if self._exit_conditions_cache is None:
            available_rules = self.rule_engine.get_available_rules()
            exit_rules = {}
            
            for rule_name in available_rules['exit']:
                # Get rule expression and required indicators
                expression = self.rule_engine.get_rule_expression(rule_name, 'exit')
                indicators = self.rule_engine.get_required_indicators(rule_name, 'exit')
                
                exit_rules[rule_name] = {
                    'status': 'Active',  # Assume active if loaded
                    'expression': expression,
                    'required_indicators': indicators,
                    'total_conditions': len(indicators)  # Simplified
                }
            
            self._exit_conditions_cache = exit_rules
        
        return self._exit_conditions_cache
    
    def get_strategy_configs_enhanced(self) -> Dict[str, Any]:
        """
        Backward compatibility method for existing ConfigLoader.get_strategy_configs_enhanced()
        
        Returns:
            Dictionary compatible with existing code
        """
        if self._strategy_configs_cache is None:
            strategies = self.rule_engine.get_available_strategies()
            strategy_configs = {}
            
            for strategy_name in strategies:
                config = self.rule_engine.get_strategy_config(strategy_name)
                if config:
                    strategy_configs[strategy_name] = config
            
            self._strategy_configs_cache = strategy_configs
        
        return self._strategy_configs_cache
    
    def evaluate_entry_rule(self, rule_name: str, rule_config: Dict[str, Any], 
                           market_data: pd.Series, execution_data: Optional[pd.Series] = None,
                           trade_record: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Backward compatibility method for existing entry rule evaluation
        
        Args:
            rule_name: Name of the rule
            rule_config: Rule configuration (not used in new engine)
            market_data: Market data as pandas Series
            execution_data: Execution data (optional)
            trade_record: Trade record (optional)
            
        Returns:
            Tuple of (rule_triggered: bool, matched_conditions: List[str])
        """
        try:
            # Evaluate using rule engine
            result = self.rule_engine.evaluate_entry_rule(rule_name, market_data)
            
            return result.triggered, result.matched_conditions
            
        except Exception as e:
            self.logger.error(f"Error evaluating entry rule '{rule_name}': {e}")
            return False, []
    
    def evaluate_exit_rule(self, rule_name: str, rule_config: Dict[str, Any],
                          market_data: pd.Series, execution_data: Optional[pd.Series] = None,
                          trade_record: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Backward compatibility method for existing exit rule evaluation
        
        Args:
            rule_name: Name of the rule
            rule_config: Rule configuration (not used in new engine)
            market_data: Market data as pandas Series
            execution_data: Execution data (optional)
            trade_record: Trade record with SL/TP data (optional)
            
        Returns:
            Tuple of (rule_triggered: bool, matched_conditions: List[str])
        """
        try:
            # Prepare trade data for exit evaluation
            trade_data = None
            if trade_record:
                trade_data = {
                    'sl': trade_record.get('sl'),
                    'tp': trade_record.get('tp'),
                    'entry_price': trade_record.get('entry_price')
                }
            
            # Evaluate using rule engine
            result = self.rule_engine.evaluate_exit_rule(rule_name, market_data, trade_data)
            
            return result.triggered, result.matched_conditions
            
        except Exception as e:
            self.logger.error(f"Error evaluating exit rule '{rule_name}': {e}")
            return False, []
    
    def evaluate_condition(self, indicator: str, operator: str, value: Any,
                          market_data: pd.Series) -> bool:
        """
        Backward compatibility method for single condition evaluation
        
        Args:
            indicator: Indicator name
            operator: Comparison operator
            value: Comparison value
            market_data: Market data as pandas Series
            
        Returns:
            Boolean result of condition evaluation
        """
        try:
            # Get indicator value from market data
            indicator_value = market_data.get(indicator)
            
            # Resolve value if it's another indicator
            if isinstance(value, str) and value in market_data:
                comparison_value = market_data.get(value)
            else:
                comparison_value = value
            
            # Use operator resolver
            from .operator_resolver import evaluate_condition
            return evaluate_condition(indicator_value, operator, comparison_value)
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{indicator} {operator} {value}': {e}")
            return False
    
    def get_rule_engine(self) -> RuleEngine:
        """
        Get the underlying rule engine instance
        
        Returns:
            RuleEngine instance for direct access
        """
        return self.rule_engine
    
    def validate_integration(self) -> List[str]:
        """
        Validate the integration and return any issues
        
        Returns:
            List of validation issues
        """
        issues = []
        
        try:
            # Check rule engine initialization
            info = self.rule_engine.get_engine_info()
            if info['entry_rules_count'] == 0:
                issues.append("No entry rules loaded")
            if info['exit_rules_count'] == 0:
                issues.append("No exit rules loaded")
            if info['strategies_count'] == 0:
                issues.append("No strategies loaded")
            
            # Validate rule engine configuration
            validation_results = self.rule_engine.validate_all_rules()
            for rule_name, rule_issues in validation_results.items():
                for issue in rule_issues:
                    issues.append(f"Rule '{rule_name}': {issue}")
        
        except Exception as e:
            issues.append(f"Error during validation: {e}")
        
        return issues
    
    def reload_rules(self) -> bool:
        """
        Reload rules from configuration
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear caches
            self._entry_conditions_cache = None
            self._exit_conditions_cache = None
            self._strategy_configs_cache = None
            
            # Reload rule engine
            return self.rule_engine.reload_rules()
            
        except Exception as e:
            self.logger.error(f"Error reloading rules: {e}")
            return False


# Convenience functions for easy integration

def create_integration_helper(config_path: Optional[str] = None) -> RuleEngineIntegrationHelper:
    """
    Create a RuleEngineIntegrationHelper instance
    
    Args:
        config_path: Path to Excel config file (optional)
        
    Returns:
        RuleEngineIntegrationHelper instance
    """
    return RuleEngineIntegrationHelper(config_path)


def evaluate_single_condition(indicator: str, operator: str, value: Any, 
                             data: Union[pd.Series, Dict[str, Any]]) -> bool:
    """
    Evaluate a single condition against data
    
    Args:
        indicator: Indicator name
        operator: Comparison operator  
        value: Comparison value
        data: Data as pandas Series or dictionary
        
    Returns:
        Boolean result of condition evaluation
    """
    try:
        from .operator_resolver import evaluate_condition
        
        # Get indicator value
        if isinstance(data, pd.Series):
            indicator_value = data.get(indicator)
        elif isinstance(data, dict):
            indicator_value = data.get(indicator)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Resolve comparison value if it's another indicator
        if isinstance(value, str):
            if isinstance(data, pd.Series) and value in data:
                comparison_value = data.get(value)
            elif isinstance(data, dict) and value in data:
                comparison_value = data.get(value)
            else:
                comparison_value = value
        else:
            comparison_value = value
        
        return evaluate_condition(indicator_value, operator, comparison_value)
        
    except Exception:
        return False