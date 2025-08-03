"""
Rule parser for the vAlgo Rule Engine

Handles parsing and evaluation of complex rule expressions with multiple conditions,
logic operators, and condition groups.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from .models import Rule, Condition, ConditionGroup, RuleResult, IndicatorData, LogicOperator
from .operator_resolver import OperatorResolver


class RuleParser:
    """
    Parses and evaluates complex rule expressions
    
    Handles multi-condition rules with AND/OR logic between condition groups
    """
    
    def __init__(self, operator_resolver: Optional[OperatorResolver] = None):
        """
        Initialize rule parser
        
        Args:
            operator_resolver: Custom operator resolver (uses default if None)
        """
        self.logger = logging.getLogger(__name__)
        self.operator_resolver = operator_resolver or OperatorResolver()
    
    def evaluate_rule(self, rule: Rule, indicator_data: IndicatorData, 
                     trade_data: Optional[Dict[str, Any]] = None) -> RuleResult:
        """
        Evaluate a complete rule against indicator data
        
        Args:
            rule: Rule to evaluate
            indicator_data: Current indicator values
            trade_data: Additional trade data (for exit conditions)
            
        Returns:
            RuleResult with evaluation details
        """
        result = RuleResult(rule_name=rule.name, triggered=False)
        
        try:
            # Check if rule is active
            if not rule.is_active():
                result.error_message = f"Rule '{rule.name}' is inactive"
                return result
            
            # Evaluate all condition groups
            group_results = []
            
            for group in rule.condition_groups:
                group_result = self._evaluate_condition_group(group, indicator_data, trade_data)
                group_results.append(group_result)
                
                # Add details to result
                if group_result['triggered']:
                    result.matched_conditions.extend(group_result['matched_conditions'])
                else:
                    result.failed_conditions.extend(group_result['failed_conditions'])
                
                # Update evaluation details
                result.evaluation_details[f"group_{group.order}"] = group_result
            
            # Combine group results using logic operators
            rule_triggered = self._combine_group_results(rule.condition_groups, group_results)
            result.triggered = rule_triggered
            
            self.logger.debug(f"Rule '{rule.name}' evaluation: {result.triggered}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule '{rule.name}': {e}")
            result.error_message = str(e)
            return result
    
    def _evaluate_condition_group(self, group: ConditionGroup, indicator_data: IndicatorData,
                                 trade_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single condition group
        
        Args:
            group: ConditionGroup to evaluate
            indicator_data: Current indicator values
            trade_data: Additional trade data
            
        Returns:
            Dictionary with group evaluation results
        """
        group_result = {
            'triggered': False,
            'matched_conditions': [],
            'failed_conditions': [],
            'condition_results': []
        }
        
        try:
            condition_results = []
            
            # Evaluate each condition in the group
            for condition in group.conditions:
                condition_result = self._evaluate_single_condition(
                    condition, indicator_data, trade_data
                )
                condition_results.append(condition_result)
                
                if condition_result['triggered']:
                    group_result['matched_conditions'].append(str(condition))
                else:
                    group_result['failed_conditions'].append(str(condition))
            
            group_result['condition_results'] = condition_results
            
            # For conditions within a group, use OR logic (any condition can trigger the group)
            group_result['triggered'] = any(cr['triggered'] for cr in condition_results)
            
            return group_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition group {group.order}: {e}")
            group_result['error'] = str(e)
            return group_result
    
    def _evaluate_single_condition(self, condition: Condition, indicator_data: IndicatorData,
                                  trade_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single condition
        
        Args:
            condition: Condition to evaluate
            indicator_data: Current indicator values
            trade_data: Additional trade data (for SL/TP values)
            
        Returns:
            Dictionary with condition evaluation result
        """
        condition_result = {
            'condition': str(condition),
            'triggered': False,
            'left_value': None,
            'right_value': None,
            'error': None
        }
        
        try:
            # Get left value (indicator value)
            left_value = self._resolve_value(condition.indicator, indicator_data, trade_data)
            condition_result['left_value'] = left_value
            
            # Get right value (comparison value)
            right_value = self._resolve_value(condition.value, indicator_data, trade_data)
            condition_result['right_value'] = right_value
            
            # Handle None values
            if left_value is None:
                condition_result['error'] = f"Indicator '{condition.indicator}' not found or is None"
                return condition_result
            
            if right_value is None:
                condition_result['error'] = f"Value '{condition.value}' could not be resolved or is None"
                return condition_result
            
            # Evaluate condition using operator resolver
            triggered = self.operator_resolver.evaluate_condition(
                left_value, condition.operator, right_value
            )
            condition_result['triggered'] = triggered
            
            self.logger.debug(f"Condition '{condition}': {left_value} {condition.operator} {right_value} = {triggered}")
            return condition_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            condition_result['error'] = str(e)
            return condition_result
    
    def _resolve_value(self, value: Union[str, float, int], indicator_data: IndicatorData,
                      trade_data: Optional[Dict[str, Any]] = None) -> Any:
        """
        Resolve a value which could be an indicator name, trade data field, or literal value
        
        Args:
            value: Value to resolve
            indicator_data: Current indicator values
            trade_data: Additional trade data
            
        Returns:
            Resolved value
        """
        if value is None:
            return None
        
        # Convert to string for processing
        value_str = str(value).strip()
        
        # Check if it's a numeric literal
        try:
            # Try integer first
            if '.' not in value_str:
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            pass
        
        # Check if it's an indicator name
        if indicator_data.has(value_str):
            return indicator_data.get(value_str)
        
        # Check if it's a trade data field (SL, TP, etc.)
        if trade_data and value_str.upper() in ['SL', 'TP', 'STOPLOSS', 'TAKEPROFIT']:
            field_map = {
                'SL': 'sl',
                'STOPLOSS': 'sl',
                'TP': 'tp',
                'TAKEPROFIT': 'tp'
            }
            field_name = field_map.get(value_str.upper(), value_str.lower())
            if field_name in trade_data:
                return trade_data[field_name]
        
        # Check for special indicator aliases
        indicator_aliases = {
            'LTP_Close': 'close',
            'LTP_High': 'high',
            'LTP_Low': 'low',
            'LTP_Open': 'open',
            'Current_Close': 'CurrentCandle_Close',
            'Current_High': 'CurrentCandle_High',
            'Current_Low': 'CurrentCandle_Low',
            'Previous_Close': 'PreviousCandle_Close',
            'Previosuday_High': 'PreviousDayCandle_High',  # Note: typo in original config
            'Previosuday_Low': 'PreviousDayCandle_Low'
        }
        
        if value_str in indicator_aliases:
            alias_indicator = indicator_aliases[value_str]
            if indicator_data.has(alias_indicator):
                return indicator_data.get(alias_indicator)
        
        # Return as literal string if nothing else matches
        return value_str
    
    def _combine_group_results(self, condition_groups: List[ConditionGroup], 
                              group_results: List[Dict[str, Any]]) -> bool:
        """
        Combine condition group results using logic operators
        
        Args:
            condition_groups: List of condition groups
            group_results: Results from each group
            
        Returns:
            Overall rule trigger status
        """
        if not group_results:
            return False
        
        if len(group_results) == 1:
            return group_results[0]['triggered']
        
        # Start with first group result
        combined_result = group_results[0]['triggered']
        
        # Apply logic operators between groups
        for i in range(len(condition_groups) - 1):
            current_group = condition_groups[i]
            next_result = group_results[i + 1]['triggered']
            
            # Determine logic operator
            if current_group.logic_to_next == LogicOperator.AND:
                combined_result = combined_result and next_result
            elif current_group.logic_to_next == LogicOperator.OR:
                combined_result = combined_result or next_result
            else:
                # Default to AND if no logic specified
                combined_result = combined_result and next_result
        
        return combined_result
    
    def build_rule_expression(self, rule: Rule) -> str:
        """
        Build a human-readable expression from a rule
        
        Args:
            rule: Rule to build expression for
            
        Returns:
            Human-readable rule expression
        """
        if not rule.condition_groups:
            return "No conditions"
        
        group_expressions = []
        
        for group in rule.condition_groups:
            if len(group.conditions) == 1:
                # Single condition
                group_expr = str(group.conditions[0])
            else:
                # Multiple conditions in group (OR logic within group)
                condition_strs = [str(cond) for cond in group.conditions]
                group_expr = "(" + " OR ".join(condition_strs) + ")"
            
            group_expressions.append(group_expr)
        
        # Join groups with their logic operators
        if len(group_expressions) == 1:
            return group_expressions[0]
        
        final_expression = group_expressions[0]
        for i in range(len(rule.condition_groups) - 1):
            logic_op = rule.condition_groups[i].logic_to_next
            if logic_op == LogicOperator.OR:
                final_expression += " OR " + group_expressions[i + 1]
            else:
                final_expression += " AND " + group_expressions[i + 1]
        
        return final_expression
    
    def get_required_indicators(self, rule: Rule) -> List[str]:
        """
        Get list of indicators required by a rule
        
        Args:
            rule: Rule to analyze
            
        Returns:
            List of required indicator names
        """
        indicators = set()
        
        for group in rule.condition_groups:
            for condition in group.conditions:
                # Add left side indicator
                indicators.add(condition.indicator)
                
                # Check if right side is also an indicator
                if isinstance(condition.value, str):
                    value_str = condition.value.strip()
                    # Common indicator patterns
                    if any(pattern in value_str.lower() for pattern in 
                          ['ema_', 'sma_', 'rsi_', 'cpr_', 'current', 'previous']):
                        indicators.add(value_str)
        
        return sorted(list(indicators))
    
    def validate_rule(self, rule: Rule, available_indicators: List[str]) -> List[str]:
        """
        Validate a rule against available indicators
        
        Args:
            rule: Rule to validate
            available_indicators: List of available indicator names
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not rule.condition_groups:
            issues.append(f"Rule '{rule.name}' has no condition groups")
            return issues
        
        required_indicators = self.get_required_indicators(rule)
        missing_indicators = [ind for ind in required_indicators if ind not in available_indicators]
        
        if missing_indicators:
            issues.append(f"Rule '{rule.name}' requires missing indicators: {missing_indicators}")
        
        # Validate operators
        for group in rule.condition_groups:
            for condition in group.conditions:
                if not self.operator_resolver.is_operator_supported(condition.operator):
                    issues.append(f"Rule '{rule.name}' uses unsupported operator: {condition.operator}")
        
        return issues