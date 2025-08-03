"""
Condition Group Evaluator for vAlgo Rule Engine

Handles AND/OR logic evaluation within rules based on condition_order.
Provides sophisticated rule evaluation with proper logical operators.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .models import Rule, Condition, ConditionGroup, RuleResult, IndicatorData, LogicOperator
from .operator_resolver import OperatorResolver
from .enhanced_condition_loader import RuleCondition


@dataclass
class ConditionEvaluationResult:
    """Result of evaluating a single condition"""
    condition: Condition
    result: bool
    indicator_value: Any = None
    comparison_value: Any = None
    error_message: str = ""


@dataclass
class GroupEvaluationResult:
    """Result of evaluating a condition group"""
    group_result: bool
    condition_results: List[ConditionEvaluationResult]
    logic_operator: LogicOperator
    evaluation_details: Dict[str, Any]


class ConditionGroupEvaluator:
    """
    Evaluates condition groups with AND/OR logic based on condition_order
    
    Features:
    - Proper AND/OR logic evaluation
    - Short-circuit evaluation for performance
    - Detailed evaluation tracking
    - Error handling and reporting
    """
    
    def __init__(self, operator_resolver: Optional[OperatorResolver] = None):
        """
        Initialize condition group evaluator
        
        Args:
            operator_resolver: Custom operator resolver (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.operator_resolver = operator_resolver or OperatorResolver()
    
    def evaluate_rule_conditions(self, rule_conditions: List[RuleCondition], 
                                indicator_data: IndicatorData,
                                rule_name: str) -> RuleResult:
        """
        Evaluate rule conditions with proper AND/OR logic
        
        Args:
            rule_conditions: List of RuleCondition objects sorted by condition_order
            indicator_data: Current indicator values
            rule_name: Name of the rule for result tracking
            
        Returns:
            RuleResult with evaluation details
        """
        try:
            if not rule_conditions:
                self.logger.debug(f"Rule '{rule_name}': No conditions to evaluate")
                return RuleResult(
                    rule_name=rule_name,
                    triggered=False,
                    error_message="No conditions to evaluate"
                )
            
            # Group conditions by logic (AND/OR groups)
            condition_groups = self._build_condition_groups(rule_conditions)
            self.logger.debug(f"Rule '{rule_name}': {len(rule_conditions)} conditions grouped into {len(condition_groups)} groups")
            
            # Evaluate each group
            group_results = []
            overall_result = True
            matched_conditions = []
            evaluation_details = {
                'total_conditions': len(rule_conditions),
                'total_groups': len(condition_groups),
                'group_evaluations': []
            }
            
            for i, group in enumerate(condition_groups):
                self.logger.debug(f"Rule '{rule_name}': Evaluating group {i+1}/{len(condition_groups)} with {len(group)} conditions")
                group_result = self._evaluate_condition_group(group, indicator_data)
                group_results.append(group_result)
                
                self.logger.debug(f"Rule '{rule_name}': Group {i+1} result: {group_result.group_result} "
                                f"({group_result.logic_operator.value} logic)")
                
                # Add to evaluation details
                evaluation_details['group_evaluations'].append({
                    'group_index': i,
                    'group_result': group_result.group_result,
                    'logic_operator': group_result.logic_operator.value,
                    'conditions_count': len(group_result.condition_results)
                })
                
                # Collect matched conditions
                for cond_result in group_result.condition_results:
                    if cond_result.result:
                        condition_desc = f"{cond_result.condition.indicator} {cond_result.condition.operator} {cond_result.condition.value}"
                        matched_conditions.append(condition_desc)
                        self.logger.debug(f"Rule '{rule_name}': Condition matched: {condition_desc}")
                    else:
                        condition_desc = f"{cond_result.condition.indicator} {cond_result.condition.operator} {cond_result.condition.value}"
                        self.logger.debug(f"Rule '{rule_name}': Condition failed: {condition_desc} "
                                        f"(value: {cond_result.indicator_value}, error: {cond_result.error_message})")
                
                # Apply AND logic between groups (all groups must be true)
                if not group_result.group_result:
                    overall_result = False
                    self.logger.debug(f"Rule '{rule_name}': Group {i+1} failed, overall result = False")
                    # Continue evaluation for complete details (no short circuit for debugging)
            
            # Create result
            result = RuleResult(
                rule_name=rule_name,
                triggered=overall_result,
                matched_conditions=matched_conditions,
                evaluation_details=evaluation_details
            )
            
            status = "TRIGGERED" if overall_result else "NOT TRIGGERED"
            self.logger.debug(f"Rule '{rule_name}' final result: {status} "
                            f"(matched {len(matched_conditions)}/{len(rule_conditions)} conditions)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule '{rule_name}': {e}")
            return RuleResult(
                rule_name=rule_name,
                triggered=False,
                error_message=f"Evaluation error: {e}"
            )
    
    def _build_condition_groups(self, rule_conditions: List[RuleCondition]) -> List[List[RuleCondition]]:
        """
        Build condition groups based on logic operators
        
        Conditions with OR logic are grouped together.
        AND logic creates separate groups.
        
        Args:
            rule_conditions: List of RuleCondition objects sorted by condition_order
            
        Returns:
            List of condition groups
        """
        if not rule_conditions:
            return []
        
        groups = []
        current_group = []
        
        for i, condition in enumerate(rule_conditions):
            current_group.append(condition)
            
            # Check if this is the end of a group
            is_last = (i == len(rule_conditions) - 1)
            next_is_or = not is_last and rule_conditions[i + 1].logic == "OR"
            current_is_and = condition.logic == "AND" or condition.logic == ""
            
            # End group if:
            # 1. This is the last condition, OR
            # 2. Current condition has AND logic (or empty) and next is not OR
            if is_last or (current_is_and and not next_is_or):
                groups.append(current_group.copy())
                current_group = []
        
        # Handle any remaining conditions
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _evaluate_condition_group(self, condition_group: List[RuleCondition], 
                                 indicator_data: IndicatorData) -> GroupEvaluationResult:
        """
        Evaluate a single condition group
        
        Args:
            condition_group: List of conditions in the group
            indicator_data: Current indicator values
            
        Returns:
            GroupEvaluationResult with evaluation details
        """
        if not condition_group:
            return GroupEvaluationResult(
                group_result=False,
                condition_results=[],
                logic_operator=LogicOperator.AND,
                evaluation_details={'error': 'Empty condition group'}
            )
        
        condition_results = []
        
        # Evaluate each condition in the group
        for rule_condition in condition_group:
            condition = Condition(
                indicator=rule_condition.indicator,
                operator=rule_condition.operator,
                value=rule_condition.value,
                exit_type=rule_condition.exit_type if hasattr(rule_condition, 'exit_type') else None
            )
            
            cond_result = self._evaluate_single_condition(condition, indicator_data)
            condition_results.append(cond_result)
        
        # Determine group logic (OR if any condition has OR logic, else AND)
        has_or_logic = any(cond.logic == "OR" for cond in condition_group)
        logic_operator = LogicOperator.OR if has_or_logic else LogicOperator.AND
        
        # Calculate group result based on logic
        if logic_operator == LogicOperator.OR:
            # OR logic: at least one condition must be true
            group_result = any(cond_result.result for cond_result in condition_results)
        else:
            # AND logic: all conditions must be true
            group_result = all(cond_result.result for cond_result in condition_results)
        
        evaluation_details = {
            'logic_used': logic_operator.value,
            'conditions_evaluated': len(condition_results),
            'conditions_passed': sum(1 for cr in condition_results if cr.result),
            'conditions_failed': sum(1 for cr in condition_results if not cr.result)
        }
        
        return GroupEvaluationResult(
            group_result=group_result,
            condition_results=condition_results,
            logic_operator=logic_operator,
            evaluation_details=evaluation_details
        )
    
    def _evaluate_single_condition(self, condition: Condition, 
                                  indicator_data: IndicatorData) -> ConditionEvaluationResult:
        """
        Evaluate a single condition
        
        Args:
            condition: Condition to evaluate
            indicator_data: Current indicator values
            
        Returns:
            ConditionEvaluationResult with details
        """
        try:
            # PHASE 2 FIX: Add comprehensive exit debugging
            self.logger.debug(f"[CONDITION_EVAL] Evaluating: {condition.indicator} {condition.operator} {condition.value}")
            
            # Get indicator value
            indicator_value = indicator_data.get(condition.indicator)
            self.logger.debug(f"[CONDITION_EVAL] Retrieved indicator '{condition.indicator}' = {indicator_value}")
            
            if indicator_value is None:
                available_indicators = list(indicator_data.data.keys()) if hasattr(indicator_data, 'data') else 'N/A'
                self.logger.warning(f"[CONDITION_EVAL] Indicator '{condition.indicator}' not found. Available: {available_indicators}")
                return ConditionEvaluationResult(
                    condition=condition,
                    result=False,
                    error_message=f"Indicator '{condition.indicator}' not found in data"
                )
            
            # Resolve comparison value
            comparison_value = condition.value
            if isinstance(condition.value, str) and condition.value in indicator_data.data:
                # Value is another indicator
                comparison_value = indicator_data.get(condition.value)
                self.logger.debug(f"[CONDITION_EVAL] Retrieved comparison indicator '{condition.value}' = {comparison_value}")
                if comparison_value is None:
                    self.logger.warning(f"[CONDITION_EVAL] Comparison indicator '{condition.value}' not found in data")
                    return ConditionEvaluationResult(
                        condition=condition,
                        result=False,
                        indicator_value=indicator_value,
                        error_message=f"Comparison indicator '{condition.value}' not found in data"
                    )
            else:
                self.logger.debug(f"[CONDITION_EVAL] Using literal comparison value: {comparison_value}")
            
            # Log the mathematical comparison being performed
            self.logger.debug(f"[CONDITION_EVAL] Mathematical comparison: {indicator_value} {condition.operator} {comparison_value}")
            
            # Evaluate condition using operator resolver
            result = self.operator_resolver.evaluate_condition(
                indicator_value, condition.operator, comparison_value
            )
            
            # Log the result with detailed context
            self.logger.debug(f"[CONDITION_EVAL] Result: {indicator_value} {condition.operator} {comparison_value} = {result}")
            
            return ConditionEvaluationResult(
                condition=condition,
                result=result,
                indicator_value=indicator_value,
                comparison_value=comparison_value
            )
            
        except Exception as e:
            return ConditionEvaluationResult(
                condition=condition,
                result=False,
                error_message=f"Evaluation error: {e}"
            )
    
    def get_evaluation_summary(self, result: RuleResult) -> Dict[str, Any]:
        """
        Get a detailed summary of rule evaluation
        
        Args:
            result: RuleResult from evaluation
            
        Returns:
            Dictionary with evaluation summary
        """
        details = result.evaluation_details or {}
        
        summary = {
            'rule_name': result.rule_name,
            'triggered': result.triggered,
            'total_conditions': details.get('total_conditions', 0),
            'matched_conditions_count': len(result.matched_conditions),
            'total_groups': details.get('total_groups', 0),
            'matched_conditions': result.matched_conditions,
            'error_message': result.error_message
        }
        
        # Add group-level details
        group_evaluations = details.get('group_evaluations', [])
        if group_evaluations:
            summary['group_summary'] = {
                'groups_passed': sum(1 for g in group_evaluations if g['group_result']),
                'groups_failed': sum(1 for g in group_evaluations if not g['group_result']),
                'group_details': group_evaluations
            }
        
        return summary
    
    def validate_rule_conditions(self, rule_conditions: List[RuleCondition], 
                                available_indicators: List[str]) -> List[str]:
        """
        Validate rule conditions against available indicators
        
        Args:
            rule_conditions: List of conditions to validate
            available_indicators: List of available indicator names
            
        Returns:
            List of validation issues
        """
        issues = []
        
        try:
            for condition in rule_conditions:
                # Check if indicator exists
                if condition.indicator not in available_indicators:
                    issues.append(f"Indicator '{condition.indicator}' not available (condition order: {condition.condition_order})")
                
                # Check if comparison value is an indicator
                if isinstance(condition.value, str) and condition.value not in available_indicators:
                    # Try to convert to number - if it fails, it might be a missing indicator
                    try:
                        float(condition.value)
                    except ValueError:
                        issues.append(f"Comparison value '{condition.value}' might be missing indicator (condition order: {condition.condition_order})")
                
                # Validate operator
                if not self.operator_resolver.is_supported_operator(condition.operator):
                    issues.append(f"Unsupported operator '{condition.operator}' (condition order: {condition.condition_order})")
                
                # Validate logic operator
                if condition.logic and condition.logic not in ['AND', 'OR', '']:
                    issues.append(f"Invalid logic operator '{condition.logic}' (condition order: {condition.condition_order})")
        
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return issues


# Convenience functions
def evaluate_conditions_with_logic(rule_conditions: List[RuleCondition], 
                                  indicator_data: IndicatorData,
                                  rule_name: str,
                                  operator_resolver: Optional[OperatorResolver] = None) -> RuleResult:
    """
    Convenience function to evaluate conditions with AND/OR logic
    
    Args:
        rule_conditions: List of conditions to evaluate
        indicator_data: Current indicator values
        rule_name: Name of the rule
        operator_resolver: Custom operator resolver (optional)
        
    Returns:
        RuleResult with evaluation details
    """
    evaluator = ConditionGroupEvaluator(operator_resolver)
    return evaluator.evaluate_rule_conditions(rule_conditions, indicator_data, rule_name)