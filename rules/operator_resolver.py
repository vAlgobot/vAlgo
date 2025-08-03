"""
Operator resolver for the vAlgo Rule Engine

Handles evaluation of comparison operators between indicator values.
Supports numerical comparisons, string comparisons, and special trading operators.
"""

import operator
from typing import Any, Union, Callable, Dict
import math
import logging


class OperatorResolver:
    """
    Resolves and evaluates comparison operators for rule conditions
    
    Supports standard operators (>, <, =, etc.) and trading-specific comparisons
    """
    
    def __init__(self):
        """Initialize operator resolver with supported operators"""
        self.logger = logging.getLogger(__name__)
        
        # Standard comparison operators
        self.operators: Dict[str, Callable[[Any, Any], bool]] = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '=': operator.eq,
            '==': operator.eq,
            '!=': operator.ne,
            '<>': operator.ne,
            'GT': operator.gt,      # Alternative syntax
            'LT': operator.lt,
            'GTE': operator.ge,
            'LTE': operator.le,
            'EQ': operator.eq,
            'NEQ': operator.ne,
        }
        
        # Special trading operators
        self.special_operators = {
            'CROSSES_ABOVE': self._crosses_above,
            'CROSSES_BELOW': self._crosses_below,
            'WITHIN': self._within_range,
            'OUTSIDE': self._outside_range,
            'INCREASING': self._is_increasing,
            'DECREASING': self._is_decreasing
        }
        
        # Combine all operators
        self.all_operators = {**self.operators, **self.special_operators}
    
    def resolve_operator(self, operator_str: str) -> Callable[[Any, Any], bool]:
        """
        Get operator function from string
        
        Args:
            operator_str: Operator string (e.g., '>', 'CROSSES_ABOVE')
            
        Returns:
            Operator function
            
        Raises:
            ValueError: If operator is not supported
        """
        if not operator_str:
            raise ValueError("Operator string cannot be empty")
        
        # Normalize operator string
        normalized_op = operator_str.strip().upper()
        
        # Check for exact match first
        if normalized_op in self.all_operators:
            return self.all_operators[normalized_op]
        
        # Check original case for standard operators
        if operator_str.strip() in self.operators:
            return self.operators[operator_str.strip()]
        
        # If not found, raise error with suggestions
        available_ops = list(self.all_operators.keys())
        raise ValueError(f"Unsupported operator '{operator_str}'. Available operators: {available_ops}")
    
    def evaluate_condition(self, left_value: Any, operator_str: str, right_value: Any) -> bool:
        """
        Evaluate a single condition
        
        Args:
            left_value: Left operand (usually indicator value)
            operator_str: Comparison operator
            right_value: Right operand (value or another indicator)
            
        Returns:
            Boolean result of comparison
            
        Raises:
            ValueError: If operator is not supported or evaluation fails
        """
        try:
            # Get operator function
            op_func = self.resolve_operator(operator_str)
            
            # Convert values for comparison
            left_converted = self._convert_value(left_value)
            right_converted = self._convert_value(right_value)
            
            # Handle None values
            if left_converted is None or right_converted is None:
                self.logger.warning(f"None value in condition: {left_value} {operator_str} {right_value}")
                return False
            
            # PHASE 2 FIX: Add detailed operator evaluation debugging
            result = op_func(left_converted, right_converted)
            
            # Enhanced logging for exit condition debugging
            self.logger.debug(f"[OPERATOR_EVAL] Mathematical evaluation: {left_converted} {operator_str} {right_converted} = {result}")
            self.logger.debug(f"[OPERATOR_EVAL] Value types: {type(left_converted).__name__} {operator_str} {type(right_converted).__name__}")
            
            if not result:
                # Log why condition failed for debugging
                self.logger.debug(f"[OPERATOR_EVAL] ❌ Condition FAILED: {left_converted} {operator_str} {right_converted}")
            else:
                self.logger.debug(f"[OPERATOR_EVAL] ✅ Condition PASSED: {left_converted} {operator_str} {right_converted}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{left_value} {operator_str} {right_value}': {e}")
            raise ValueError(f"Failed to evaluate condition: {e}")
    
    def _convert_value(self, value: Any) -> Union[float, str, None]:
        """
        Convert value to appropriate type for comparison
        
        Args:
            value: Value to convert
            
        Returns:
            Converted value (float for numbers, string for text, None for invalid)
        """
        if value is None:
            return None
        
        # Handle pandas NaN and similar
        if hasattr(value, 'isna') and value.isna():
            return None
        
        # Handle objects with __float__ method (like pandas values)
        if hasattr(value, '__float__'):
            try:
                float_val = float(value)
                if math.isnan(float_val):
                    return None
                if math.isinf(float_val):
                    return None  # Treat infinity as None for safety
                return float_val
            except (ValueError, TypeError):
                pass
        
        # Handle string representations of NaN
        if isinstance(value, str):
            if value.lower().strip() in ['nan', 'none', 'null', '']:
                return None
            
            # Try to convert string to number
            try:
                # Handle special values
                if value.lower() == 'inf':
                    return None  # Treat infinity as None for safety
                elif value.lower() == '-inf':
                    return None  # Treat infinity as None for safety
                
                # Try float conversion
                return float(value)
            except (ValueError, TypeError):
                # Return as string if not convertible
                return str(value).strip()
        
        # Handle numeric types
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return None  # Treat NaN and infinity as None for safety
            return float(value)
        
        # Try to convert other types to float first
        try:
            float_val = float(value)
            if math.isnan(float_val) or math.isinf(float_val):
                return None
            return float_val
        except (ValueError, TypeError):
            # Convert other types to string as fallback
            return str(value)
    
    def get_supported_operators(self) -> list:
        """Get list of all supported operators"""
        return list(self.all_operators.keys())
    
    def is_operator_supported(self, operator_str: str) -> bool:
        """Check if operator is supported"""
        try:
            self.resolve_operator(operator_str)
            return True
        except ValueError:
            return False
    
    # Special trading operators implementation
    
    def _crosses_above(self, current_value: float, threshold: float) -> bool:
        """
        Check if value crosses above threshold
        Note: This is a simplified version. Full implementation would need historical data.
        """
        # For now, just check if current value is above threshold
        # In full implementation, would check previous value was below
        return current_value > threshold
    
    def _crosses_below(self, current_value: float, threshold: float) -> bool:
        """
        Check if value crosses below threshold
        Note: This is a simplified version. Full implementation would need historical data.
        """
        # For now, just check if current value is below threshold
        # In full implementation, would check previous value was above
        return current_value < threshold
    
    def _within_range(self, value: float, range_value: str) -> bool:
        """
        Check if value is within range
        Expected format: "min,max" or "center±tolerance"
        """
        try:
            if ',' in range_value:
                # Format: "min,max"
                parts = range_value.split(',')
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                return min_val <= value <= max_val
            elif '±' in range_value or '+/-' in range_value:
                # Format: "center±tolerance"
                separator = '±' if '±' in range_value else '+/-'
                parts = range_value.split(separator)
                center = float(parts[0].strip())
                tolerance = float(parts[1].strip())
                return (center - tolerance) <= value <= (center + tolerance)
            else:
                # Single value - exact match
                return value == float(range_value)
        except (ValueError, IndexError):
            self.logger.error(f"Invalid range format: {range_value}")
            return False
    
    def _outside_range(self, value: float, range_value: str) -> bool:
        """Check if value is outside range"""
        return not self._within_range(value, range_value)
    
    def _is_increasing(self, current_value: float, previous_value: float) -> bool:
        """Check if value is increasing"""
        return current_value > previous_value
    
    def _is_decreasing(self, current_value: float, previous_value: float) -> bool:
        """Check if value is decreasing"""
        return current_value < previous_value


# Singleton instance for global use
default_resolver = OperatorResolver()


def evaluate_condition(left_value: Any, operator_str: str, right_value: Any) -> bool:
    """
    Convenience function for evaluating conditions using default resolver
    
    Args:
        left_value: Left operand
        operator_str: Comparison operator
        right_value: Right operand
        
    Returns:
        Boolean result
    """
    return default_resolver.evaluate_condition(left_value, operator_str, right_value)


def get_supported_operators() -> list:
    """Get list of supported operators"""
    return default_resolver.get_supported_operators()