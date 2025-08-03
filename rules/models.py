"""
Data models for the vAlgo Rule Engine

Defines the core data structures used throughout the rule engine:
- Condition: Individual condition (indicator operator value)
- ConditionGroup: Group of conditions with order and logic
- Rule: Complete rule with multiple condition groups
- RuleResult: Result of rule evaluation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime


class LogicOperator(Enum):
    """Logic operators for combining conditions"""
    AND = "AND"
    OR = "OR"
    NONE = ""


class ExitType(Enum):
    """Types of exit conditions"""
    LTP_EXIT = "LTP_exit"
    EMA_EXIT = "EMA_exit"
    SIGNAL_EXIT = "Signal_exit"
    TIME_EXIT = "Time_exit"


@dataclass
class Condition:
    """
    Individual condition for rule evaluation
    
    Represents a single condition like "ema_9 > ema_20" or "rsi_14 < 30"
    """
    indicator: str          # Indicator name (e.g., "ema_9", "rsi_14")
    operator: str          # Comparison operator (">", "<", "=", etc.)
    value: Union[str, float, int]  # Comparison value or another indicator
    exit_type: Optional[str] = None  # For exit conditions (LTP_exit, EMA_exit)
    
    def __post_init__(self):
        """Validate condition after initialization"""
        if not self.indicator:
            raise ValueError("Condition indicator cannot be empty")
        if not self.operator:
            raise ValueError("Condition operator cannot be empty")
        if self.value is None or self.value == "":
            raise ValueError("Condition value cannot be None or empty")
    
    def __str__(self) -> str:
        """String representation of condition"""
        return f"{self.indicator} {self.operator} {self.value}"


@dataclass
class ConditionGroup:
    """
    Group of conditions with order and logic
    
    Represents conditions that belong together in evaluation order
    with logic operator to next group
    """
    order: int                      # Evaluation order (1, 2, 3, etc.)
    conditions: List[Condition]     # List of conditions in this group
    logic_to_next: LogicOperator = LogicOperator.NONE  # Logic to next group
    exit_type: Optional[str] = None  # For exit condition groups
    
    def __post_init__(self):
        """Validate condition group after initialization"""
        if self.order < 1:
            raise ValueError("Condition order must be >= 1")
        if not self.conditions:
            raise ValueError("ConditionGroup must have at least one condition")
        
        # Convert string logic operators to enum
        if isinstance(self.logic_to_next, str):
            logic_str = self.logic_to_next.upper().strip()
            if logic_str in ["AND", "&", "&&"]:
                self.logic_to_next = LogicOperator.AND
            elif logic_str in ["OR", "|", "||"]:
                self.logic_to_next = LogicOperator.OR
            else:
                self.logic_to_next = LogicOperator.NONE
    
    def __str__(self) -> str:
        """String representation of condition group"""
        if len(self.conditions) == 1:
            return str(self.conditions[0])
        else:
            condition_strs = [str(cond) for cond in self.conditions]
            return f"({' OR '.join(condition_strs)})"


@dataclass
class Rule:
    """
    Complete rule with multiple condition groups
    
    Represents a full rule like "EMA_Crossover_Long" with all its conditions
    """
    name: str                                   # Rule name
    rule_type: str                             # "entry" or "exit"
    status: str                                # "Active" or "Inactive"
    condition_groups: List[ConditionGroup]     # Ordered list of condition groups
    total_conditions: int = 0                  # Total number of individual conditions
    
    def __post_init__(self):
        """Calculate total conditions and validate rule"""
        if not self.name:
            raise ValueError("Rule name cannot be empty")
        if not self.condition_groups:
            raise ValueError("Rule must have at least one condition group")
        
        # Calculate total conditions
        self.total_conditions = sum(len(group.conditions) for group in self.condition_groups)
        
        # Sort condition groups by order
        self.condition_groups.sort(key=lambda x: x.order)
    
    def is_active(self) -> bool:
        """Check if rule is active"""
        return self.status.lower() == "active"
    
    def get_expression(self) -> str:
        """Get human-readable rule expression"""
        if not self.condition_groups:
            return "No conditions"
        
        group_expressions = []
        for group in self.condition_groups:
            group_expressions.append(str(group))
        
        # Join groups with AND by default (can be customized later)
        return " AND ".join(group_expressions)


@dataclass
class RuleResult:
    """
    Result of rule evaluation
    
    Contains all information about whether a rule was triggered and why
    """
    rule_name: str                              # Name of evaluated rule
    triggered: bool                             # Whether rule was triggered
    timestamp: Optional[datetime] = None        # When evaluation occurred
    matched_conditions: List[str] = field(default_factory=list)  # Which conditions matched
    failed_conditions: List[str] = field(default_factory=list)   # Which conditions failed
    evaluation_details: Dict[str, Any] = field(default_factory=dict)  # Detailed evaluation info
    error_message: Optional[str] = None         # Error message if evaluation failed
    
    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def add_matched_condition(self, condition: str, value: Any = None):
        """Add a matched condition to the result"""
        self.matched_conditions.append(condition)
        if value is not None:
            self.evaluation_details[condition] = value
    
    def add_failed_condition(self, condition: str, actual_value: Any = None, expected: Any = None):
        """Add a failed condition to the result"""
        self.failed_conditions.append(condition)
        if actual_value is not None or expected is not None:
            self.evaluation_details[condition] = {
                'actual': actual_value,
                'expected': expected
            }
    
    def get_success_rate(self) -> float:
        """Get percentage of conditions that matched"""
        total_conditions = len(self.matched_conditions) + len(self.failed_conditions)
        if total_conditions == 0:
            return 0.0
        return (len(self.matched_conditions) / total_conditions) * 100
    
    def __str__(self) -> str:
        """String representation of result"""
        status = "TRIGGERED" if self.triggered else "NOT TRIGGERED"
        return f"Rule '{self.rule_name}': {status} ({len(self.matched_conditions)} matched, {len(self.failed_conditions)} failed)"


@dataclass
class IndicatorData:
    """
    Container for indicator data used in rule evaluation
    
    Wraps pandas Series/DataFrame to provide consistent interface
    """
    data: Dict[str, Any]        # Indicator values by name
    timestamp: Optional[datetime] = None
    
    def get(self, indicator_name: str, default: Any = None) -> Any:
        """Get indicator value by name"""
        return self.data.get(indicator_name, default)
    
    def has(self, indicator_name: str) -> bool:
        """Check if indicator exists"""
        return indicator_name in self.data
    
    def get_all_indicators(self) -> List[str]:
        """Get list of all available indicators"""
        return list(self.data.keys())
    
    @classmethod
    def from_pandas_series(cls, series: Any, timestamp: Optional[datetime] = None) -> 'IndicatorData':
        """Create IndicatorData from pandas Series"""
        return cls(data=series.to_dict(), timestamp=timestamp)
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any], timestamp: Optional[datetime] = None) -> 'IndicatorData':
        """Create IndicatorData from dictionary"""
        return cls(data=data_dict, timestamp=timestamp)


# Type aliases for convenience
RuleDict = Dict[str, Rule]
ConditionDict = Dict[str, List[Condition]]