"""
vAlgo Rule Engine Module

A modular, testable rule engine for strategy condition evaluation.
Supports complex rule configurations with multiple conditions, logic operators,
and independent testing using CSV data.

Components:
- RuleEngine: Main rule evaluation engine
- ConditionLoader: Parse rules from config
- OperatorResolver: Handle condition operators
- RuleParser: Parse complex rule expressions
- Models: Data models for rules and results

Usage:
    from rules import RuleEngine, ConditionLoader
    
    # Load rules from config
    loader = ConditionLoader(config_path)
    rules = loader.load_entry_conditions()
    
    # Create engine and evaluate
    engine = RuleEngine(rules)
    result = engine.evaluate_rule("EMA_Crossover_Long", indicator_data)
"""

from .engine import RuleEngine
from .condition_loader import ConditionLoader
from .operator_resolver import OperatorResolver
from .rule_parser import RuleParser
from .models import Rule, Condition, RuleResult, ConditionGroup
from .integration_helper import RuleEngineIntegrationHelper, create_integration_helper, evaluate_single_condition
from .signal_processor import SignalDataProcessor, SignalAnalysisResult

# Enhanced components for Excel-driven strategy configuration
from .enhanced_rule_engine import EnhancedRuleEngine, EnhancedSignalProcessor, create_enhanced_rule_engine
from .enhanced_condition_loader import EnhancedConditionLoader, StrategyRow, RuleCondition, load_enhanced_strategy_config
from .condition_group_evaluator import ConditionGroupEvaluator, evaluate_conditions_with_logic
from .entry_exit_state_machine import EntryExitStateMachine, TradeTracker, TradeState, create_state_machine_from_config

__version__ = "1.0.0"
__author__ = "vAlgo Team"

__all__ = [
    # Original components
    "RuleEngine",
    "ConditionLoader", 
    "OperatorResolver",
    "RuleParser",
    "Rule",
    "Condition",
    "RuleResult",
    "ConditionGroup",
    "RuleEngineIntegrationHelper",
    "create_integration_helper",
    "evaluate_single_condition",
    "SignalDataProcessor",
    "SignalAnalysisResult",
    
    # Enhanced components
    "EnhancedRuleEngine",
    "EnhancedSignalProcessor", 
    "create_enhanced_rule_engine",
    "EnhancedConditionLoader",
    "StrategyRow",
    "RuleCondition", 
    "load_enhanced_strategy_config",
    "ConditionGroupEvaluator",
    "evaluate_conditions_with_logic",
    "EntryExitStateMachine",
    "TradeTracker",
    "TradeState",
    "create_state_machine_from_config"
]