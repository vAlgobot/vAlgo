"""
Enhanced Condition Loader for vAlgo Rule Engine

Supports Excel-driven strategy configuration with:
- Row-by-row strategy processing (1:1 entry-exit pairing)
- Active status filtering within rules
- condition_order based AND/OR logic
- User-friendly Excel configuration
"""

import signal
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import time


from .models import Rule, Condition, ConditionGroup, LogicOperator, ExitType


@dataclass
class StrategyRow:
    """Represents a single row from strategy_config sheet"""
    strategy_name: str
    entry_rule: str
    exit_rule: str
    active: bool
    position_size: float = 1000.0
    risk_per_trade: float = 0.02
    max_positions: int = 1
    additional_params: Dict[str, Any] = None
    signal_type: str = ""


@dataclass
class RunStrategyConfig:
    """Represents configuration from Run_Strategy sheet for strategy selection"""
    strategy_name: str
    run_status: bool
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    max_daily_entries: Optional[int] = None
    notes: str = ""


@dataclass
class RuleCondition:
    """Enhanced condition with order and logic support"""
    condition_order: int
    indicator: str
    operator: str
    value: str
    logic: str = ""  # AND, OR, or empty for last condition
    active: bool = True
    exit_type: str = "Signal"  # For exit conditions


class EnhancedConditionLoader:
    """
    Enhanced condition loader supporting Excel-driven strategy configuration
    
    Features:
    - Row-by-row strategy processing
    - Active status filtering
    - condition_order based evaluation
    - AND/OR logic support
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize enhanced condition loader
        
        Args:
            config_path: Path to Excel config file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/config.xlsx"
        self.config_data: Dict[str, pd.DataFrame] = {}
        self.run_strategy_configs: Dict[str, RunStrategyConfig] = {}
        self._loaded = False
    
    def load_config(self, force_reload: bool = False) -> bool:
        """
        Load configuration from Excel file
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        if self._loaded and not force_reload:
            return True
        
        try:
            if not Path(self.config_path).exists():
                self.logger.error(f"Config file not found: {self.config_path}")
                return False
            
            # Load all sheets from Excel file
            excel_file = pd.ExcelFile(self.config_path)
            
            # Load required sheets for enhanced functionality
            required_sheets = ['Strategy_Config', 'Entry_Conditions', 'Exit_Conditions']
            loaded_sheets = 0
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if not df.empty:
                        # Normalize sheet names for consistent access
                        normalized_name = sheet_name.lower().replace('_conditions', '_condition')
                        self.config_data[normalized_name] = df
                        self.logger.debug(f"Loaded sheet: {sheet_name} as {normalized_name}")
                        loaded_sheets += 1
                except Exception as e:
                    self.logger.warning(f"Error loading sheet {sheet_name}: {e}")
            
            # Check if required sheets are present
            required_normalized = ['strategy_config', 'entry_condition', 'exit_condition']
            missing_sheets = []
            for sheet in required_normalized:
                if sheet not in self.config_data:
                    missing_sheets.append(sheet)
            
            if missing_sheets:
                self.logger.error(f"Missing required sheets: {missing_sheets}")
                return False
            
            # Load Run_Strategy configuration (optional for backward compatibility)
            if not self.load_run_strategy_config():
                self.logger.warning("Run_Strategy sheet not found or invalid - using all active strategies")
            
            self.logger.info(f"Enhanced config loaded successfully from {self.config_path} ({loaded_sheets} sheets)")
            self._loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading enhanced config: {e}")
            return False
    
    def load_run_strategy_config(self) -> bool:
        """
        Load Run_Strategy configuration from Excel sheet
        
        Returns:
            True if successful, False if sheet missing or invalid
        """
        try:
            # Check if Run_Strategy sheet exists
            if 'run_strategy' not in self.config_data:
                return False
            
            df = self.config_data['run_strategy']
            self.run_strategy_configs.clear()
            
            for _, row in df.iterrows():
                # Extract required fields
                strategy_name = str(row.get('Strategy_Name', '')).strip()
                if not strategy_name:
                    continue
                
                # Parse Run_Status (support different formats)
                run_status_value = row.get('Run_Status', 'Active')
                if pd.isna(run_status_value):
                    run_status = False
                else:
                    run_status_str = str(run_status_value).lower().strip()
                    run_status = run_status_str in ['true', '1', 'yes', 'active', 'on']
                
                # Parse time fields
                start_time = None
                end_time = None
                
                try:
                    start_time_str = str(row.get('Start_Time', '')).strip()
                    if start_time_str and start_time_str != 'nan':
                        # Handle different time formats
                        if ':' in start_time_str:
                            time_parts = start_time_str.split(':')
                            start_time = time(int(time_parts[0]), int(time_parts[1]), 
                                            int(time_parts[2]) if len(time_parts) > 2 else 0)
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Invalid Start_Time for {strategy_name}: {start_time_str}")
                
                try:
                    end_time_str = str(row.get('End_Time', '')).strip()
                    if end_time_str and end_time_str != 'nan':
                        # Handle different time formats
                        if ':' in end_time_str:
                            time_parts = end_time_str.split(':')
                            end_time = time(int(time_parts[0]), int(time_parts[1]), 
                                          int(time_parts[2]) if len(time_parts) > 2 else 0)
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Invalid End_Time for {strategy_name}: {end_time_str}")
                
                # Parse Max_Daily_Entries
                max_daily_entries = None
                try:
                    max_entries_value = row.get('Max_Daily_Entries')
                    if pd.notna(max_entries_value) and str(max_entries_value).strip():
                        max_daily_entries = int(float(max_entries_value))
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid Max_Daily_Entries for {strategy_name}: {max_entries_value}")
                
                # Get notes
                notes = str(row.get('Notes', '')).strip()
                
                # Create RunStrategyConfig
                run_config = RunStrategyConfig(
                    strategy_name=strategy_name,
                    run_status=run_status,
                    start_time=start_time,
                    end_time=end_time,
                    max_daily_entries=max_daily_entries,
                    notes=notes
                )
                
                self.run_strategy_configs[strategy_name] = run_config
                self.logger.debug(f"Loaded Run_Strategy config: {strategy_name} (status: {run_status})")
            
            self.logger.info(f"Loaded {len(self.run_strategy_configs)} Run_Strategy configurations")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading Run_Strategy config: {e}")
            return False
    
    def load_strategy_rows(self) -> List[StrategyRow]:
        """
        Load strategy configuration as individual rows (not grouped by strategy)
        Each row represents one entry_rule + exit_rule pair
        
        Returns:
            List of StrategyRow objects for active strategies
        """
        if not self.load_config():
            self.logger.error("Failed to load config for strategy rows")
            return []
        
        if 'strategy_config' not in self.config_data:
            self.logger.error("Strategy_Config sheet not found")
            return []
        
        try:
            df = self.config_data['strategy_config']
            strategy_rows = []
            
            for _, row in df.iterrows():
                # Parse Active status (support different formats)
                active_value = row.get('Active', row.get('Status', 'True'))
                if pd.isna(active_value):
                    active = False
                else:
                    active_str = str(active_value).lower().strip()
                    active = active_str in ['true', '1', 'yes', 'active', 'on']
                
                # Only process active rows from Strategy_Config
                if not active:
                    self.logger.debug(f"Skipping inactive strategy row: {row.get('Strategy_Name', 'Unknown')}")
                    continue
                
                # Extract required fields
                strategy_name = str(row.get('Strategy_Name', '')).strip()
                entry_rule = str(row.get('Entry_Rule', '')).strip()
                exit_rule = str(row.get('Exit_Rule', '')).strip()
                signal_type = str(row.get('Signal_Type', '')).strip()
                
                
                # STRICT VALIDATION: Skip rows with invalid strategy names (NaN, empty, etc.)
                if not strategy_name or strategy_name.lower() in ['nan', 'none', '']:
                    self.logger.warning(f"Skipping row with invalid strategy name: '{strategy_name}' - NaN or empty values not allowed")
                    continue
                
                # STRICT VALIDATION: Skip rows with missing required fields
                if not entry_rule or not exit_rule:
                    self.logger.warning(f"Skipping incomplete strategy row '{strategy_name}': missing entry_rule='{entry_rule}' or exit_rule='{exit_rule}'")
                    continue
                
                # STRICT VALIDATION: Run_Strategy sheet is MANDATORY for enhanced backtesting
                if not self.run_strategy_configs:
                    raise ValueError(
                        "Run_Strategy sheet is mandatory for enhanced backtesting. "
                        "Please ensure Run_Strategy sheet exists and contains strategy selections. "
                        "No fallback to 'all active' strategies is allowed."
                    )
                
                # PHASE 2 FILTERING: Check Run_Strategy selection (STRICT - no fallbacks)
                if strategy_name not in self.run_strategy_configs:
                    self.logger.debug(f"Strategy '{strategy_name}' not found in Run_Strategy sheet - skipping (strict filtering)")
                    continue
                
                run_config = self.run_strategy_configs[strategy_name]
                if not run_config.run_status:
                    self.logger.info(f"Strategy '{strategy_name}' disabled in Run_Strategy sheet - skipping")
                    continue
                
                self.logger.debug(f"Strategy '{strategy_name}' selected for execution (Run_Strategy: Active)")
                
                # Parse optional fields with defaults
                position_size = float(row.get('Position_Size', 0)) if pd.notna(row.get('Position_Size')) else 1000.0
                risk_per_trade = float(row.get('Risk_Per_Trade', 0.02)) if pd.notna(row.get('Risk_Per_Trade')) else 0.02
                max_positions = int(row.get('Max_Positions', 1)) if pd.notna(row.get('Max_Positions')) else 1
                
                
                # Store additional parameters
                additional_params = {}
                for col in df.columns:
                    if col not in ['Strategy_Name', 'Entry_Rule', 'Exit_Rule', 'Active', 'Status', 
                                   'Position_Size', 'Risk_Per_Trade', 'Max_Positions']:
                        additional_params[col] = row.get(col)
                
                strategy_row = StrategyRow(
                    strategy_name=strategy_name,
                    entry_rule=entry_rule,
                    exit_rule=exit_rule,
                    active=active,
                    signal_type=signal_type,
                    position_size=position_size,
                    risk_per_trade=risk_per_trade,
                    max_positions=max_positions,
                    additional_params=additional_params
                )
                
                strategy_rows.append(strategy_row)
            
            # STRICT VALIDATION: Ensure we have at least one active strategy
            if not strategy_rows:
                total_in_config = len(df) if not df.empty else 0
                total_in_run_strategy = len(self.run_strategy_configs) if self.run_strategy_configs else 0
                active_in_run_strategy = len([c for c in self.run_strategy_configs.values() if c.run_status]) if self.run_strategy_configs else 0
                
                error_msg = (
                    f"No active strategies found after enhanced filtering. "
                    f"Total strategies in Strategy_Config: {total_in_config}, "
                    f"Total in Run_Strategy: {total_in_run_strategy}, "
                    f"Active in Run_Strategy: {active_in_run_strategy}. "
                    f"Please ensure at least one strategy is Active in Strategy_Config and enabled in Run_Strategy."
                )
                
                raise ValueError(error_msg)
            
            self.logger.info(f"Successfully loaded {len(strategy_rows)} active strategy rows (after Run_Strategy filtering)")
            
            # Log filtering summary
            if self.run_strategy_configs:
                total_run_configs = len(self.run_strategy_configs)
                active_run_configs = len([c for c in self.run_strategy_configs.values() if c.run_status])
                self.logger.info(f"Run_Strategy filtering: {active_run_configs}/{total_run_configs} strategies enabled")
                
                # Log individual strategy details
                for strategy_name, config in self.run_strategy_configs.items():
                    status = "ENABLED" if config.run_status else "DISABLED"
                    self.logger.debug(f"Strategy '{strategy_name}': {status}")
            
            return strategy_rows
            
        except Exception as e:
            self.logger.error(f"Error loading strategy rows: {e}")
            # Re-raise the exception instead of returning empty list for strict validation
            raise
    
    def get_run_strategy_config(self, strategy_name: str) -> Optional[RunStrategyConfig]:
        """
        Get Run_Strategy configuration for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            RunStrategyConfig object or None if not found
        """
        return self.run_strategy_configs.get(strategy_name)
    
    def load_rule_conditions(self, rule_name: str, rule_type: str) -> List[RuleCondition]:
        """
        Load conditions for a specific rule with Active filtering and condition_order
        
        Args:
            rule_name: Name of the rule to load
            rule_type: "entry" or "exit"
            
        Returns:
            List of RuleCondition objects sorted by condition_order
        """
        if not self.load_config():
            self.logger.error("Failed to load config for rule conditions")
            return []
        
        sheet_name = f"{rule_type}_condition"
        if sheet_name not in self.config_data:
            self.logger.error(f"{rule_type.title()}_Condition sheet not found")
            return []
        
        try:
            df = self.config_data[sheet_name]
            
            # Forward fill Rule_Name to handle merged cells
            df = df.copy()
            df['Rule_Name_Filled'] = df['Rule_Name'].ffill()
            
            # Filter for the specific rule
            rule_df = df[df['Rule_Name_Filled'] == rule_name].copy()
            
            if rule_df.empty:
                self.logger.warning(f"No conditions found for {rule_type} rule '{rule_name}'")
                return []
            
            conditions = []
            
            for _, row in rule_df.iterrows():
                # Parse Active status (default to True if not specified)
                active_value = row.get('Active', row.get('Status', True))
                if pd.isna(active_value):
                    active = True
                else:
                    active_str = str(active_value).lower().strip()
                    active = active_str in ['true', '1', 'yes', 'active', 'on']
                
                # Only include active conditions
                if not active:
                    self.logger.debug(f"Skipping inactive condition: order {condition_order} in rule '{rule_name}'")
                    continue
                
                # Extract condition components with proper NaN handling
                condition_order = int(row.get('Condition_Order', 1)) if pd.notna(row.get('Condition_Order')) else 1
                
                # PHASE 2 FIX: Proper NaN handling for indicator values
                indicator_raw = row.get('Indicator', row.get('Indicator_1', ''))
                indicator = str(indicator_raw).strip() if pd.notna(indicator_raw) else ''
                
                operator_raw = row.get('Operator', row.get('Operator_1', ''))
                operator = str(operator_raw).strip() if pd.notna(operator_raw) else ''
                
                value_raw = row.get('Value', row.get('Value_1', ''))
                value = str(value_raw).strip() if pd.notna(value_raw) else ''
                
                logic_raw = row.get('Logic', '')
                logic = str(logic_raw).strip().upper() if pd.notna(logic_raw) else ''
                
                # For exit conditions, get exit type
                exit_type = "Signal"
                if rule_type == "exit":
                    exit_type_raw = row.get('Exit_Type', 'Signal')
                    exit_type = str(exit_type_raw).strip() if pd.notna(exit_type_raw) else 'Signal'
                
                # Skip incomplete conditions (including NaN converted to 'nan')
                if not indicator or not operator or not value or indicator == 'nan' or operator == 'nan' or value == 'nan':
                    self.logger.debug(f"Skipping incomplete/NaN condition in rule '{rule_name}': indicator='{indicator}', operator='{operator}', value='{value}'")
                    continue
                
                condition = RuleCondition(
                    condition_order=condition_order,
                    indicator=indicator,
                    operator=operator,
                    value=value,
                    logic=logic,
                    active=active,
                    exit_type=exit_type
                )
                
                conditions.append(condition)
            
            # Sort by condition_order
            conditions.sort(key=lambda x: x.condition_order)
            
            self.logger.debug(f"Loaded {len(conditions)} active conditions for {rule_type} rule '{rule_name}':")
            for cond in conditions:
                self.logger.debug(f"  Order {cond.condition_order}: {cond.indicator} {cond.operator} {cond.value} (Logic: {cond.logic})")
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error loading conditions for rule '{rule_name}': {e}")
            return []
    
    def build_rule_from_conditions(self, rule_name: str, conditions: List[RuleCondition], 
                                  rule_type: str) -> Optional[Rule]:
        """
        Build a Rule object from RuleCondition list
        
        Args:
            rule_name: Name of the rule
            conditions: List of RuleCondition objects
            rule_type: "entry" or "exit"
            
        Returns:
            Rule object or None if error
        """
        if not conditions:
            self.logger.warning(f"No conditions to build rule '{rule_name}'")
            return None
        
        try:
            # Convert RuleCondition objects to Condition objects
            rule_conditions = []
            condition_groups = []
            
            current_group = []
            
            for i, rule_cond in enumerate(conditions):
                # Create Condition object
                condition = Condition(
                    indicator=rule_cond.indicator,
                    operator=rule_cond.operator,
                    value=rule_cond.value,
                    exit_type=rule_cond.exit_type if rule_type == "exit" else None
                )
                
                rule_conditions.append(condition)
                current_group.append(condition)
                
                # Check if this is the end of a group (OR logic or last condition)
                is_last = (i == len(conditions) - 1)
                has_or_logic = rule_cond.logic == "OR"
                
                if has_or_logic or is_last:
                    # End current group
                    group_logic = LogicOperator.OR if len(current_group) > 1 else LogicOperator.AND
                    
                    condition_group = ConditionGroup(
                        conditions=current_group.copy(),
                        logic_operator=group_logic
                    )
                    condition_groups.append(condition_group)
                    current_group = []
            
            # If no groups were created, create a single AND group
            if not condition_groups and rule_conditions:
                condition_groups = [ConditionGroup(
                    conditions=rule_conditions,
                    logic_operator=LogicOperator.AND
                )]
            
            # Create Rule object
            rule = Rule(
                name=rule_name,
                conditions=rule_conditions,
                condition_groups=condition_groups,
                rule_type=rule_type
            )
            
            return rule
            
        except Exception as e:
            self.logger.error(f"Error building rule '{rule_name}': {e}")
            return None
    
    def load_entry_conditions_enhanced(self) -> Dict[str, Rule]:
        """
        Load all entry conditions with enhanced filtering and logic
        
        Returns:
            Dictionary mapping rule names to Rule objects
        """
        strategy_rows = self.load_strategy_rows()
        
        # Get unique entry rules from active strategies
        entry_rule_names = set()
        for row in strategy_rows:
            entry_rule_names.add(row.entry_rule)
        
        entry_rules = {}
        
        for rule_name in entry_rule_names:
            conditions = self.load_rule_conditions(rule_name, "entry")
            if conditions:
                rule = self.build_rule_from_conditions(rule_name, conditions, "entry")
                if rule:
                    entry_rules[rule_name] = rule
        
        self.logger.info(f"Loaded {len(entry_rules)} enhanced entry rules")
        return entry_rules
    
    def load_exit_conditions_enhanced(self) -> Dict[str, Rule]:
        """
        Load all exit conditions with enhanced filtering and logic
        
        Returns:
            Dictionary mapping rule names to Rule objects
        """
        strategy_rows = self.load_strategy_rows()
        
        # Get unique exit rules from active strategies
        exit_rule_names = set()
        for row in strategy_rows:
            exit_rule_names.add(row.exit_rule)
        
        exit_rules = {}
        
        for rule_name in exit_rule_names:
            conditions = self.load_rule_conditions(rule_name, "exit")
            if conditions:
                rule = self.build_rule_from_conditions(rule_name, conditions, "exit")
                if rule:
                    exit_rules[rule_name] = rule
        
        self.logger.info(f"Loaded {len(exit_rules)} enhanced exit rules")
        return exit_rules
    
    def get_strategy_row_pairs(self) -> List[Tuple[StrategyRow, str, str]]:
        """
        Get strategy rows with their entry-exit rule pairs
        
        Returns:
            List of tuples: (StrategyRow, entry_rule_name, exit_rule_name)
        """
        strategy_rows = self.load_strategy_rows()
        pairs = []
        
        for row in strategy_rows:
            pairs.append((row, row.entry_rule, row.exit_rule))
        
        return pairs
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the enhanced configuration
        
        Returns:
            List of validation issues
        """
        issues = []
        
        try:
            # Load strategy rows
            strategy_rows = self.load_strategy_rows()
            if not strategy_rows:
                issues.append("No active strategy rows found in Strategy_Config")
                return issues
            
            # Check if all referenced entry/exit rules exist
            entry_rule_names = set()
            exit_rule_names = set()
            
            for row in strategy_rows:
                entry_rule_names.add(row.entry_rule)
                exit_rule_names.add(row.exit_rule)
            
            # Validate entry rules
            for rule_name in entry_rule_names:
                conditions = self.load_rule_conditions(rule_name, "entry")
                if not conditions:
                    issues.append(f"Entry rule '{rule_name}' has no active conditions")
            
            # Validate exit rules
            for rule_name in exit_rule_names:
                conditions = self.load_rule_conditions(rule_name, "exit")
                if not conditions:
                    issues.append(f"Exit rule '{rule_name}' has no active conditions")
            
            self.logger.info(f"Configuration validation completed with {len(issues)} issues")
            
        except Exception as e:
            issues.append(f"Error during validation: {e}")
        
        return issues
    
    def validate_run_strategy_configuration(self) -> List[str]:
        """
        Validate Run_Strategy sheet configuration
        
        Returns:
            List of validation issues found
        """
        issues = []
        
        try:
            if not self.run_strategy_configs:
                return ["Run_Strategy sheet not loaded or empty"]
            
            # Get all strategy names from Strategy_Config
            strategy_config_names = set()
            if 'strategy_config' in self.config_data:
                df = self.config_data['strategy_config']
                for _, row in df.iterrows():
                    strategy_name = str(row.get('Strategy_Name', '')).strip()
                    if strategy_name:
                        strategy_config_names.add(strategy_name)
            
            # Validate each Run_Strategy entry
            for strategy_name, run_config in self.run_strategy_configs.items():
                # Check if strategy exists in Strategy_Config
                if strategy_name not in strategy_config_names:
                    issues.append(f"Strategy '{strategy_name}' in Run_Strategy sheet not found in Strategy_Config")
                
                # Validate time windows
                if run_config.start_time and run_config.end_time:
                    if run_config.start_time >= run_config.end_time:
                        issues.append(f"Strategy '{strategy_name}': Start_Time ({run_config.start_time}) must be before End_Time ({run_config.end_time})")
                elif run_config.start_time and not run_config.end_time:
                    issues.append(f"Strategy '{strategy_name}': Start_Time specified but End_Time missing")
                elif not run_config.start_time and run_config.end_time:
                    issues.append(f"Strategy '{strategy_name}': End_Time specified but Start_Time missing")
                
                # Validate Max_Daily_Entries
                if run_config.max_daily_entries is not None:
                    if run_config.max_daily_entries < 1:
                        issues.append(f"Strategy '{strategy_name}': Max_Daily_Entries ({run_config.max_daily_entries}) must be >= 1")
                    elif run_config.max_daily_entries > 100:
                        issues.append(f"Strategy '{strategy_name}': Max_Daily_Entries ({run_config.max_daily_entries}) seems unusually high (>100)")
            
            # Check for strategies in Strategy_Config but missing from Run_Strategy
            missing_strategies = strategy_config_names - set(self.run_strategy_configs.keys())
            if missing_strategies:
                for strategy in missing_strategies:
                    issues.append(f"Strategy '{strategy}' in Strategy_Config but missing from Run_Strategy sheet")
            
            self.logger.info(f"Run_Strategy validation completed with {len(issues)} issues")
            
        except Exception as e:
            issues.append(f"Error during Run_Strategy validation: {e}")
        
        return issues
    
    def get_strategy_selection_summary(self) -> Dict[str, Any]:
        """
        Get summary of strategy selection and configuration
        
        Returns:
            Dictionary with strategy selection details
        """
        summary = {
            'total_strategies_in_config': 0,
            'total_strategies_in_run_strategy': len(self.run_strategy_configs),
            'active_strategies_in_run_strategy': 0,
            'selected_strategies': [],
            'disabled_strategies': [],
            'missing_from_run_strategy': []
        }
        
        try:
            # Get all strategies from Strategy_Config
            strategy_config_names = set()
            if 'strategy_config' in self.config_data:
                df = self.config_data['strategy_config']
                for _, row in df.iterrows():
                    active_value = row.get('Active', row.get('Status', 'True'))
                    if pd.isna(active_value):
                        active = False
                    else:
                        active_str = str(active_value).lower().strip()
                        active = active_str in ['true', '1', 'yes', 'active', 'on']
                    
                    if active:
                        strategy_name = str(row.get('Strategy_Name', '')).strip()
                        if strategy_name:
                            strategy_config_names.add(strategy_name)
                
                summary['total_strategies_in_config'] = len(strategy_config_names)
            
            # Analyze Run_Strategy configurations
            for strategy_name, run_config in self.run_strategy_configs.items():
                if run_config.run_status:
                    summary['active_strategies_in_run_strategy'] += 1
                    summary['selected_strategies'].append({
                        'name': strategy_name,
                        'start_time': str(run_config.start_time) if run_config.start_time else None,
                        'end_time': str(run_config.end_time) if run_config.end_time else None,
                        'max_daily_entries': run_config.max_daily_entries,
                        'notes': run_config.notes
                    })
                else:
                    summary['disabled_strategies'].append(strategy_name)
            
            # Find strategies missing from Run_Strategy
            missing_strategies = strategy_config_names - set(self.run_strategy_configs.keys())
            summary['missing_from_run_strategy'] = list(missing_strategies)
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary


# Convenience functions
def load_enhanced_strategy_config(config_path: Optional[str] = None) -> EnhancedConditionLoader:
    """
    Load enhanced strategy configuration
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Configured EnhancedConditionLoader
    """
    loader = EnhancedConditionLoader(config_path)
    loader.load_config()
    return loader