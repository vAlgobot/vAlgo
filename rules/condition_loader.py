"""
Condition loader for the vAlgo Rule Engine

Extracts and parses rule conditions from Excel configuration files.
Converts Excel-based rule definitions into Rule engine data models.
"""

import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .models import Rule, Condition, ConditionGroup, LogicOperator, ExitType


class ConditionLoader:
    """
    Loads and parses rule conditions from Excel configuration
    
    Extracts Entry_Conditions and Exit_Conditions from config.xlsx
    and converts them into Rule engine data models
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize condition loader
        
        Args:
            config_path: Path to Excel config file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/config.xlsx"
        self.config_data: Dict[str, pd.DataFrame] = {}
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
            
            # Load required sheets
            required_sheets = ['Entry_Conditions', 'Exit_Conditions', 'Strategy_Config']
            loaded_sheets = 0
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if not df.empty:
                        self.config_data[sheet_name.lower()] = df
                        self.logger.debug(f"Loaded sheet: {sheet_name}")
                        loaded_sheets += 1
                except Exception as e:
                    self.logger.warning(f"Error loading sheet {sheet_name}: {e}")
            
            # Check if required sheets are present
            missing_sheets = []
            for sheet in required_sheets:
                if sheet.lower() not in self.config_data:
                    missing_sheets.append(sheet)
            
            if missing_sheets:
                self.logger.error(f"Missing required sheets: {missing_sheets}")
                return False
            
            self.logger.info(f"Config loaded successfully from {self.config_path} ({loaded_sheets} sheets)")
            self._loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return False
    
    def load_entry_conditions(self) -> Dict[str, Rule]:
        """
        Load and parse entry conditions from config
        
        Returns:
            Dictionary mapping rule names to Rule objects
        """
        if not self.load_config():
            self.logger.error("Failed to load config for entry conditions")
            return {}
        
        if 'entry_conditions' not in self.config_data:
            self.logger.error("Entry_Conditions sheet not found")
            return {}
        
        try:
            df = self.config_data['entry_conditions']
            return self._parse_conditions_dataframe(df, rule_type="entry")
            
        except Exception as e:
            self.logger.error(f"Error parsing entry conditions: {e}")
            return {}
    
    def load_exit_conditions(self) -> Dict[str, Rule]:
        """
        Load and parse exit conditions from config
        
        Returns:
            Dictionary mapping rule names to Rule objects
        """
        if not self.load_config():
            self.logger.error("Failed to load config for exit conditions")
            return {}
        
        if 'exit_conditions' not in self.config_data:
            self.logger.error("Exit_Conditions sheet not found")
            return {}
        
        try:
            df = self.config_data['exit_conditions']
            return self._parse_conditions_dataframe(df, rule_type="exit")
            
        except Exception as e:
            self.logger.error(f"Error parsing exit conditions: {e}")
            return {}
    
    def load_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load strategy configurations that link entry and exit rules
        
        Returns:
            Dictionary mapping strategy names to their configurations
        """
        if not self.load_config():
            self.logger.error("Failed to load config for strategy configs")
            return {}
        
        if 'strategy_config' not in self.config_data:
            self.logger.error("Strategy_Config sheet not found")
            return {}
        
        try:
            df = self.config_data['strategy_config']
            return self._parse_strategy_configs(df)
            
        except Exception as e:
            self.logger.error(f"Error parsing strategy configs: {e}")
            return {}
    
    def _parse_conditions_dataframe(self, df: pd.DataFrame, rule_type: str) -> Dict[str, Rule]:
        """
        Parse conditions dataframe into Rule objects
        
        Args:
            df: Conditions dataframe
            rule_type: "entry" or "exit"
            
        Returns:
            Dictionary of Rule objects
        """
        # Forward fill Rule_Name to handle blank cells
        df = df.copy()
        rule_name_col = 'Rule_Name'
        if rule_name_col not in df.columns:
            self.logger.error(f"Rule_Name column not found in {rule_type} conditions")
            return {}
        
        df['Rule_Name_Filled'] = df[rule_name_col].ffill()
        df['Status_Filled'] = df['Status'].ffill() if 'Status' in df.columns else 'Active'
        
        # For exit conditions, also forward fill Exit_Type
        if rule_type == "exit" and 'Exit_Type' in df.columns:
            df['Exit_Type_Filled'] = df['Exit_Type'].ffill()
        
        parsed_rules = {}
        
        # Group by filled Rule_Name
        for rule_name, rule_group in df.groupby('Rule_Name_Filled'):
            if pd.isna(rule_name) or rule_name == '':
                continue
            
            try:
                rule = self._parse_single_rule(rule_group, rule_name, rule_type)
                if rule:
                    parsed_rules[rule_name] = rule
            except Exception as e:
                self.logger.error(f"Error parsing rule '{rule_name}': {e}")
                continue
        
        self.logger.info(f"Parsed {len(parsed_rules)} {rule_type} rules")
        return parsed_rules
    
    def _parse_single_rule(self, rule_group: pd.DataFrame, rule_name: str, rule_type: str) -> Optional[Rule]:
        """
        Parse a single rule from grouped dataframe rows
        
        Args:
            rule_group: DataFrame rows for this rule
            rule_name: Name of the rule
            rule_type: "entry" or "exit"
            
        Returns:
            Rule object or None if parsing fails
        """
        # Get rule status
        rule_status = rule_group['Status_Filled'].iloc[0]
        if pd.isna(rule_status):
            rule_status = 'Active'
        
        # Process conditions by Condition_Order
        condition_groups = []
        
        if 'Condition_Order' in rule_group.columns:
            # Group by Condition_Order
            for order, order_group in rule_group.groupby('Condition_Order'):
                if pd.isna(order):
                    continue
                
                condition_group = self._parse_condition_group(order_group, int(order), rule_type)
                if condition_group:
                    condition_groups.append(condition_group)
        else:
            # Fallback: treat each row as separate condition group
            for i, (_, row) in enumerate(rule_group.iterrows()):
                condition_group = self._parse_condition_group(pd.DataFrame([row]), i + 1, rule_type)
                if condition_group:
                    condition_groups.append(condition_group)
        
        if not condition_groups:
            self.logger.warning(f"No valid conditions found for rule '{rule_name}'")
            return None
        
        # Sort condition groups by order
        condition_groups.sort(key=lambda x: x.order)
        
        return Rule(
            name=rule_name,
            rule_type=rule_type,
            status=rule_status,
            condition_groups=condition_groups
        )
    
    def _parse_condition_group(self, order_group: pd.DataFrame, order: int, rule_type: str) -> Optional[ConditionGroup]:
        """
        Parse a condition group from dataframe rows
        
        Args:
            order_group: DataFrame rows for this condition order
            order: Condition order number
            rule_type: "entry" or "exit"
            
        Returns:
            ConditionGroup object or None
        """
        conditions = []
        logic_ops = []
        exit_types = []
        
        for _, row in order_group.iterrows():
            # Parse condition based on rule type
            if rule_type == "entry":
                condition = self._parse_entry_condition(row)
            else:
                condition = self._parse_exit_condition(row)
            
            if condition:
                conditions.append(condition)
                
                # Get logic operator
                logic = row.get('Logic', '')
                logic_ops.append(str(logic) if pd.notna(logic) else '')
                
                # Get exit type for exit conditions
                if rule_type == "exit":
                    exit_type = row.get('Exit_Type_Filled', 'Signal')
                    exit_types.append(str(exit_type) if pd.notna(exit_type) else 'Signal')
        
        if not conditions:
            return None
        
        # Determine logic to next group
        logic_to_next = LogicOperator.NONE
        if logic_ops and logic_ops[0]:
            logic_str = logic_ops[0].upper().strip()
            if logic_str in ['AND', '&']:
                logic_to_next = LogicOperator.AND
            elif logic_str in ['OR', '|']:
                logic_to_next = LogicOperator.OR
        
        # Determine exit type
        exit_type = None
        if rule_type == "exit" and exit_types:
            exit_type = exit_types[0]
        
        return ConditionGroup(
            order=order,
            conditions=conditions,
            logic_to_next=logic_to_next,
            exit_type=exit_type
        )
    
    def _parse_entry_condition(self, row: pd.Series) -> Optional[Condition]:
        """Parse entry condition from dataframe row"""
        indicator = row.get('Indicator_1', '')
        operator = row.get('Operator_1', '')
        value = row.get('Value_1', '')
        
        if pd.isna(indicator) or pd.isna(operator) or pd.isna(value):
            return None
        
        return Condition(
            indicator=str(indicator).strip(),
            operator=str(operator).strip(),
            value=str(value).strip()
        )
    
    def _parse_exit_condition(self, row: pd.Series) -> Optional[Condition]:
        """Parse exit condition from dataframe row"""
        indicator = row.get('Indicator', '')
        operator = row.get('Operator', '')
        value = row.get('Value', '')
        exit_type = row.get('Exit_Type_Filled', '')
        
        if pd.isna(indicator) or pd.isna(operator) or pd.isna(value):
            return None
        
        return Condition(
            indicator=str(indicator).strip(),
            operator=str(operator).strip(),
            value=str(value).strip(),
            exit_type=str(exit_type).strip() if pd.notna(exit_type) else None
        )
    
    def _parse_strategy_configs(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Parse strategy configurations"""
        # Forward fill Strategy_Name
        df = df.copy()
        df['Strategy_Name_Filled'] = df['Strategy_Name'].ffill()
        
        strategies = {}
        
        for strategy_name, strategy_group in df.groupby('Strategy_Name_Filled'):
            if pd.isna(strategy_name) or strategy_name == '':
                continue
            
            # Get strategy parameters from first row
            first_row = strategy_group.iloc[0]
            
            # Collect entry and exit rules
            entry_rules = []
            exit_rules = []
            
            for _, row in strategy_group.iterrows():
                entry_rule = row.get('Entry_Rule', '')
                if pd.notna(entry_rule) and entry_rule != '':
                    entry_rules.append(str(entry_rule))
                
                exit_rule = row.get('Exit_Rule', '')
                if pd.notna(exit_rule) and exit_rule != '':
                    exit_rules.append(str(exit_rule))
            
            # Remove duplicates while preserving order
            entry_rules = list(dict.fromkeys(entry_rules))
            exit_rules = list(dict.fromkeys(exit_rules))
            
            strategies[strategy_name] = {
                'entry_rules': entry_rules,
                'exit_rules': exit_rules,
                'status': str(first_row.get('Status', 'Active')),
                'signal_type': str(first_row.get('Signal_Type', 'Call')),
                'strike_preference': str(first_row.get('Strike_Preference', 'ATM')),
                'breakout_confirmation': self._parse_boolean(first_row.get('Breakout_Confirmation', False)),
                'sl_method': str(first_row.get('SL_Method', 'ATR')),
                'tp_method': str(first_row.get('TP_Method', 'Hybrid')),
                'real_option_data': self._parse_boolean(first_row.get('Real_Option_Data', True))
            }
        
        self.logger.info(f"Parsed {len(strategies)} strategy configurations")
        return strategies
    
    def _parse_boolean(self, value: Any) -> bool:
        """Parse boolean value from various input types"""
        if pd.isna(value):
            return False
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        str_value = str(value).lower().strip()
        return str_value in ['true', '1', 'yes', 'on', 'enabled']
    
    def get_available_rules(self) -> Dict[str, List[str]]:
        """
        Get list of available rules by type
        
        Returns:
            Dictionary with 'entry' and 'exit' rule lists
        """
        if not self.load_config():
            return {'entry': [], 'exit': []}
        
        entry_rules = list(self.load_entry_conditions().keys())
        exit_rules = list(self.load_exit_conditions().keys())
        
        return {
            'entry': entry_rules,
            'exit': exit_rules
        }
    
    def validate_config(self) -> List[str]:
        """
        Validate the loaded configuration
        
        Returns:
            List of validation issues
        """
        issues = []
        
        if not self.load_config():
            issues.append("Failed to load configuration file")
            return issues
        
        # Check required sheets
        required_sheets = ['entry_conditions', 'exit_conditions', 'strategy_config']
        for sheet in required_sheets:
            if sheet not in self.config_data:
                issues.append(f"Missing required sheet: {sheet}")
        
        # Validate entry conditions
        try:
            entry_rules = self.load_entry_conditions()
            if not entry_rules:
                issues.append("No valid entry rules found")
            else:
                for rule_name, rule in entry_rules.items():
                    if not rule.condition_groups:
                        issues.append(f"Entry rule '{rule_name}' has no conditions")
        except Exception as e:
            issues.append(f"Error validating entry conditions: {e}")
        
        # Validate exit conditions
        try:
            exit_rules = self.load_exit_conditions()
            if not exit_rules:
                issues.append("No valid exit rules found")
            else:
                for rule_name, rule in exit_rules.items():
                    if not rule.condition_groups:
                        issues.append(f"Exit rule '{rule_name}' has no conditions")
        except Exception as e:
            issues.append(f"Error validating exit conditions: {e}")
        
        # Validate strategies
        try:
            strategies = self.load_strategy_configs()
            if not strategies:
                issues.append("No valid strategies found")
            else:
                for strategy_name, strategy in strategies.items():
                    if not strategy.get('entry_rules'):
                        issues.append(f"Strategy '{strategy_name}' has no entry rules")
                    if not strategy.get('exit_rules'):
                        issues.append(f"Strategy '{strategy_name}' has no exit rules")
        except Exception as e:
            issues.append(f"Error validating strategies: {e}")
        
        return issues