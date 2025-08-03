#!/usr/bin/env python3
"""
Enhanced Exit Analyzer for vAlgo Trading System
==============================================

Advanced exit analysis system that tracks and analyzes how trades exit - 
whether by equity-based SL/TP or option chain-based SL/TP triggers.

Features:
- Dual exit method tracking (equity vs options)
- Detailed exit reason analysis
- Performance comparison between exit methods
- Effective logging for backtesting summary
- Statistical analysis of exit triggers
- Options vs equity efficiency metrics

Author: vAlgo Development Team
Created: July 11, 2025
Version: 1.0.0 (Production)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from data_manager.options_database import OptionsDatabase, create_options_database


class ExitAnalysisResult:
    """Container for exit analysis results."""
    
    def __init__(self):
        self.equity_exits = {
            'sl_hits': 0,
            'tp_hits': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.0,
            'avg_exit_time_minutes': 0.0,
            'trades': []
        }
        
        self.options_exits = {
            'sl_hits': 0,
            'tp_hits': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.0,
            'avg_exit_time_minutes': 0.0,
            'trades': []
        }
        
        self.comparison_metrics = {
            'equity_dominance': 0.0,
            'options_dominance': 0.0,
            'efficiency_ratio': 0.0,
            'faster_exit_method': 'unknown',
            'more_profitable_method': 'unknown'
        }
        
        self.detailed_analysis = {
            'exit_timing_analysis': {},
            'pnl_distribution': {},
            'exit_method_effectiveness': {},
            'recommendations': []
        }


class RealMoneyExitAnalysisResult:
    """Container for real money exit analysis results with actual strike data."""
    
    def __init__(self):
        # Real money exit tracking
        self.real_money_exits = {
            'breakout_confirmed_exits': {'count': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0, 'trades': []},
            'immediate_exits': {'count': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0, 'trades': []},
            'cpr_based_exits': {'count': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0, 'trades': []},
            'basic_sl_tp_exits': {'count': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0, 'trades': []}
        }
        
        # Real strike performance
        self.strike_performance = {
            'atm_strikes': {'count': 0, 'total_pnl': 0.0, 'win_rate': 0.0},
            'itm_strikes': {'count': 0, 'total_pnl': 0.0, 'win_rate': 0.0},
            'otm_strikes': {'count': 0, 'total_pnl': 0.0, 'win_rate': 0.0}
        }
        
        # Entry/Exit precision metrics
        self.precision_metrics = {
            'entry_premium_accuracy': 0.0,
            'exit_premium_accuracy': 0.0,
            'real_vs_theoretical_pnl': 0.0,
            'commission_impact': 0.0,
            'slippage_impact': 0.0
        }
        
        # CPR effectiveness
        self.cpr_effectiveness = {
            'cpr_sl_hits': 0,
            'cpr_tp_hits': 0,
            'cpr_accuracy_rate': 0.0,
            'non_cpr_performance': 0.0
        }
        
        # Real database integration stats
        self.database_integration = {
            'real_data_queries': 0,
            'cache_hits': 0,
            'data_quality_score': 0.0,
            'real_vs_provided_variance': 0.0
        }


class EnhancedExitAnalyzer:
    """
    Advanced exit analyzer for tracking and analyzing dual exit methods.
    """
    
    def __init__(self, options_db: Optional[OptionsDatabase] = None, enable_real_money_analysis: bool = False):
        """
        Initialize Enhanced Exit Analyzer with optional real money capabilities.
        
        Args:
            options_db: Optional OptionsDatabase for real money analysis
            enable_real_money_analysis: Enable real money analysis features
        """
        self.logger = get_logger(__name__)
        
        # Exit tracking data
        self.exit_history = []
        self.exit_stats = defaultdict(lambda: defaultdict(int))
        
        # Analysis results
        self.current_analysis = ExitAnalysisResult()
        
        # Real money analysis
        self.enable_real_money_analysis = enable_real_money_analysis
        self.options_db = options_db or (create_options_database() if enable_real_money_analysis else None)
        self.real_money_analysis = RealMoneyExitAnalysisResult() if enable_real_money_analysis else None
        self.real_money_history = [] if enable_real_money_analysis else None
        
        # Configuration
        self.config = {
            'track_exit_timing': True,
            'analyze_pnl_distribution': True,
            'generate_recommendations': True,
            'min_trades_for_analysis': 5,
            'confidence_threshold': 0.7,
            'enable_real_strike_verification': enable_real_money_analysis,
            'track_database_performance': enable_real_money_analysis,
            'analyze_cpr_effectiveness': enable_real_money_analysis
        }
        
        self.logger.info("EnhancedExitAnalyzer initialized successfully")
    
    def analyze_trade_exit(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single trade exit and determine the exit method.
        
        Args:
            trade_data: Trade data containing exit information
            
        Returns:
            Analysis result for the trade
        """
        try:
            # Extract key information
            exit_reason = trade_data.get('exit_reason', 'UNKNOWN')
            pnl = float(trade_data.get('pnl', 0))
            entry_time = trade_data.get('entry_time')
            exit_time = trade_data.get('exit_time')
            
            # Determine exit method
            exit_method = self._determine_exit_method(exit_reason, trade_data)
            
            # Calculate exit timing
            exit_duration = self._calculate_exit_duration(entry_time, exit_time)
            
            # Analyze exit effectiveness
            effectiveness = self._analyze_exit_effectiveness(trade_data, exit_method)
            
            # Create analysis result
            analysis_result = {
                'trade_id': trade_data.get('trade_id', 'unknown'),
                'exit_method': exit_method,
                'exit_reason': exit_reason,
                'exit_trigger': self._get_exit_trigger(exit_reason),
                'pnl': pnl,
                'exit_duration_minutes': exit_duration,
                'effectiveness_score': effectiveness,
                'exit_price_data': {
                    'equity_price_at_exit': trade_data.get('equity_price_at_exit'),
                    'option_premium_at_exit': trade_data.get('exit_premium'),
                    'equity_sl_level': trade_data.get('sl_tp_levels', {}).get('equity_sl'),
                    'equity_tp_level': trade_data.get('sl_tp_levels', {}).get('equity_tp'),
                    'option_sl_level': trade_data.get('sl_tp_levels', {}).get('options_sl'),
                    'option_tp_level': trade_data.get('sl_tp_levels', {}).get('options_tp')
                },
                'market_conditions': {
                    'volatility_at_exit': trade_data.get('volatility_at_exit'),
                    'volume_at_exit': trade_data.get('volume_at_exit'),
                    'time_to_expiry': trade_data.get('greeks', {}).get('dte', 0)
                }
            }
            
            # Add to exit history
            self.exit_history.append(analysis_result)
            
            # Update statistics
            self._update_exit_statistics(analysis_result)
            
            self.logger.debug(f"Analyzed exit for trade {analysis_result['trade_id']}: "
                            f"{exit_method} - {exit_reason} - PnL: {pnl:.2f}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade exit: {e}")
            return {}
    
    def _determine_exit_method(self, exit_reason: str, trade_data: Dict[str, Any]) -> str:
        """Determine the primary exit method based on exit reason."""
        try:
            # Map exit reasons to methods
            if 'EQUITY' in exit_reason.upper():
                return 'equity'
            elif 'OPTION' in exit_reason.upper():
                return 'options'
            elif 'PREMIUM' in exit_reason.upper():
                return 'options'
            elif 'UNDERLYING' in exit_reason.upper():
                return 'equity'
            elif exit_reason.upper() in ['TP_HIT', 'SL_HIT']:
                # Need to analyze which triggered first
                return self._analyze_dual_trigger(trade_data)
            elif exit_reason.upper() in ['EXPIRED', 'EXPIRY']:
                return 'time_based'
            else:
                return 'unknown'
                
        except Exception as e:
            self.logger.error(f"Error determining exit method: {e}")
            return 'unknown'
    
    def _analyze_dual_trigger(self, trade_data: Dict[str, Any]) -> str:
        """Analyze which exit method triggered first in dual SL/TP setup."""
        try:
            # Get exit data
            exit_premium = trade_data.get('exit_premium', 0)
            equity_price_at_exit = trade_data.get('equity_price_at_exit', 0)
            sl_tp_levels = trade_data.get('sl_tp_levels', {})
            
            # Check options levels
            options_sl = sl_tp_levels.get('options_sl', 0)
            options_tp = sl_tp_levels.get('options_tp', 0)
            
            # Check equity levels
            equity_sl = sl_tp_levels.get('equity_sl', 0)
            equity_tp = sl_tp_levels.get('equity_tp', 0)
            
            # Determine which triggered
            option_type = trade_data.get('option_type', 'CALL')
            
            # Calculate distances from trigger levels
            option_sl_distance = abs(exit_premium - options_sl) if options_sl > 0 else float('inf')
            option_tp_distance = abs(exit_premium - options_tp) if options_tp > 0 else float('inf')
            
            if option_type == 'CALL':
                equity_sl_distance = abs(equity_price_at_exit - equity_sl) if equity_sl > 0 else float('inf')
                equity_tp_distance = abs(equity_price_at_exit - equity_tp) if equity_tp > 0 else float('inf')
            else:  # PUT
                equity_sl_distance = abs(equity_price_at_exit - equity_sl) if equity_sl > 0 else float('inf')
                equity_tp_distance = abs(equity_price_at_exit - equity_tp) if equity_tp > 0 else float('inf')
            
            # Find minimum distance (closest to trigger)
            min_distance = min(option_sl_distance, option_tp_distance, equity_sl_distance, equity_tp_distance)
            
            if min_distance == option_sl_distance or min_distance == option_tp_distance:
                return 'options'
            elif min_distance == equity_sl_distance or min_distance == equity_tp_distance:
                return 'equity'
            else:
                return 'unknown'
                
        except Exception as e:
            self.logger.error(f"Error analyzing dual trigger: {e}")
            return 'unknown'
    
    def _get_exit_trigger(self, exit_reason: str) -> str:
        """Get the exit trigger type (SL/TP/Other)."""
        try:
            exit_reason_upper = exit_reason.upper()
            
            if 'SL' in exit_reason_upper or 'STOP' in exit_reason_upper:
                return 'stop_loss'
            elif 'TP' in exit_reason_upper or 'TARGET' in exit_reason_upper or 'PROFIT' in exit_reason_upper:
                return 'take_profit'
            elif 'EXPIR' in exit_reason_upper:
                return 'expiry'
            elif 'MANUAL' in exit_reason_upper:
                return 'manual'
            else:
                return 'other'
                
        except Exception:
            return 'other'
    
    def _calculate_exit_duration(self, entry_time: datetime, exit_time: datetime) -> float:
        """Calculate exit duration in minutes."""
        try:
            if not entry_time or not exit_time:
                return 0.0
            
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
            
            duration = exit_time - entry_time
            return duration.total_seconds() / 60.0
            
        except Exception as e:
            self.logger.error(f"Error calculating exit duration: {e}")
            return 0.0
    
    def _analyze_exit_effectiveness(self, trade_data: Dict[str, Any], exit_method: str) -> float:
        """Analyze the effectiveness of the exit method (0-1 score)."""
        try:
            pnl = trade_data.get('pnl', 0)
            entry_premium = trade_data.get('entry_premium', 0)
            exit_premium = trade_data.get('exit_premium', 0)
            
            # Calculate return percentage
            if entry_premium > 0:
                return_pct = (exit_premium - entry_premium) / entry_premium
            else:
                return_pct = 0
            
            # Base effectiveness on return and exit method appropriateness
            base_score = 0.5  # Neutral
            
            # Positive return increases effectiveness
            if return_pct > 0:
                base_score += min(return_pct, 0.5)  # Cap at 1.0
            else:
                base_score += max(return_pct, -0.5)  # Floor at 0.0
            
            # Adjust for exit method appropriateness
            if exit_method == 'options':
                # Options exits are generally more precise
                base_score += 0.1
            elif exit_method == 'equity':
                # Equity exits might be less precise but more predictable
                base_score += 0.05
            
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            self.logger.error(f"Error analyzing exit effectiveness: {e}")
            return 0.5
    
    def _update_exit_statistics(self, analysis_result: Dict[str, Any]) -> None:
        """Update exit statistics with new analysis result."""
        try:
            exit_method = analysis_result.get('exit_method', 'unknown')
            exit_trigger = analysis_result.get('exit_trigger', 'other')
            pnl = analysis_result.get('pnl', 0)
            
            # Update counters
            self.exit_stats[exit_method][exit_trigger] += 1
            self.exit_stats[exit_method]['total_pnl'] += pnl
            self.exit_stats[exit_method]['total_trades'] += 1
            
            # Update duration stats
            duration = analysis_result.get('exit_duration_minutes', 0)
            if f'{exit_method}_durations' not in self.exit_stats:
                self.exit_stats[f'{exit_method}_durations'] = []
            self.exit_stats[f'{exit_method}_durations'].append(duration)
            
        except Exception as e:
            self.logger.error(f"Error updating exit statistics: {e}")
    
    def generate_comprehensive_analysis(self) -> ExitAnalysisResult:
        """Generate comprehensive exit analysis."""
        try:
            if len(self.exit_history) < self.config['min_trades_for_analysis']:
                self.logger.warning(f"Insufficient trades for analysis: {len(self.exit_history)}")
                return ExitAnalysisResult()
            
            result = ExitAnalysisResult()
            
            # Analyze equity exits
            equity_exits = [e for e in self.exit_history if e.get('exit_method') == 'equity']
            result.equity_exits = self._analyze_exit_group(equity_exits, 'equity')
            
            # Analyze options exits
            options_exits = [e for e in self.exit_history if e.get('exit_method') == 'options']
            result.options_exits = self._analyze_exit_group(options_exits, 'options')
            
            # Generate comparison metrics
            result.comparison_metrics = self._generate_comparison_metrics(
                result.equity_exits, result.options_exits
            )
            
            # Generate detailed analysis
            result.detailed_analysis = self._generate_detailed_analysis()
            
            self.current_analysis = result
            
            self.logger.info(f"Generated comprehensive analysis for {len(self.exit_history)} trades")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive analysis: {e}")
            return ExitAnalysisResult()
    
    def _analyze_exit_group(self, exits: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
        """Analyze a group of exits for a specific method."""
        try:
            if not exits:
                return {
                    'sl_hits': 0, 'tp_hits': 0, 'total_pnl': 0.0,
                    'avg_pnl': 0.0, 'win_rate': 0.0, 'avg_exit_time_minutes': 0.0,
                    'trades': []
                }
            
            # Count SL/TP hits
            sl_hits = len([e for e in exits if e.get('exit_trigger') == 'stop_loss'])
            tp_hits = len([e for e in exits if e.get('exit_trigger') == 'take_profit'])
            
            # Calculate PnL metrics
            pnls = [e.get('pnl', 0) for e in exits]
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / len(exits) if exits else 0
            
            # Calculate win rate
            winning_trades = len([p for p in pnls if p > 0])
            win_rate = (winning_trades / len(exits)) * 100 if exits else 0
            
            # Calculate average exit time
            durations = [e.get('exit_duration_minutes', 0) for e in exits]
            avg_exit_time = sum(durations) / len(durations) if durations else 0
            
            return {
                'sl_hits': sl_hits,
                'tp_hits': tp_hits,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'win_rate': win_rate,
                'avg_exit_time_minutes': avg_exit_time,
                'trades': exits
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing exit group for {method}: {e}")
            return {}
    
    def _generate_comparison_metrics(self, equity_exits: Dict[str, Any], 
                                   options_exits: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison metrics between exit methods."""
        try:
            total_trades = len(self.exit_history)
            equity_count = len(equity_exits.get('trades', []))
            options_count = len(options_exits.get('trades', []))
            
            # Calculate dominance
            equity_dominance = (equity_count / total_trades) * 100 if total_trades > 0 else 0
            options_dominance = (options_count / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate efficiency ratio
            equity_avg_pnl = equity_exits.get('avg_pnl', 0)
            options_avg_pnl = options_exits.get('avg_pnl', 0)
            
            if equity_avg_pnl != 0 and options_avg_pnl != 0:
                efficiency_ratio = options_avg_pnl / equity_avg_pnl
            else:
                efficiency_ratio = 1.0
            
            # Determine faster exit method
            equity_avg_time = equity_exits.get('avg_exit_time_minutes', 0)
            options_avg_time = options_exits.get('avg_exit_time_minutes', 0)
            
            if equity_avg_time > 0 and options_avg_time > 0:
                if equity_avg_time < options_avg_time:
                    faster_exit_method = 'equity'
                else:
                    faster_exit_method = 'options'
            else:
                faster_exit_method = 'unknown'
            
            # Determine more profitable method
            if equity_avg_pnl > options_avg_pnl:
                more_profitable_method = 'equity'
            elif options_avg_pnl > equity_avg_pnl:
                more_profitable_method = 'options'
            else:
                more_profitable_method = 'equal'
            
            return {
                'equity_dominance': equity_dominance,
                'options_dominance': options_dominance,
                'efficiency_ratio': efficiency_ratio,
                'faster_exit_method': faster_exit_method,
                'more_profitable_method': more_profitable_method,
                'total_trades_analyzed': total_trades,
                'equity_trades_count': equity_count,
                'options_trades_count': options_count
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comparison metrics: {e}")
            return {}
    
    def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis with recommendations."""
        try:
            detailed = {
                'exit_timing_analysis': self._analyze_exit_timing(),
                'pnl_distribution': self._analyze_pnl_distribution(),
                'exit_method_effectiveness': self._analyze_exit_effectiveness_overall(),
                'recommendations': self._generate_recommendations()
            }
            
            return detailed
            
        except Exception as e:
            self.logger.error(f"Error generating detailed analysis: {e}")
            return {}
    
    def _analyze_exit_timing(self) -> Dict[str, Any]:
        """Analyze exit timing patterns."""
        try:
            timing_analysis = {}
            
            for method in ['equity', 'options']:
                method_exits = [e for e in self.exit_history if e.get('exit_method') == method]
                
                if method_exits:
                    durations = [e.get('exit_duration_minutes', 0) for e in method_exits]
                    
                    timing_analysis[method] = {
                        'avg_duration_minutes': np.mean(durations),
                        'median_duration_minutes': np.median(durations),
                        'min_duration_minutes': np.min(durations),
                        'max_duration_minutes': np.max(durations),
                        'std_duration_minutes': np.std(durations)
                    }
            
            return timing_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing exit timing: {e}")
            return {}
    
    def _analyze_pnl_distribution(self) -> Dict[str, Any]:
        """Analyze PnL distribution by exit method."""
        try:
            pnl_distribution = {}
            
            for method in ['equity', 'options']:
                method_exits = [e for e in self.exit_history if e.get('exit_method') == method]
                
                if method_exits:
                    pnls = [e.get('pnl', 0) for e in method_exits]
                    
                    pnl_distribution[method] = {
                        'avg_pnl': np.mean(pnls),
                        'median_pnl': np.median(pnls),
                        'min_pnl': np.min(pnls),
                        'max_pnl': np.max(pnls),
                        'std_pnl': np.std(pnls),
                        'positive_pnl_count': len([p for p in pnls if p > 0]),
                        'negative_pnl_count': len([p for p in pnls if p < 0]),
                        'total_pnl': sum(pnls)
                    }
            
            return pnl_distribution
            
        except Exception as e:
            self.logger.error(f"Error analyzing PnL distribution: {e}")
            return {}
    
    def _analyze_exit_effectiveness_overall(self) -> Dict[str, Any]:
        """Analyze overall exit effectiveness."""
        try:
            effectiveness = {}
            
            for method in ['equity', 'options']:
                method_exits = [e for e in self.exit_history if e.get('exit_method') == method]
                
                if method_exits:
                    effectiveness_scores = [e.get('effectiveness_score', 0.5) for e in method_exits]
                    
                    effectiveness[method] = {
                        'avg_effectiveness': np.mean(effectiveness_scores),
                        'median_effectiveness': np.median(effectiveness_scores),
                        'high_effectiveness_count': len([s for s in effectiveness_scores if s > 0.7]),
                        'low_effectiveness_count': len([s for s in effectiveness_scores if s < 0.3]),
                        'total_trades': len(method_exits)
                    }
            
            return effectiveness
            
        except Exception as e:
            self.logger.error(f"Error analyzing exit effectiveness: {e}")
            return {}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        try:
            recommendations = []
            
            if len(self.exit_history) < self.config['min_trades_for_analysis']:
                return ["Insufficient data for recommendations. Need at least 5 trades."]
            
            # Analyze exit method performance
            equity_exits = [e for e in self.exit_history if e.get('exit_method') == 'equity']
            options_exits = [e for e in self.exit_history if e.get('exit_method') == 'options']
            
            equity_avg_pnl = np.mean([e.get('pnl', 0) for e in equity_exits]) if equity_exits else 0
            options_avg_pnl = np.mean([e.get('pnl', 0) for e in options_exits]) if options_exits else 0
            
            # Performance recommendations
            if options_avg_pnl > equity_avg_pnl * 1.2:
                recommendations.append("Options-based exits are significantly more profitable. Consider increasing options exit priority.")
            elif equity_avg_pnl > options_avg_pnl * 1.2:
                recommendations.append("Equity-based exits are more profitable. Consider increasing equity exit priority.")
            
            # Timing recommendations
            equity_avg_time = np.mean([e.get('exit_duration_minutes', 0) for e in equity_exits]) if equity_exits else 0
            options_avg_time = np.mean([e.get('exit_duration_minutes', 0) for e in options_exits]) if options_exits else 0
            
            if equity_avg_time > 0 and options_avg_time > 0:
                if options_avg_time < equity_avg_time * 0.8:
                    recommendations.append("Options exits are faster. Consider tightening options exit criteria for quicker exits.")
                elif equity_avg_time < options_avg_time * 0.8:
                    recommendations.append("Equity exits are faster. Consider tightening equity exit criteria for quicker exits.")
            
            # SL/TP ratio recommendations
            total_sl_hits = len([e for e in self.exit_history if e.get('exit_trigger') == 'stop_loss'])
            total_tp_hits = len([e for e in self.exit_history if e.get('exit_trigger') == 'take_profit'])
            
            if total_sl_hits > total_tp_hits * 2:
                recommendations.append("High SL hit ratio. Consider widening SL levels or tightening TP levels.")
            elif total_tp_hits > total_sl_hits * 2:
                recommendations.append("High TP hit ratio. Consider tightening SL levels or widening TP levels.")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Exit methods are performing adequately. Continue monitoring for optimization opportunities.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations. Please review manually."]
    
    def export_analysis_report(self, output_path: str = None) -> str:
        """Export comprehensive analysis report."""
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"outputs/analysis/exit_analysis_{timestamp}.json"
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Generate analysis if not done
            if not self.current_analysis or not self.current_analysis.equity_exits:
                self.generate_comprehensive_analysis()
            
            # Prepare export data
            export_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_trades_analyzed': len(self.exit_history),
                'equity_exits_analysis': self.current_analysis.equity_exits,
                'options_exits_analysis': self.current_analysis.options_exits,
                'comparison_metrics': self.current_analysis.comparison_metrics,
                'detailed_analysis': self.current_analysis.detailed_analysis,
                'raw_exit_history': self.exit_history,
                'exit_statistics': dict(self.exit_stats)
            }
            
            # Export to JSON
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported exit analysis report to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis report: {e}")
            return ""
    
    def get_summary_log(self) -> str:
        """Get summary log for backtesting output."""
        try:
            if not self.current_analysis or not self.current_analysis.equity_exits:
                self.generate_comprehensive_analysis()
            
            total_trades = len(self.exit_history)
            equity_exits = self.current_analysis.equity_exits
            options_exits = self.current_analysis.options_exits
            comparison = self.current_analysis.comparison_metrics
            
            summary_lines = [
                "=" * 80,
                "EXIT ANALYSIS SUMMARY",
                "=" * 80,
                f"Total Trades Analyzed: {total_trades}",
                "",
                "EQUITY EXITS:",
                f"  - Total: {len(equity_exits.get('trades', []))} ({comparison.get('equity_dominance', 0):.1f}%)",
                f"  - SL Hits: {equity_exits.get('sl_hits', 0)}",
                f"  - TP Hits: {equity_exits.get('tp_hits', 0)}",
                f"  - Total P&L: {equity_exits.get('total_pnl', 0):.2f}",
                f"  - Avg P&L: {equity_exits.get('avg_pnl', 0):.2f}",
                f"  - Win Rate: {equity_exits.get('win_rate', 0):.1f}%",
                f"  - Avg Exit Time: {equity_exits.get('avg_exit_time_minutes', 0):.1f} minutes",
                "",
                "OPTIONS EXITS:",
                f"  - Total: {len(options_exits.get('trades', []))} ({comparison.get('options_dominance', 0):.1f}%)",
                f"  - SL Hits: {options_exits.get('sl_hits', 0)}",
                f"  - TP Hits: {options_exits.get('tp_hits', 0)}",
                f"  - Total P&L: {options_exits.get('total_pnl', 0):.2f}",
                f"  - Avg P&L: {options_exits.get('avg_pnl', 0):.2f}",
                f"  - Win Rate: {options_exits.get('win_rate', 0):.1f}%",
                f"  - Avg Exit Time: {options_exits.get('avg_exit_time_minutes', 0):.1f} minutes",
                "",
                "COMPARISON METRICS:",
                f"  - Efficiency Ratio (Options/Equity): {comparison.get('efficiency_ratio', 1.0):.2f}",
                f"  - Faster Exit Method: {comparison.get('faster_exit_method', 'unknown').upper()}",
                f"  - More Profitable Method: {comparison.get('more_profitable_method', 'unknown').upper()}",
                "",
                "RECOMMENDATIONS:",
            ]
            
            # Add recommendations
            for rec in self.current_analysis.detailed_analysis.get('recommendations', []):
                summary_lines.append(f"  - {rec}")
            
            summary_lines.append("=" * 80)
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating summary log: {e}")
            return f"Error generating exit analysis summary: {str(e)}"
    
    def analyze_real_money_exit(self, real_trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze real money trade exit with actual strike prices and CPR data.
        
        Args:
            real_trade_data: Real money trade data from OptionsRealMoneyIntegrator
            
        Returns:
            Comprehensive real money exit analysis
        """
        try:
            if not self.enable_real_money_analysis:
                self.logger.warning("Real money analysis not enabled")
                return {}
            
            # Extract real money specific data
            trade_id = real_trade_data.get('trade_id', 'unknown')
            entry_method = real_trade_data.get('entry_method', 'immediate')
            exit_reason = real_trade_data.get('exit_reason', 'UNKNOWN')
            real_pnl = real_trade_data.get('real_pnl', 0)
            
            # Analyze entry precision
            entry_analysis = self._analyze_real_entry_precision(real_trade_data)
            
            # Analyze exit precision  
            exit_analysis = self._analyze_real_exit_precision(real_trade_data)
            
            # Analyze CPR effectiveness
            cpr_analysis = self._analyze_cpr_effectiveness(real_trade_data)
            
            # Analyze strike performance
            strike_analysis = self._analyze_real_strike_performance(real_trade_data)
            
            # Analyze database integration performance
            db_analysis = self._analyze_database_integration(real_trade_data)
            
            # Create comprehensive real money analysis
            real_analysis = {
                'trade_id': trade_id,
                'entry_method': entry_method,
                'exit_reason': exit_reason,
                'real_pnl': real_pnl,
                'entry_precision': entry_analysis,
                'exit_precision': exit_analysis,
                'cpr_effectiveness': cpr_analysis,
                'strike_performance': strike_analysis,
                'database_integration': db_analysis,
                'real_money_score': self._calculate_real_money_score(
                    entry_analysis, exit_analysis, cpr_analysis, strike_analysis
                ),
                'accuracy_metrics': {
                    'entry_accuracy': entry_analysis.get('accuracy_score', 0),
                    'exit_accuracy': exit_analysis.get('accuracy_score', 0),
                    'overall_accuracy': (entry_analysis.get('accuracy_score', 0) + exit_analysis.get('accuracy_score', 0)) / 2
                },
                'recommendations': self._generate_real_money_recommendations(real_trade_data)
            }
            
            # Add to real money history
            if self.real_money_history is not None:
                self.real_money_history.append(real_analysis)
            
            # Update real money analysis stats
            self._update_real_money_stats(real_analysis)
            
            self.logger.info(f"Real money exit analysis completed for {trade_id}: "
                           f"Score {real_analysis['real_money_score']:.2f}, "
                           f"Entry Method: {entry_method}, Exit: {exit_reason}")
            
            return real_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing real money exit: {e}")
            return {}
    
    def _analyze_real_entry_precision(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze precision of real money entry execution."""
        try:
            entry_method = trade_data.get('entry_method', 'immediate')
            breakout_delay = trade_data.get('breakout_delay_minutes', 0)
            # Handle both field names for compatibility
            real_entry_premium = trade_data.get('real_entry_premium') or trade_data.get('entry_premium', 0)
            theoretical_premium = trade_data.get('entry_premium', 0)
            
            # Calculate entry precision score
            if theoretical_premium > 0:
                entry_variance = abs(real_entry_premium - theoretical_premium) / theoretical_premium
                accuracy_score = max(0, 1 - entry_variance)
            else:
                accuracy_score = 0
            
            return {
                'entry_method': entry_method,
                'breakout_delay_minutes': breakout_delay,
                'real_entry_premium': real_entry_premium,
                'theoretical_premium': theoretical_premium,
                'variance_percentage': (entry_variance * 100) if theoretical_premium > 0 else 0,
                'accuracy_score': accuracy_score,
                'precision_grade': self._get_precision_grade(accuracy_score),
                'entry_effectiveness': 'High' if entry_method == 'breakout_confirmation' and accuracy_score > 0.8 else 'Medium'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing entry precision: {e}")
            return {}
    
    def _analyze_real_exit_precision(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze precision of real money exit execution."""
        try:
            exit_reason = trade_data.get('exit_reason', 'UNKNOWN')
            exit_premium = trade_data.get('exit_premium', 0)
            sl_tp_levels = trade_data.get('cpr_sl_tp_levels', {})
            
            # Determine expected exit level
            expected_exit = 0
            if 'SL' in exit_reason:
                expected_exit = sl_tp_levels.get('sl_price', 0)
            elif 'TP' in exit_reason:
                tp_levels = sl_tp_levels.get('tp_levels', [])
                expected_exit = tp_levels[0].get('price', 0) if tp_levels else 0
            
            # Calculate exit precision
            if expected_exit > 0:
                exit_variance = abs(exit_premium - expected_exit) / expected_exit
                accuracy_score = max(0, 1 - exit_variance)
            else:
                accuracy_score = 0.5  # Neutral for unknown exits
            
            return {
                'exit_reason': exit_reason,
                'exit_premium': exit_premium,
                'expected_exit_level': expected_exit,
                'variance_percentage': (exit_variance * 100) if expected_exit > 0 else 0,
                'accuracy_score': accuracy_score,
                'precision_grade': self._get_precision_grade(accuracy_score),
                'exit_timing_quality': 'Precise' if accuracy_score > 0.9 else 'Good' if accuracy_score > 0.7 else 'Moderate'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing exit precision: {e}")
            return {}
    
    def _analyze_cpr_effectiveness(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CPR and advanced SL/TP engine effectiveness."""
        try:
            sl_tp_levels = trade_data.get('cpr_sl_tp_levels', {})
            zone_type = trade_data.get('zone_type', 'Unknown')
            exit_reason = trade_data.get('exit_reason', 'UNKNOWN')
            real_pnl = trade_data.get('real_pnl', 0)
            
            # Determine if CPR was used
            cpr_used = 'cpr' in zone_type.lower() or sl_tp_levels.get('sl_method') == 'CPR'
            
            # Analyze CPR effectiveness
            cpr_hit = False
            if cpr_used:
                if 'TP' in exit_reason and real_pnl > 0:
                    cpr_hit = True
                elif 'SL' in exit_reason:
                    cpr_hit = True  # SL also shows CPR working
            
            return {
                'cpr_used': cpr_used,
                'zone_type': zone_type,
                'sl_method': sl_tp_levels.get('sl_method', 'Unknown'),
                'tp_method': sl_tp_levels.get('tp_levels', [{}])[0].get('method', 'Unknown') if sl_tp_levels.get('tp_levels') else 'Unknown',
                'risk_reward_ratio': trade_data.get('risk_reward_ratio', 0),
                'zone_strength': sl_tp_levels.get('zone_strength', 0),
                'cpr_hit': cpr_hit,
                'cpr_effectiveness_score': 1.0 if (cpr_used and cpr_hit and real_pnl > 0) else 0.5 if cpr_used else 0.0,
                'recommendation': 'Continue CPR' if (cpr_used and real_pnl > 0) else 'Review CPR settings' if cpr_used else 'Consider enabling CPR'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing CPR effectiveness: {e}")
            return {}
    
    def _analyze_real_strike_performance(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze real strike selection performance."""
        try:
            strike = trade_data.get('strike', 0)
            underlying_price = trade_data.get('underlying_entry_price', 0)
            signal_type = trade_data.get('signal_type', 'CALL')
            real_pnl = trade_data.get('real_pnl', 0)
            
            # Determine moneyness
            if underlying_price > 0:
                if signal_type == 'CALL':
                    if strike <= underlying_price - 100:
                        moneyness = 'ITM'
                    elif strike >= underlying_price + 100:
                        moneyness = 'OTM'
                    else:
                        moneyness = 'ATM'
                else:  # PUT
                    if strike >= underlying_price + 100:
                        moneyness = 'ITM'
                    elif strike <= underlying_price - 100:
                        moneyness = 'OTM'
                    else:
                        moneyness = 'ATM'
            else:
                moneyness = 'Unknown'
            
            # Calculate strike effectiveness
            strike_distance = abs(strike - underlying_price) if underlying_price > 0 else 0
            
            return {
                'strike': strike,
                'underlying_price': underlying_price,
                'moneyness': moneyness,
                'strike_distance': strike_distance,
                'signal_type': signal_type,
                'real_pnl': real_pnl,
                'strike_effectiveness': 'High' if real_pnl > 0 else 'Low',
                'optimal_strike': moneyness == 'ATM' and real_pnl > 0,
                'strike_recommendation': 'Optimal' if (moneyness == 'ATM' and real_pnl > 0) else 'Review distance'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing strike performance: {e}")
            return {}
    
    def _get_precision_grade(self, accuracy_score: float) -> str:
        """Convert accuracy score to letter grade."""
        if accuracy_score >= 0.95:
            return 'A+'
        elif accuracy_score >= 0.90:
            return 'A'
        elif accuracy_score >= 0.85:
            return 'A-'
        elif accuracy_score >= 0.80:
            return 'B+'
        elif accuracy_score >= 0.75:
            return 'B'
        elif accuracy_score >= 0.70:
            return 'B-'
        elif accuracy_score >= 0.65:
            return 'C+'
        elif accuracy_score >= 0.60:
            return 'C'
        else:
            return 'D'
    
    def _analyze_database_integration(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze database integration performance for real money analysis."""
        try:
            return {
                'database_connected': self.options_db is not None,
                'real_data_queries': trade_data.get('database_queries', 0),
                'data_accuracy': 'High' if self.options_db else 'Simulated',
                'integration_score': 1.0 if self.options_db else 0.5
            }
        except Exception as e:
            self.logger.error(f"Error analyzing database integration: {e}")
            return {'database_connected': False, 'integration_score': 0.0}
    
    def get_real_money_summary(self) -> Dict[str, Any]:
        """Get real money analysis summary."""
        try:
            return {
                'real_money_enabled': self.enable_real_money_analysis,
                'database_integration': self.options_db is not None,
                'analysis_method': 'Real Money' if self.enable_real_money_analysis else 'Simulation',
                'total_real_trades_analyzed': len([h for h in self.exit_history if h.get('real_money_analysis')]),
                'summary': 'Real money analysis with actual strike prices and database integration'
            }
        except Exception as e:
            self.logger.error(f"Error getting real money summary: {e}")
            return {'real_money_enabled': False, 'error': str(e)}
    
    def _calculate_real_money_score(self, entry_analysis: Dict[str, Any], 
                                  exit_analysis: Dict[str, Any],
                                  cpr_analysis: Dict[str, Any], 
                                  strike_analysis: Dict[str, Any]) -> float:
        """Calculate overall real money performance score."""
        try:
            entry_score = entry_analysis.get('accuracy_score', 0) * 0.3
            exit_score = exit_analysis.get('accuracy_score', 0) * 0.3
            cpr_score = cpr_analysis.get('effectiveness_score', 0) * 0.25
            strike_score = strike_analysis.get('performance_score', 0) * 0.15
            
            return min(1.0, entry_score + exit_score + cpr_score + strike_score)
        except Exception as e:
            self.logger.error(f"Error calculating real money score: {e}")
            return 0.0
    
    def _generate_real_money_recommendations(self, trade_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on real money trade analysis."""
        try:
            recommendations = []
            
            # Entry method recommendations
            entry_method = trade_data.get('entry_method', 'immediate')
            real_pnl = trade_data.get('real_pnl', 0)
            
            if entry_method == 'immediate' and real_pnl < 0:
                recommendations.append("Consider using breakout confirmation for better entry timing")
            
            if entry_method == 'breakout_confirmation' and real_pnl > 0:
                recommendations.append("Breakout confirmation strategy working well - continue using")
            
            # CPR recommendations
            zone_type = trade_data.get('zone_type', 'Unknown')
            if 'CPR' in zone_type:
                recommendations.append("CPR-based analysis detected - good for precise SL/TP")
            
            # Strike selection recommendations
            strike = trade_data.get('strike', 0)
            if strike > 0:
                recommendations.append(f"Strike {strike} selected - monitor performance for future reference")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Continue monitoring real money performance for optimization")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating real money recommendations: {e}")
            return ["Error generating recommendations - check logs"]
    
    def _update_real_money_stats(self, real_analysis: Dict[str, Any]) -> None:
        """Update real money statistics tracking."""
        try:
            # Add to exit history with real money flag
            real_analysis['real_money_analysis'] = True
            real_analysis['analysis_timestamp'] = datetime.now()
            
            # Update exit history
            self.exit_history.append(real_analysis)
            
            self.logger.debug(f"Updated real money stats for trade {real_analysis.get('trade_id', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error updating real money stats: {e}")


# Convenience functions
def create_exit_analyzer() -> EnhancedExitAnalyzer:
    """Create enhanced exit analyzer."""
    return EnhancedExitAnalyzer()


def analyze_backtest_exits(trades_data: List[Dict[str, Any]]) -> ExitAnalysisResult:
    """
    Analyze exits from backtest trades data.
    
    Args:
        trades_data: List of trade dictionaries
        
    Returns:
        ExitAnalysisResult with comprehensive analysis
    """
    analyzer = create_exit_analyzer()
    
    # Process each trade
    for trade in trades_data:
        analyzer.analyze_trade_exit(trade)
    
    # Generate comprehensive analysis
    return analyzer.generate_comprehensive_analysis()




# Example usage
if __name__ == "__main__":
    # Example usage of the Enhanced Exit Analyzer
    try:
        # Create analyzer
        analyzer = create_exit_analyzer()
        
        # Example trade data
        sample_trades = [
            {
                'trade_id': 'TRADE_001',
                'exit_reason': 'OPTIONS_SL',
                'pnl': -150.0,
                'entry_time': datetime.now() - timedelta(minutes=30),
                'exit_time': datetime.now() - timedelta(minutes=5),
                'exit_premium': 80.0,
                'entry_premium': 100.0,
                'option_type': 'CALL',
                'sl_tp_levels': {
                    'options_sl': 75.0,
                    'options_tp': 150.0,
                    'equity_sl': 21600,
                    'equity_tp': 21700
                }
            },
            {
                'trade_id': 'TRADE_002',
                'exit_reason': 'EQUITY_TP',
                'pnl': 200.0,
                'entry_time': datetime.now() - timedelta(minutes=45),
                'exit_time': datetime.now() - timedelta(minutes=10),
                'exit_premium': 120.0,
                'entry_premium': 100.0,
                'option_type': 'PUT',
                'sl_tp_levels': {
                    'options_sl': 80.0,
                    'options_tp': 140.0,
                    'equity_sl': 21700,
                    'equity_tp': 21600
                }
            }
        ]
        
        print("Analyzing sample trades...")
        
        # Analyze each trade
        for trade in sample_trades:
            result = analyzer.analyze_trade_exit(trade)
            print(f"Trade {result.get('trade_id')}: {result.get('exit_method')} - {result.get('exit_trigger')}")
        
        # Generate comprehensive analysis
        print("\nGenerating comprehensive analysis...")
        analysis = analyzer.generate_comprehensive_analysis()
        
        # Print summary
        print(analyzer.get_summary_log())
        
        # Export report
        report_path = analyzer.export_analysis_report()
        print(f"\nAnalysis report exported to: {report_path}")
        
    except Exception as e:
        print(f"Error in example usage: {e}")