#!/usr/bin/env python3
"""
Performance Tracker
===================

Real-time performance monitoring and optimization tracking for live trading.
Generates 5-minute interval reports with detailed performance metrics.

Features:
- Real-time processing time tracking
- 5-minute interval optimization reports
- Request/response time monitoring
- System performance statistics
- Memory and CPU usage tracking

Author: vAlgo Development Team
Created: July 2, 2025
Version: 1.0.0 (Production)
"""

import os
import sys
import time
from pathlib import Path

# Optional psutil import for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.logger import get_logger
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: datetime
    operation: str
    processing_time: float
    indicator_count: int
    data_points: int
    memory_usage: float
    cpu_usage: float
    success: bool
    error_message: str = ""


class PerformanceTracker:
    """
    Real-time performance tracking and optimization monitoring.
    Tracks processing times, system resources, and generates optimization reports.
    """
    
    def __init__(self, output_dir: str = "outputs/performance", 
                 report_interval_minutes: int = 5, max_history_hours: int = 24):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger = get_logger(__name__)
        except:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # Performance tracking settings
        self.report_interval_minutes = report_interval_minutes
        self.max_history_hours = max_history_hours
        self.max_metrics = int(max_history_hours * 60 / report_interval_minutes * 100)  # Buffer size
        
        # Performance data storage
        self.metrics_history = deque(maxlen=self.max_metrics)
        self.current_interval_metrics = []
        self.operation_stats = defaultdict(list)
        
        # Threading for automatic reporting
        self.reporting_thread = None
        self.stop_reporting = threading.Event()
        self.last_report_time = datetime.now()
        
        # System baseline (if psutil available)
        if PSUTIL_AVAILABLE:
            self.baseline_memory = psutil.virtual_memory().percent
            self.baseline_cpu = psutil.cpu_percent(interval=1)
        else:
            self.baseline_memory = 50.0  # Default fallback
            self.baseline_cpu = 10.0
        
        self.logger.info(f"[PERF] Performance tracker initialized - {report_interval_minutes}min intervals")
    
    def start_operation(self, operation_name: str) -> Dict[str, Any]:
        """Start tracking a performance operation."""
        return {
            'operation': operation_name,
            'start_time': time.time(),
            'start_timestamp': datetime.now(),
            'start_memory': psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 50.0,
            'start_cpu': psutil.cpu_percent() if PSUTIL_AVAILABLE else 10.0
        }
    
    def end_operation(self, operation_context: Dict[str, Any], 
                     indicator_count: int = 0, data_points: int = 0, 
                     success: bool = True, error_message: str = "") -> PerformanceMetric:
        """End tracking and record performance metric."""
        try:
            end_time = time.time()
            processing_time = end_time - operation_context['start_time']
            
            # Get current system metrics
            if PSUTIL_AVAILABLE:
                current_memory = psutil.virtual_memory().percent
                current_cpu = psutil.cpu_percent()
            else:
                current_memory = 50.0  # Fallback
                current_cpu = 10.0
            
            # Create performance metric
            metric = PerformanceMetric(
                timestamp=operation_context['start_timestamp'],
                operation=operation_context['operation'],
                processing_time=processing_time,
                indicator_count=indicator_count,
                data_points=data_points,
                memory_usage=current_memory,
                cpu_usage=current_cpu,
                success=success,
                error_message=error_message
            )
            
            # Add to tracking
            self.metrics_history.append(metric)
            self.current_interval_metrics.append(metric)
            self.operation_stats[operation_context['operation']].append(metric)
            
            # Log performance if significant
            if processing_time > 1.0 or not success:
                status = "SUCCESS" if success else "FAILED"
                self.logger.info(f"[PERF] {operation_context['operation']} - {status} - "
                               f"{processing_time:.3f}s - {indicator_count} indicators - {data_points} points")
            
            return metric
            
        except Exception as e:
            self.logger.error(f"Failed to end performance operation: {e}")
            return PerformanceMetric(
                timestamp=datetime.now(),
                operation=operation_context.get('operation', 'unknown'),
                processing_time=0.0,
                indicator_count=0,
                data_points=0,
                memory_usage=0.0,
                cpu_usage=0.0,
                success=False,
                error_message=str(e)
            )
    
    def start_automatic_reporting(self) -> None:
        """Start automatic performance reporting thread."""
        try:
            if self.reporting_thread and self.reporting_thread.is_alive():
                self.logger.warning("[PERF] Automatic reporting already running")
                return
            
            self.stop_reporting.clear()
            self.reporting_thread = threading.Thread(target=self._reporting_worker, daemon=True)
            self.reporting_thread.start()
            
            self.logger.info(f"[PERF] Started automatic reporting every {self.report_interval_minutes} minutes")
            
        except Exception as e:
            self.logger.error(f"Failed to start automatic reporting: {e}")
    
    def stop_automatic_reporting(self) -> None:
        """Stop automatic performance reporting."""
        try:
            self.stop_reporting.set()
            if self.reporting_thread:
                self.reporting_thread.join(timeout=10)
            
            # Generate final report
            self.generate_interval_report(force=True)
            
            self.logger.info("[PERF] Stopped automatic reporting")
            
        except Exception as e:
            self.logger.error(f"Failed to stop automatic reporting: {e}")
    
    def _reporting_worker(self) -> None:
        """Worker thread for automatic performance reporting."""
        while not self.stop_reporting.is_set():
            try:
                # Wait for interval or stop signal
                if self.stop_reporting.wait(timeout=self.report_interval_minutes * 60):
                    break  # Stop signal received
                
                # Generate interval report
                self.generate_interval_report()
                
            except Exception as e:
                self.logger.error(f"Reporting worker error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def generate_interval_report(self, force: bool = False) -> str:
        """Generate 5-minute interval performance report."""
        try:
            current_time = datetime.now()
            
            # Check if it's time for a report
            if not force and (current_time - self.last_report_time).total_seconds() < (self.report_interval_minutes * 60 - 30):
                return ""
            
            if not self.current_interval_metrics:
                self.logger.debug("[PERF] No metrics to report in current interval")
                return ""
            
            # Generate report filename
            timestamp_str = current_time.strftime('%Y%m%d_%H%M%S')
            report_filename = f"performance_interval_{timestamp_str}.xlsx"
            report_path = self.output_dir / report_filename
            
            # Create comprehensive report
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # Interval metrics sheet
                interval_df = pd.DataFrame([asdict(metric) for metric in self.current_interval_metrics])
                interval_df.to_excel(writer, sheet_name='Interval_Metrics', index=False)
                
                # Operation summary sheet
                summary_data = self._generate_operation_summary()
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Operation_Summary', index=False)
                
                # System performance sheet
                system_data = self._generate_system_performance()
                system_df = pd.DataFrame([system_data])
                system_df.to_excel(writer, sheet_name='System_Performance', index=False)
                
                # Optimization recommendations sheet
                optimization_data = self._generate_optimization_recommendations()
                if optimization_data:
                    opt_df = pd.DataFrame(optimization_data)
                    opt_df.to_excel(writer, sheet_name='Optimization', index=False)
            
            # Log report generation
            interval_duration = self.report_interval_minutes
            metrics_count = len(self.current_interval_metrics)
            avg_processing_time = sum(m.processing_time for m in self.current_interval_metrics) / metrics_count if metrics_count > 0 else 0
            
            self.logger.info(f"[PERF] Generated {interval_duration}min report: {metrics_count} operations, "
                           f"avg {avg_processing_time:.3f}s - {report_path}")
            
            # Reset interval metrics
            self.current_interval_metrics = []
            self.last_report_time = current_time
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate interval report: {e}")
            return ""
    
    def _generate_operation_summary(self) -> List[Dict[str, Any]]:
        """Generate operation-wise performance summary."""
        try:
            summary_data = []
            
            for operation, metrics in self.operation_stats.items():
                if not metrics:
                    continue
                
                # Filter recent metrics (current interval)
                recent_metrics = [m for m in metrics if m in self.current_interval_metrics]
                if not recent_metrics:
                    continue
                
                processing_times = [m.processing_time for m in recent_metrics]
                success_count = sum(1 for m in recent_metrics if m.success)
                
                summary_data.append({
                    'Operation': operation,
                    'Count': len(recent_metrics),
                    'Success_Rate': (success_count / len(recent_metrics)) * 100 if recent_metrics else 0,
                    'Avg_Processing_Time': sum(processing_times) / len(processing_times) if processing_times else 0,
                    'Min_Processing_Time': min(processing_times) if processing_times else 0,
                    'Max_Processing_Time': max(processing_times) if processing_times else 0,
                    'Total_Processing_Time': sum(processing_times),
                    'Avg_Indicators': sum(m.indicator_count for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                    'Total_Data_Points': sum(m.data_points for m in recent_metrics),
                    'Avg_Memory_Usage': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                    'Avg_CPU_Usage': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
                })
            
            return summary_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate operation summary: {e}")
            return []
    
    def _generate_system_performance(self) -> Dict[str, Any]:
        """Generate current system performance metrics."""
        try:
            if PSUTIL_AVAILABLE:
                current_memory = psutil.virtual_memory()
                current_cpu = psutil.cpu_percent(interval=1)
            else:
                # Create mock memory object
                class MockMemory:
                    percent = 50.0
                    available = 8 * (1024**3)  # 8GB
                current_memory = MockMemory()
                current_cpu = 10.0
            
            # Calculate interval averages
            if self.current_interval_metrics:
                avg_memory = sum(m.memory_usage for m in self.current_interval_metrics) / len(self.current_interval_metrics)
                avg_cpu = sum(m.cpu_usage for m in self.current_interval_metrics) / len(self.current_interval_metrics)
                max_memory = max(m.memory_usage for m in self.current_interval_metrics)
                max_cpu = max(m.cpu_usage for m in self.current_interval_metrics)
            else:
                avg_memory = current_memory.percent if PSUTIL_AVAILABLE else 50.0
                avg_cpu = current_cpu
                max_memory = current_memory.percent if PSUTIL_AVAILABLE else 50.0
                max_cpu = current_cpu
            
            return {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Interval_Minutes': self.report_interval_minutes,
                'Operations_Count': len(self.current_interval_metrics),
                'Current_Memory_Percent': current_memory.percent,
                'Current_Memory_Available_GB': round(current_memory.available / (1024**3), 2),
                'Current_CPU_Percent': current_cpu,
                'Interval_Avg_Memory': round(avg_memory, 2),
                'Interval_Avg_CPU': round(avg_cpu, 2),
                'Interval_Max_Memory': round(max_memory, 2),
                'Interval_Max_CPU': round(max_cpu, 2),
                'Memory_Change_From_Baseline': round(current_memory.percent - self.baseline_memory, 2),
                'CPU_Change_From_Baseline': round(current_cpu - self.baseline_cpu, 2),
                'Total_Metrics_Tracked': len(self.metrics_history)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate system performance: {e}")
            return {}
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on performance analysis."""
        try:
            recommendations = []
            
            if not self.current_interval_metrics:
                return recommendations
            
            # Analyze processing times
            processing_times = [m.processing_time for m in self.current_interval_metrics if m.success]
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                max_time = max(processing_times)
                
                if avg_time > 2.0:
                    recommendations.append({
                        'Category': 'Performance',
                        'Priority': 'High',
                        'Issue': 'High average processing time',
                        'Current_Value': f"{avg_time:.3f}s",
                        'Recommendation': 'Consider optimizing indicator calculations or using ta library',
                        'Impact': 'Reduce latency in live trading'
                    })
                
                if max_time > 10.0:
                    recommendations.append({
                        'Category': 'Performance',
                        'Priority': 'Critical',
                        'Issue': 'Very high maximum processing time',
                        'Current_Value': f"{max_time:.3f}s",
                        'Recommendation': 'Investigate slow operations and optimize algorithms',
                        'Impact': 'Prevent system timeouts and missed trades'
                    })
            
            # Analyze memory usage
            memory_usages = [m.memory_usage for m in self.current_interval_metrics]
            if memory_usages:
                avg_memory = sum(memory_usages) / len(memory_usages)
                max_memory = max(memory_usages)
                
                if avg_memory > 80.0:
                    recommendations.append({
                        'Category': 'Memory',
                        'Priority': 'High',
                        'Issue': 'High memory usage',
                        'Current_Value': f"{avg_memory:.1f}%",
                        'Recommendation': 'Optimize data structures and clear unused DataFrames',
                        'Impact': 'Prevent system slowdown and crashes'
                    })
            
            # Analyze error rates
            failed_operations = [m for m in self.current_interval_metrics if not m.success]
            if failed_operations:
                error_rate = (len(failed_operations) / len(self.current_interval_metrics)) * 100
                
                if error_rate > 5.0:
                    recommendations.append({
                        'Category': 'Reliability',
                        'Priority': 'High',
                        'Issue': 'High error rate',
                        'Current_Value': f"{error_rate:.1f}%",
                        'Recommendation': 'Investigate error causes and add error handling',
                        'Impact': 'Improve system reliability and data quality'
                    })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization recommendations: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary statistics."""
        try:
            current_time = datetime.now()
            
            if self.metrics_history:
                recent_metrics = [m for m in self.metrics_history 
                               if (current_time - m.timestamp).total_seconds() < 3600]  # Last hour
                
                if recent_metrics:
                    avg_processing_time = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
                    success_rate = (sum(1 for m in recent_metrics if m.success) / len(recent_metrics)) * 100
                    avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
                    avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
                else:
                    avg_processing_time = 0.0
                    success_rate = 100.0
                    avg_memory = psutil.virtual_memory().percent
                    avg_cpu = psutil.cpu_percent()
            else:
                avg_processing_time = 0.0
                success_rate = 100.0
                avg_memory = psutil.virtual_memory().percent
                avg_cpu = psutil.cpu_percent()
            
            return {
                'total_operations': len(self.metrics_history),
                'operations_last_hour': len([m for m in self.metrics_history 
                                           if (current_time - m.timestamp).total_seconds() < 3600]),
                'avg_processing_time': round(avg_processing_time, 3),
                'success_rate': round(success_rate, 1),
                'avg_memory_usage': round(avg_memory, 1),
                'avg_cpu_usage': round(avg_cpu, 1),
                'last_report_time': self.last_report_time,
                'next_report_in_minutes': max(0, self.report_interval_minutes - 
                                            (current_time - self.last_report_time).total_seconds() / 60)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def generate_session_performance_report(self) -> str:
        """Generate comprehensive session performance report."""
        try:
            if not self.metrics_history:
                return ""
            
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"session_performance_{timestamp_str}.xlsx"
            report_path = self.output_dir / report_filename
            
            # Create comprehensive session report
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # All metrics sheet
                all_metrics_df = pd.DataFrame([asdict(metric) for metric in self.metrics_history])
                all_metrics_df.to_excel(writer, sheet_name='All_Metrics', index=False)
                
                # Session summary
                session_summary = self._generate_session_summary()
                summary_df = pd.DataFrame([session_summary])
                summary_df.to_excel(writer, sheet_name='Session_Summary', index=False)
                
                # Operation breakdown
                operation_breakdown = self._generate_full_operation_breakdown()
                if operation_breakdown:
                    breakdown_df = pd.DataFrame(operation_breakdown)
                    breakdown_df.to_excel(writer, sheet_name='Operation_Breakdown', index=False)
            
            self.logger.info(f"[PERF] Generated session performance report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate session performance report: {e}")
            return ""
    
    def _generate_session_summary(self) -> Dict[str, Any]:
        """Generate complete session performance summary."""
        try:
            if not self.metrics_history:
                return {}
            
            session_start = min(m.timestamp for m in self.metrics_history)
            session_end = max(m.timestamp for m in self.metrics_history)
            session_duration = (session_end - session_start).total_seconds()
            
            processing_times = [m.processing_time for m in self.metrics_history if m.success]
            success_count = sum(1 for m in self.metrics_history if m.success)
            
            return {
                'Session_Start': session_start.strftime('%Y-%m-%d %H:%M:%S'),
                'Session_End': session_end.strftime('%Y-%m-%d %H:%M:%S'),
                'Session_Duration_Hours': round(session_duration / 3600, 2),
                'Total_Operations': len(self.metrics_history),
                'Successful_Operations': success_count,
                'Success_Rate': (success_count / len(self.metrics_history)) * 100,
                'Avg_Processing_Time': sum(processing_times) / len(processing_times) if processing_times else 0,
                'Min_Processing_Time': min(processing_times) if processing_times else 0,
                'Max_Processing_Time': max(processing_times) if processing_times else 0,
                'Total_Processing_Time': sum(processing_times),
                'Operations_Per_Hour': (len(self.metrics_history) / session_duration * 3600) if session_duration > 0 else 0,
                'Unique_Operations': len(set(m.operation for m in self.metrics_history))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate session summary: {e}")
            return {}
    
    def _generate_full_operation_breakdown(self) -> List[Dict[str, Any]]:
        """Generate full operation breakdown for session report."""
        try:
            breakdown_data = []
            
            for operation, metrics in self.operation_stats.items():
                if not metrics:
                    continue
                
                processing_times = [m.processing_time for m in metrics if m.success]
                success_count = sum(1 for m in metrics if m.success)
                
                breakdown_data.append({
                    'Operation': operation,
                    'Total_Count': len(metrics),
                    'Success_Count': success_count,
                    'Success_Rate': (success_count / len(metrics)) * 100 if metrics else 0,
                    'Avg_Processing_Time': sum(processing_times) / len(processing_times) if processing_times else 0,
                    'Min_Processing_Time': min(processing_times) if processing_times else 0,
                    'Max_Processing_Time': max(processing_times) if processing_times else 0,
                    'Total_Processing_Time': sum(processing_times),
                    'Avg_Indicators': sum(m.indicator_count for m in metrics) / len(metrics) if metrics else 0,
                    'Total_Data_Points': sum(m.data_points for m in metrics),
                    'First_Execution': min(m.timestamp for m in metrics).strftime('%Y-%m-%d %H:%M:%S'),
                    'Last_Execution': max(m.timestamp for m in metrics).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return breakdown_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate operation breakdown: {e}")
            return []