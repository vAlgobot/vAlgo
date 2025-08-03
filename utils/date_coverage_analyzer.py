#!/usr/bin/env python3
"""
Date Coverage Analyzer for vAlgo Trading System
===============================================

Comprehensive utility to analyze and compare date coverage between
options chain data and OHLC instrument data to ensure data alignment
and identify any missing periods.

Features:
- Compare date coverage between options and OHLC data
- Identify missing dates in either dataset
- Generate detailed coverage reports
- Analyze data quality and gaps
- Export analysis results

Author: vAlgo Development Team
Created: July 12, 2025
Version: 1.0.0 (Production)
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from data_manager.options_database import OptionsDatabase, create_options_database
from data_manager.database import DatabaseManager


class DateCoverageAnalyzer:
    """
    Comprehensive date coverage analyzer for options and OHLC data alignment.
    
    Analyzes date coverage between options chain data and OHLC instrument data
    to ensure complete data alignment and identify any missing periods.
    """
    
    def __init__(self, options_db: Optional[OptionsDatabase] = None,
                 ohlc_db: Optional[DatabaseManager] = None):
        """
        Initialize Date Coverage Analyzer.
        
        Args:
            options_db: Optional OptionsDatabase instance
            ohlc_db: Optional DatabaseManager instance
        """
        self.logger = get_logger(__name__)
        
        # Initialize database connections
        self.options_db = options_db or create_options_database()
        self.ohlc_db = ohlc_db or DatabaseManager()
        
        self.logger.info("DateCoverageAnalyzer initialized successfully")
    
    def analyze_complete_coverage(self, symbol: str = "NIFTY") -> Dict[str, Any]:
        """
        Analyze complete date coverage between options and OHLC data.
        
        Args:
            symbol: Symbol to analyze (default: NIFTY)
            
        Returns:
            Comprehensive coverage analysis report
        """
        try:
            self.logger.info(f"Starting complete date coverage analysis for {symbol}")
            
            # Get dates from both datasets
            options_dates = set(self.options_db.get_distinct_dates("trading"))
            ohlc_dates = set(self.ohlc_db.get_available_dates(symbol=symbol))
            
            # Perform comparison analysis
            comparison_result = self._compare_date_sets(options_dates, ohlc_dates, symbol)
            
            # Get detailed coverage stats
            options_coverage = self.options_db.get_date_coverage_stats()
            ohlc_coverage = self.ohlc_db.get_ohlc_date_coverage_stats(symbol=symbol)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(comparison_result, options_coverage, ohlc_coverage)
            
            # Compile comprehensive report
            report = {
                "analysis_info": {
                    "symbol": symbol,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analyzer_version": "1.0.0"
                },
                "date_comparison": comparison_result,
                "options_coverage": options_coverage,
                "ohlc_coverage": ohlc_coverage,
                "recommendations": recommendations,
                "summary": self._generate_summary(comparison_result, options_coverage, ohlc_coverage)
            }
            
            self.logger.info(f"Complete coverage analysis completed for {symbol}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error in complete coverage analysis: {e}")
            return {"error": str(e)}
    
    def _compare_date_sets(self, options_dates: Set[str], ohlc_dates: Set[str], 
                          symbol: str) -> Dict[str, Any]:
        """Compare two sets of dates and analyze differences."""
        try:
            # Find overlaps and differences
            common_dates = options_dates.intersection(ohlc_dates)
            options_only = options_dates - ohlc_dates
            ohlc_only = ohlc_dates - options_dates
            
            # Sort for analysis
            options_sorted = sorted(list(options_dates))
            ohlc_sorted = sorted(list(ohlc_dates))
            common_sorted = sorted(list(common_dates))
            
            # Calculate coverage percentages
            total_unique_dates = len(options_dates.union(ohlc_dates))
            options_coverage_pct = (len(options_dates) / total_unique_dates * 100) if total_unique_dates > 0 else 0
            ohlc_coverage_pct = (len(ohlc_dates) / total_unique_dates * 100) if total_unique_dates > 0 else 0
            overlap_pct = (len(common_dates) / total_unique_dates * 100) if total_unique_dates > 0 else 0
            
            return {
                "date_counts": {
                    "options_dates": len(options_dates),
                    "ohlc_dates": len(ohlc_dates),
                    "common_dates": len(common_dates),
                    "total_unique_dates": total_unique_dates
                },
                "date_ranges": {
                    "options_range": {
                        "start": options_sorted[0] if options_sorted else None,
                        "end": options_sorted[-1] if options_sorted else None
                    },
                    "ohlc_range": {
                        "start": ohlc_sorted[0] if ohlc_sorted else None,
                        "end": ohlc_sorted[-1] if ohlc_sorted else None
                    },
                    "common_range": {
                        "start": common_sorted[0] if common_sorted else None,
                        "end": common_sorted[-1] if common_sorted else None
                    }
                },
                "coverage_percentages": {
                    "options_coverage": round(options_coverage_pct, 2),
                    "ohlc_coverage": round(ohlc_coverage_pct, 2),
                    "overlap_coverage": round(overlap_pct, 2)
                },
                "missing_data": {
                    "options_missing": {
                        "count": len(ohlc_only),
                        "dates": sorted(list(ohlc_only))[:20],  # First 20 for brevity
                        "sample_missing": sorted(list(ohlc_only))[:5] if ohlc_only else []
                    },
                    "ohlc_missing": {
                        "count": len(options_only),
                        "dates": sorted(list(options_only))[:20],  # First 20 for brevity
                        "sample_missing": sorted(list(options_only))[:5] if options_only else []
                    }
                },
                "alignment_quality": self._calculate_alignment_quality(options_dates, ohlc_dates, common_dates)
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing date sets: {e}")
            return {"error": str(e)}
    
    def _calculate_alignment_quality(self, options_dates: Set[str], ohlc_dates: Set[str], 
                                   common_dates: Set[str]) -> Dict[str, Any]:
        """Calculate alignment quality metrics."""
        try:
            total_dates = len(options_dates.union(ohlc_dates))
            
            if total_dates == 0:
                return {"score": 0, "grade": "F", "status": "No data"}
            
            # Calculate alignment score (percentage of dates that are common)
            alignment_score = (len(common_dates) / total_dates) * 100
            
            # Determine grade
            if alignment_score >= 95:
                grade = "A+"
                status = "Excellent alignment"
            elif alignment_score >= 90:
                grade = "A"
                status = "Very good alignment"
            elif alignment_score >= 85:
                grade = "A-"
                status = "Good alignment"
            elif alignment_score >= 80:
                grade = "B+"
                status = "Acceptable alignment"
            elif alignment_score >= 75:
                grade = "B"
                status = "Fair alignment"
            elif alignment_score >= 70:
                grade = "B-"
                status = "Below average alignment"
            elif alignment_score >= 60:
                grade = "C"
                status = "Poor alignment"
            else:
                grade = "D/F"
                status = "Very poor alignment"
            
            return {
                "score": round(alignment_score, 2),
                "grade": grade,
                "status": status,
                "total_dates_analyzed": total_dates,
                "common_dates_count": len(common_dates)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating alignment quality: {e}")
            return {"score": 0, "grade": "Error", "status": str(e)}
    
    def _generate_recommendations(self, comparison_result: Dict[str, Any],
                                options_coverage: Dict[str, Any],
                                ohlc_coverage: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        try:
            recommendations = []
            
            # Check alignment quality
            alignment = comparison_result.get("alignment_quality", {})
            score = alignment.get("score", 0)
            
            if score < 85:
                recommendations.append(f"⚠️  Date alignment is {alignment.get('status', 'poor')} ({score:.1f}%) - investigate data loading procedures")
            
            # Check for missing data
            missing_data = comparison_result.get("missing_data", {})
            options_missing = missing_data.get("options_missing", {}).get("count", 0)
            ohlc_missing = missing_data.get("ohlc_missing", {}).get("count", 0)
            
            if options_missing > 0:
                recommendations.append(f"📊 {options_missing} dates are missing in options data - consider loading additional option chain data")
            
            if ohlc_missing > 0:
                recommendations.append(f"📈 {ohlc_missing} dates are missing in OHLC data - consider loading additional market data")
            
            # Check data quality
            options_gaps = len(options_coverage.get("trading_dates", {}).get("date_gaps", []))
            ohlc_gaps = len(ohlc_coverage.get("trading_dates", {}).get("date_gaps", []))
            
            if options_gaps > 5:
                recommendations.append(f"🔍 Options data has {options_gaps} significant date gaps - review data continuity")
            
            if ohlc_gaps > 5:
                recommendations.append(f"🔍 OHLC data has {ohlc_gaps} significant date gaps - review data continuity")
            
            # Data density recommendations
            options_avg_records = options_coverage.get("data_density", {}).get("avg_records_per_day", 0)
            if options_avg_records < 1000:
                recommendations.append(f"📉 Low options data density ({options_avg_records:.0f} records/day) - consider data quality review")
            
            # Success message
            if score >= 95 and options_missing == 0 and ohlc_missing == 0:
                recommendations.append("✅ Excellent data alignment - both datasets are well synchronized")
            
            # Add default recommendation if none generated
            if not recommendations:
                recommendations.append("📋 Data analysis completed - no major issues detected")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return [f"❌ Error generating recommendations: {str(e)}"]
    
    def _generate_summary(self, comparison_result: Dict[str, Any],
                         options_coverage: Dict[str, Any],
                         ohlc_coverage: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the analysis."""
        try:
            date_counts = comparison_result.get("date_counts", {})
            alignment = comparison_result.get("alignment_quality", {})
            
            return {
                "overall_status": alignment.get("status", "Unknown"),
                "alignment_grade": alignment.get("grade", "Unknown"),
                "alignment_score": alignment.get("score", 0),
                "total_trading_days": date_counts.get("total_unique_dates", 0),
                "common_trading_days": date_counts.get("common_dates", 0),
                "options_data_days": date_counts.get("options_dates", 0),
                "ohlc_data_days": date_counts.get("ohlc_dates", 0),
                "data_completeness": {
                    "options_records": options_coverage.get("trading_dates", {}).get("total_dates", 0),
                    "ohlc_records": ohlc_coverage.get("trading_dates", {}).get("total_dates", 0)
                },
                "key_insight": self._get_key_insight(comparison_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
    
    def _get_key_insight(self, comparison_result: Dict[str, Any]) -> str:
        """Generate key insight from the analysis."""
        try:
            alignment_score = comparison_result.get("alignment_quality", {}).get("score", 0)
            missing_data = comparison_result.get("missing_data", {})
            options_missing = missing_data.get("options_missing", {}).get("count", 0)
            ohlc_missing = missing_data.get("ohlc_missing", {}).get("count", 0)
            
            if alignment_score >= 95 and options_missing == 0 and ohlc_missing == 0:
                return "Perfect data alignment achieved - both datasets are completely synchronized"
            elif alignment_score >= 90:
                return "Excellent data alignment with minor gaps that don't affect trading operations"
            elif alignment_score >= 80:
                return "Good data alignment with some gaps that should be reviewed for trading accuracy"
            elif options_missing > ohlc_missing:
                return f"Options data is missing {options_missing} trading days - primary focus should be option chain data loading"
            elif ohlc_missing > options_missing:
                return f"OHLC data is missing {ohlc_missing} trading days - primary focus should be market data loading"
            else:
                return "Significant data alignment issues detected - comprehensive data review recommended"
                
        except Exception:
            return "Data analysis completed - review detailed results for insights"
    
    def export_analysis_report(self, analysis_result: Dict[str, Any], 
                             output_path: Optional[str] = None) -> str:
        """
        Export analysis report to JSON file.
        
        Args:
            analysis_result: Analysis result from analyze_complete_coverage
            output_path: Optional output file path
            
        Returns:
            Path to exported file
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"outputs/analysis/date_coverage_analysis_{timestamp}.json"
            
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to JSON
            with open(output_file, 'w') as f:
                json.dump(analysis_result, f, indent=2, default=str)
            
            self.logger.info(f"Analysis report exported to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis report: {e}")
            return ""
    
    def generate_coverage_report_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Generate human-readable summary report."""
        try:
            summary = analysis_result.get("summary", {})
            recommendations = analysis_result.get("recommendations", [])
            
            report_lines = [
                "=" * 80,
                "DATE COVERAGE ANALYSIS REPORT",
                "=" * 80,
                f"Analysis Date: {analysis_result.get('analysis_info', {}).get('analysis_timestamp', 'Unknown')}",
                f"Symbol Analyzed: {analysis_result.get('analysis_info', {}).get('symbol', 'Unknown')}",
                "",
                "EXECUTIVE SUMMARY:",
                f"  Overall Status: {summary.get('overall_status', 'Unknown')}",
                f"  Alignment Grade: {summary.get('alignment_grade', 'Unknown')} ({summary.get('alignment_score', 0):.1f}%)",
                f"  Total Trading Days: {summary.get('total_trading_days', 0):,}",
                f"  Common Days: {summary.get('common_trading_days', 0):,}",
                f"  Options Data Days: {summary.get('options_data_days', 0):,}",
                f"  OHLC Data Days: {summary.get('ohlc_data_days', 0):,}",
                "",
                "KEY INSIGHT:",
                f"  {summary.get('key_insight', 'No insight available')}",
                "",
                "RECOMMENDATIONS:",
            ]
            
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"  {i}. {rec}")
            
            report_lines.extend([
                "",
                "=" * 80,
                "For detailed analysis, refer to the exported JSON report.",
                "=" * 80
            ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return f"Error generating summary report: {str(e)}"


# Convenience functions
def analyze_date_coverage(symbol: str = "NIFTY") -> Dict[str, Any]:
    """
    Convenience function to perform complete date coverage analysis.
    
    Args:
        symbol: Symbol to analyze
        
    Returns:
        Complete analysis results
    """
    analyzer = DateCoverageAnalyzer()
    return analyzer.analyze_complete_coverage(symbol)


def quick_coverage_check(symbol: str = "NIFTY") -> str:
    """
    Quick coverage check with summary report.
    
    Args:
        symbol: Symbol to analyze
        
    Returns:
        Human-readable summary report
    """
    analyzer = DateCoverageAnalyzer()
    result = analyzer.analyze_complete_coverage(symbol)
    return analyzer.generate_coverage_report_summary(result)


# Example usage
if __name__ == "__main__":
    try:
        print("Starting Date Coverage Analysis...")
        
        # Create analyzer
        analyzer = DateCoverageAnalyzer()
        
        # Run complete analysis
        symbol = "NIFTY"
        result = analyzer.analyze_complete_coverage(symbol)
        
        # Generate and print summary
        summary_report = analyzer.generate_coverage_report_summary(result)
        print(summary_report)
        
        # Export detailed report
        export_path = analyzer.export_analysis_report(result)
        if export_path:
            print(f"\nDetailed analysis exported to: {export_path}")
        
    except Exception as e:
        print(f"Error in date coverage analysis: {e}")