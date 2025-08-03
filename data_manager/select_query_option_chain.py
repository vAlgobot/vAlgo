#!/usr/bin/env python3
"""
Option Chain Diagnostic Script for vAlgo Trading System
======================================================

Comprehensive diagnostic tool for testing option premium queries and 
identifying data quality issues in the options database.

Features:
- Test specific option queries with detailed analysis
- Data quality assessment and anomaly detection
- Nearby data exploration with time/strike tolerance
- Database statistics and health checks
- Interactive testing mode for manual exploration
- Command-line interface for automated testing

Usage Examples:
    # Test specific failing case
    python select_query_option_chain.py --strike 24700 --timestamp "2025-06-06 09:45:00" --type PUT
    
    # Interactive exploration mode
    python select_query_option_chain.py --interactive
    
    # Database overview and stats
    python select_query_option_chain.py --stats
    
    # Test with time tolerance
    python select_query_option_chain.py --strike 24700 --timestamp "2025-06-06 09:45:00" --tolerance 5

Author: vAlgo Development Team
Created: July 22, 2025
Version: 1.0.0 (Production)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import duckdb
    import pandas as pd
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Please install: pip install duckdb pandas")
    sys.exit(1)

from utils.logger import get_logger


class OptionChainDiagnostic:
    """
    Comprehensive diagnostic tool for option chain database analysis.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize diagnostic tool.
        
        Args:
            db_path: Optional path to database file
        """
        self.logger = get_logger(__name__)
        
        # Setup database path
        if db_path is None:
            data_dir = project_root / "data"
            db_path = str(data_dir / "vAlgo_market_data.db")
        
        self.db_path = db_path
        self.table_name = "nifty_expired_option_chain"
        self.connection = None
        
        # Initialize connection
        self._connect()
    
    def _connect(self) -> bool:
        """Establish database connection."""
        try:
            if not Path(self.db_path).exists():
                print(f"❌ Database not found at: {self.db_path}")
                return False
            
            self.connection = duckdb.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def test_specific_option(self, strike: int, timestamp: str, option_type: str = 'PUT', 
                           show_details: bool = True) -> Dict[str, Any]:
        """
        Test specific option query and provide detailed analysis.
        
        Args:
            strike: Strike price to test
            timestamp: Timestamp in YYYY-MM-DD HH:MM:SS format
            option_type: 'CALL' or 'PUT'
            show_details: Whether to print detailed analysis
            
        Returns:
            Dictionary with analysis results
        """
        if not self.connection:
            return {"error": "Database not connected"}
        
        results = {
            "strike": strike,
            "timestamp": timestamp,
            "option_type": option_type,
            "analysis": {}
        }
        
        if show_details:
            print(f"\n{'='*70}")
            print(f"DIAGNOSTIC ANALYSIS: {option_type} STRIKE {strike} AT {timestamp}")
            print(f"{'='*70}")
        
        try:
            # Step 1: Check for exact record
            query_exact = f"""
                SELECT COUNT(*) as record_count
                FROM {self.table_name}
                WHERE strike = ? AND timestamp = ?
            """
            exact_count = self.connection.execute(query_exact, [strike, timestamp]).fetchone()[0]
            results["analysis"]["exact_match"] = exact_count > 0
            
            if show_details:
                print(f"1. EXACT MATCH CHECK")
                print(f"   Records found for strike {strike} at {timestamp}: {exact_count}")
            
            if exact_count == 0:
                # No exact match - find nearby data
                nearby_data = self._find_nearby_timestamps(strike, timestamp, show_details)
                results["analysis"]["nearby_data"] = nearby_data
                return results
            
            # Step 2: Get full record details
            ltp_column = f"{option_type.lower()}_ltp"
            iv_column = f"{option_type.lower()}_iv"
            delta_column = f"{option_type.lower()}_delta"
            
            query_details = f"""
                SELECT 
                    timestamp,
                    strike,
                    expiry_date,
                    {ltp_column} as target_ltp,
                    call_ltp,
                    put_ltp,
                    {iv_column} as target_iv,
                    {delta_column} as target_delta,
                    dte
                FROM {self.table_name}
                WHERE strike = ? AND timestamp = ?
                ORDER BY expiry_date DESC
                LIMIT 1
            """
            
            record = self.connection.execute(query_details, [strike, timestamp]).fetchone()
            
            if record:
                target_premium = record[3]  # target_ltp
                results["analysis"]["record_found"] = True
                results["analysis"]["premium_value"] = target_premium
                results["analysis"]["expiry_date"] = record[2]
                results["analysis"]["dte"] = record[8]
                
                if show_details:
                    print(f"\n2. RECORD DETAILS")
                    print(f"   Timestamp: {record[0]}")
                    print(f"   Strike: {record[1]}")
                    print(f"   Expiry Date: {record[2]}")
                    print(f"   {option_type} Premium: {target_premium}")
                    print(f"   CALL Premium: {record[4]}")
                    print(f"   PUT Premium: {record[5]}")
                    print(f"   {option_type} IV: {record[6]}")
                    print(f"   {option_type} Delta: {record[7]}")
                    print(f"   DTE: {record[8]}")
                
                # Step 3: Premium validation analysis
                if target_premium is None:
                    results["analysis"]["issue"] = "NULL_PREMIUM"
                    if show_details:
                        print(f"\n❌ ISSUE IDENTIFIED: {option_type} premium is NULL")
                        print(f"   This would cause premium lookup to fail")
                elif target_premium == 0:
                    results["analysis"]["issue"] = "ZERO_PREMIUM" 
                    if show_details:
                        print(f"\n⚠️  ISSUE IDENTIFIED: {option_type} premium is 0")
                        print(f"   This explains the 'Invalid premium 0' error")
                        print(f"   Could be legitimate (deep OTM) or data quality issue")
                elif target_premium < 0:
                    results["analysis"]["issue"] = "NEGATIVE_PREMIUM"
                    if show_details:
                        print(f"\n❌ ISSUE IDENTIFIED: {option_type} premium is negative ({target_premium})")
                        print(f"   This indicates a data quality problem")
                else:
                    results["analysis"]["issue"] = None
                    if show_details:
                        print(f"\n✅ {option_type} premium looks valid: {target_premium}")
                
                # Step 4: Data quality analysis for this strike/date
                self._analyze_strike_data_quality(strike, timestamp, show_details)
                
                # Step 5: Nearby strikes comparison
                self._analyze_nearby_strikes(timestamp, strike, show_details)
                
            else:
                results["analysis"]["record_found"] = False
                if show_details:
                    print(f"❌ No records found despite count > 0 (unexpected)")
            
        except Exception as e:
            results["error"] = str(e)
            if show_details:
                print(f"❌ Error during analysis: {e}")
        
        return results
    
    def _find_nearby_timestamps(self, strike: int, target_timestamp: str, show_details: bool = True) -> List[Dict]:
        """Find nearby timestamps for the given strike."""
        try:
            query_nearby = f"""
                SELECT 
                    timestamp, 
                    strike, 
                    put_ltp, 
                    call_ltp, 
                    expiry_date,
                    ABS(EXTRACT(EPOCH FROM (timestamp - ?::timestamp))) as time_diff_seconds
                FROM {self.table_name}
                WHERE strike = ? AND DATE(timestamp) = DATE(?)
                ORDER BY time_diff_seconds 
                LIMIT 10
            """
            
            nearby_records = self.connection.execute(
                query_nearby, [target_timestamp, strike, target_timestamp]
            ).fetchall()
            
            nearby_data = []
            if show_details and nearby_records:
                print(f"\n   NEARBY TIMESTAMPS FOR STRIKE {strike}:")
            
            for record in nearby_records:
                nearby_data.append({
                    "timestamp": str(record[0]),
                    "strike": record[1],
                    "put_ltp": record[2],
                    "call_ltp": record[3],
                    "expiry_date": str(record[4]),
                    "time_diff_minutes": round(record[5] / 60, 1)
                })
                
                if show_details:
                    print(f"   {record[0]} | PUT: {record[2]} | CALL: {record[3]} | "
                          f"Expiry: {record[4]} | Diff: {round(record[5]/60, 1)}min")
            
            return nearby_data
            
        except Exception as e:
            if show_details:
                print(f"   Error finding nearby timestamps: {e}")
            return []
    
    def _analyze_strike_data_quality(self, strike: int, timestamp: str, show_details: bool = True) -> None:
        """Analyze data quality for the given strike on the same date."""
        try:
            query_quality = f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN put_ltp > 0 THEN 1 END) as valid_put_ltp,
                    COUNT(CASE WHEN put_ltp = 0 THEN 1 END) as zero_put_ltp,
                    COUNT(CASE WHEN put_ltp IS NULL THEN 1 END) as null_put_ltp,
                    COUNT(CASE WHEN call_ltp > 0 THEN 1 END) as valid_call_ltp,
                    COUNT(CASE WHEN call_ltp = 0 THEN 1 END) as zero_call_ltp,
                    MIN(put_ltp) as min_put_ltp,
                    MAX(put_ltp) as max_put_ltp,
                    AVG(put_ltp) as avg_put_ltp,
                    MIN(call_ltp) as min_call_ltp,
                    MAX(call_ltp) as max_call_ltp,
                    AVG(call_ltp) as avg_call_ltp
                FROM {self.table_name}
                WHERE strike = ? AND DATE(timestamp) = DATE(?)
            """
            
            quality_stats = self.connection.execute(query_quality, [strike, timestamp]).fetchone()
            
            if show_details and quality_stats:
                print(f"\n3. DATA QUALITY ANALYSIS (Strike {strike} on {timestamp[:10]})")
                print(f"   Total records: {quality_stats[0]}")
                print(f"   PUT Premium: Valid={quality_stats[1]}, Zero={quality_stats[2]}, NULL={quality_stats[3]}")
                print(f"   PUT Range: {quality_stats[6]:.2f} to {quality_stats[7]:.2f} (avg: {quality_stats[8]:.2f})")
                print(f"   CALL Premium: Valid={quality_stats[4]}, Zero={quality_stats[5]}, NULL=0")
                print(f"   CALL Range: {quality_stats[9]:.2f} to {quality_stats[10]:.2f} (avg: {quality_stats[11]:.2f})")
                
                # Data quality assessment
                total_records = quality_stats[0]
                valid_put_ratio = quality_stats[1] / total_records if total_records > 0 else 0
                zero_put_ratio = quality_stats[2] / total_records if total_records > 0 else 0
                
                if zero_put_ratio > 0.5:
                    print(f"   ⚠️  WARNING: {zero_put_ratio*100:.1f}% of PUT premiums are zero")
                elif valid_put_ratio > 0.8:
                    print(f"   ✅ Good data quality: {valid_put_ratio*100:.1f}% valid PUT premiums")
                    
        except Exception as e:
            if show_details:
                print(f"   Error analyzing data quality: {e}")
    
    def _analyze_nearby_strikes(self, timestamp: str, target_strike: int, show_details: bool = True) -> None:
        """Analyze nearby strikes at the same timestamp."""
        try:
            query_strikes = f"""
                SELECT 
                    strike, 
                    put_ltp, 
                    call_ltp, 
                    expiry_date,
                    ABS(strike - ?) as strike_distance
                FROM {self.table_name}
                WHERE timestamp = ? AND (put_ltp > 0 OR call_ltp > 0)
                ORDER BY strike_distance 
                LIMIT 15
            """
            
            nearby_strikes = self.connection.execute(query_strikes, [target_strike, timestamp]).fetchall()
            
            if show_details and nearby_strikes:
                print(f"\n4. NEARBY STRIKES AT {timestamp}")
                print("   Strike | PUT Premium | CALL Premium | Expiry | Distance")
                print("   " + "-" * 55)
                
                for record in nearby_strikes:
                    strike_dist = "TARGET" if record[4] == 0 else f"{int(record[4])}"
                    print(f"   {record[0]:6d} | {record[1]:11.2f} | {record[2]:12.2f} | {record[3]} | {strike_dist}")
                    
        except Exception as e:
            if show_details:
                print(f"   Error analyzing nearby strikes: {e}")
    
    def find_nearby_data(self, strike: int, timestamp: str, tolerance_minutes: int = 5) -> Dict[str, Any]:
        """
        Find data within time tolerance of target timestamp.
        
        Args:
            strike: Strike price
            timestamp: Target timestamp
            tolerance_minutes: Time tolerance in minutes
            
        Returns:
            Dictionary with nearby data results
        """
        if not self.connection:
            return {"error": "Database not connected"}
        
        try:
            tolerance_seconds = tolerance_minutes * 60
            
            query_tolerance = f"""
                SELECT 
                    timestamp,
                    strike,
                    put_ltp,
                    call_ltp,
                    expiry_date,
                    ABS(EXTRACT(EPOCH FROM (timestamp - ?::timestamp))) as time_diff_seconds
                FROM {self.table_name}
                WHERE strike = ? 
                AND ABS(EXTRACT(EPOCH FROM (timestamp - ?::timestamp))) <= ?
                ORDER BY time_diff_seconds
                LIMIT 20
            """
            
            results = self.connection.execute(
                query_tolerance, [timestamp, strike, timestamp, tolerance_seconds]
            ).fetchall()
            
            nearby_data = []
            for record in results:
                nearby_data.append({
                    "timestamp": str(record[0]),
                    "strike": record[1],
                    "put_ltp": record[2],
                    "call_ltp": record[3],
                    "expiry_date": str(record[4]),
                    "time_diff_minutes": round(record[5] / 60, 1)
                })
            
            return {
                "target_strike": strike,
                "target_timestamp": timestamp,
                "tolerance_minutes": tolerance_minutes,
                "found_records": len(nearby_data),
                "data": nearby_data
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        if not self.connection:
            return {"error": "Database not connected"}
        
        try:
            stats = {}
            
            # Basic statistics
            basic_query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT strike) as unique_strikes,
                    COUNT(DISTINCT DATE(timestamp)) as trading_days,
                    COUNT(DISTINCT expiry_date) as unique_expiries,
                    MIN(timestamp) as earliest_timestamp,
                    MAX(timestamp) as latest_timestamp
                FROM {self.table_name}
            """
            
            basic_stats = self.connection.execute(basic_query).fetchone()
            stats["basic"] = {
                "total_records": basic_stats[0],
                "unique_strikes": basic_stats[1],
                "trading_days": basic_stats[2],
                "unique_expiries": basic_stats[3],
                "date_range": f"{basic_stats[4]} to {basic_stats[5]}"
            }
            
            # Data quality statistics
            quality_query = f"""
                SELECT 
                    COUNT(CASE WHEN put_ltp > 0 THEN 1 END) as valid_put_premiums,
                    COUNT(CASE WHEN put_ltp = 0 THEN 1 END) as zero_put_premiums,
                    COUNT(CASE WHEN put_ltp IS NULL THEN 1 END) as null_put_premiums,
                    COUNT(CASE WHEN call_ltp > 0 THEN 1 END) as valid_call_premiums,
                    COUNT(CASE WHEN call_ltp = 0 THEN 1 END) as zero_call_premiums,
                    COUNT(CASE WHEN call_ltp IS NULL THEN 1 END) as null_call_premiums,
                    AVG(put_ltp) as avg_put_premium,
                    AVG(call_ltp) as avg_call_premium
                FROM {self.table_name}
            """
            
            quality_stats = self.connection.execute(quality_query).fetchone()
            total_records = stats["basic"]["total_records"]
            
            stats["data_quality"] = {
                "put_premiums": {
                    "valid": quality_stats[0],
                    "zero": quality_stats[1],
                    "null": quality_stats[2],
                    "valid_percentage": (quality_stats[0] / total_records * 100) if total_records > 0 else 0
                },
                "call_premiums": {
                    "valid": quality_stats[3],
                    "zero": quality_stats[4], 
                    "null": quality_stats[5],
                    "valid_percentage": (quality_stats[3] / total_records * 100) if total_records > 0 else 0
                },
                "average_premiums": {
                    "put": round(quality_stats[6], 2) if quality_stats[6] else 0,
                    "call": round(quality_stats[7], 2) if quality_stats[7] else 0
                }
            }
            
            # Strike distribution
            strike_query = f"""
                SELECT 
                    MIN(strike) as min_strike,
                    MAX(strike) as max_strike,
                    COUNT(DISTINCT strike) as strike_count
                FROM {self.table_name}
            """
            
            strike_stats = self.connection.execute(strike_query).fetchone()
            stats["strikes"] = {
                "range": f"{strike_stats[0]} to {strike_stats[1]}",
                "count": strike_stats[2],
                "step": (strike_stats[1] - strike_stats[0]) // (strike_stats[2] - 1) if strike_stats[2] > 1 else 0
            }
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def print_database_stats(self) -> None:
        """Print formatted database statistics."""
        stats = self.get_database_stats()
        
        if "error" in stats:
            print(f"❌ Error getting stats: {stats['error']}")
            return
        
        print(f"\n{'='*60}")
        print("DATABASE STATISTICS")
        print(f"{'='*60}")
        
        # Basic stats
        basic = stats["basic"]
        print(f"Total Records: {basic['total_records']:,}")
        print(f"Unique Strikes: {basic['unique_strikes']}")
        print(f"Trading Days: {basic['trading_days']}")
        print(f"Unique Expiries: {basic['unique_expiries']}")
        print(f"Date Range: {basic['date_range']}")
        
        # Data quality
        quality = stats["data_quality"]
        print(f"\nDATA QUALITY:")
        print(f"PUT Premiums: {quality['put_premiums']['valid']:,} valid "
              f"({quality['put_premiums']['valid_percentage']:.1f}%), "
              f"{quality['put_premiums']['zero']:,} zero, "
              f"{quality['put_premiums']['null']:,} null")
        print(f"CALL Premiums: {quality['call_premiums']['valid']:,} valid "
              f"({quality['call_premiums']['valid_percentage']:.1f}%), "
              f"{quality['call_premiums']['zero']:,} zero, "
              f"{quality['call_premiums']['null']:,} null")
        print(f"Average Premiums: PUT={quality['average_premiums']['put']}, "
              f"CALL={quality['average_premiums']['call']}")
        
        # Strike info
        strikes = stats["strikes"]
        print(f"\nSTRIKES:")
        print(f"Range: {strikes['range']}")
        print(f"Count: {strikes['count']}")
        print(f"Estimated Step: {strikes['step']}")
    
    def interactive_mode(self) -> None:
        """Interactive testing mode."""
        print(f"\n{'='*60}")
        print("INTERACTIVE OPTION CHAIN DIAGNOSTIC MODE")
        print(f"{'='*60}")
        print("Commands:")
        print("  test <strike> <timestamp> [PUT|CALL] - Test specific option")
        print("  nearby <strike> <timestamp> [tolerance_minutes] - Find nearby data")
        print("  stats - Show database statistics")
        print("  help - Show this help")
        print("  quit - Exit interactive mode")
        print()
        
        while True:
            try:
                command = input("diagnostic> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    break
                
                elif cmd == 'help':
                    print("Available commands:")
                    print("  test <strike> <timestamp> [PUT|CALL]")
                    print("  nearby <strike> <timestamp> [tolerance_minutes]")
                    print("  stats")
                    print("  quit")
                
                elif cmd == 'test':
                    if len(command) < 3:
                        print("Usage: test <strike> <timestamp> [PUT|CALL]")
                        continue
                    
                    strike = int(command[1])
                    timestamp = command[2]
                    option_type = command[3].upper() if len(command) > 3 else 'PUT'
                    
                    self.test_specific_option(strike, timestamp, option_type)
                
                elif cmd == 'nearby':
                    if len(command) < 3:
                        print("Usage: nearby <strike> <timestamp> [tolerance_minutes]")
                        continue
                    
                    strike = int(command[1])
                    timestamp = command[2]
                    tolerance = int(command[3]) if len(command) > 3 else 5
                    
                    result = self.find_nearby_data(strike, timestamp, tolerance)
                    
                    if "error" in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"\nFound {result['found_records']} records within {tolerance} minutes:")
                        for record in result['data'][:10]:  # Show first 10
                            print(f"  {record['timestamp']} | PUT: {record['put_ltp']} | "
                                  f"CALL: {record['call_ltp']} | Diff: {record['time_diff_minutes']}min")
                
                elif cmd == 'stats':
                    self.print_database_stats()
                
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Option Chain Diagnostic Tool for vAlgo Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific failing case
  python select_query_option_chain.py --strike 24700 --timestamp "2025-06-06 09:45:00" --type PUT
  
  # Find nearby data with 10 minute tolerance
  python select_query_option_chain.py --strike 24700 --timestamp "2025-06-06 09:45:00" --nearby --tolerance 10
  
  # Interactive mode
  python select_query_option_chain.py --interactive
  
  # Database statistics
  python select_query_option_chain.py --stats
        """
    )
    
    parser.add_argument("--strike", type=int, help="Strike price to test")
    parser.add_argument("--timestamp", type=str, help="Timestamp in YYYY-MM-DD HH:MM:SS format")
    parser.add_argument("--type", choices=['PUT', 'CALL'], default='PUT', help="Option type (default: PUT)")
    parser.add_argument("--nearby", action="store_true", help="Find nearby data within tolerance")
    parser.add_argument("--tolerance", type=int, default=5, help="Time tolerance in minutes (default: 5)")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--db-path", type=str, help="Custom database path")
    
    args = parser.parse_args()
    
    # Initialize diagnostic tool
    diagnostic = OptionChainDiagnostic(args.db_path)
    
    try:
        if args.interactive:
            diagnostic.interactive_mode()
        
        elif args.stats:
            diagnostic.print_database_stats()
        
        elif args.strike and args.timestamp:
            if args.nearby:
                result = diagnostic.find_nearby_data(args.strike, args.timestamp, args.tolerance)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"\n✅ Found {result['found_records']} records within {args.tolerance} minutes of {args.timestamp}")
                    print("Timestamp | PUT Premium | CALL Premium | Time Diff")
                    print("-" * 60)
                    for record in result['data']:
                        print(f"{record['timestamp']} | {record['put_ltp']:11.2f} | {record['call_ltp']:12.2f} | {record['time_diff_minutes']:8.1f}min")
            else:
                diagnostic.test_specific_option(args.strike, args.timestamp, args.type)
        
        else:
            parser.print_help()
    
    finally:
        diagnostic.close()


if __name__ == "__main__":
    main()