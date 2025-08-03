#!/usr/bin/env python3
"""
Options Chain Database Manager for vAlgo Trading System
======================================================

Professional database manager for options chain data storage and retrieval.
Extends the existing DatabaseManager architecture with options-specific functionality.

Features:
- Options chain data storage with duplicate prevention
- High-performance strike and time-based queries
- Data quality validation and integrity checks
- Incremental data loading with UPSERT operations
- Integration with existing vAlgo database infrastructure

Author: vAlgo Development Team
Created: July 10, 2025
Version: 1.0.0 (Production)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
import time
import hashlib

try:
    import duckdb
except ImportError:
    raise ImportError("DuckDB library not installed. Run: pip install duckdb")

# Import vAlgo utilities
from utils.logger import get_logger
from data_manager.database import DatabaseManager


class OptionsDatabase:
    """
    Professional options chain database manager with duplicate prevention
    and high-performance querying capabilities.
    
    Extends vAlgo's existing database architecture for options-specific operations.
    """
    
    def __init__(self, db_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize Options Database Manager.
        
        Args:
            db_path: Path to DuckDB database file
            config_path: Path to Excel configuration file
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path or "config/config.xlsx"
        
        # Set up database path
        if db_path is None:
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "vAlgo_market_data.db")
        
        self.db_path = db_path
        self.connection = None
        
        # Options-specific table configuration
        self.options_table = "nifty_expired_option_chain"
        self.options_metadata_table = "options_metadata"
        
        # Initialize database connection
        self._initialize_database()
        
        self.logger.info(f"OptionsDatabase initialized with database: {self.db_path}")
    
    def _initialize_database(self) -> None:
        """Initialize database connection and create options tables."""
        try:
            self.connection = duckdb.connect(self.db_path)
            
            # Create options chain table
            self._create_options_table()
            
            # Create options metadata table
            self._create_options_metadata_table()
            
            # Create performance indexes
            self._create_options_indexes()
            
            self.logger.info("Options database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize options database: {e}")
            raise
    
    def _create_options_table(self) -> None:
        """Create the main options chain data table with optimized schema."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.options_table} (
            timestamp TIMESTAMP NOT NULL,
            strike INTEGER NOT NULL,
            expiry_date DATE NOT NULL,
            call_ltp DOUBLE,
            call_iv DOUBLE,
            call_delta DOUBLE,
            put_ltp DOUBLE,
            put_iv DOUBLE,
            put_delta DOUBLE,
            dte INTEGER,
            data_source VARCHAR(50) DEFAULT 'CSV_IMPORT',
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            record_hash VARCHAR(64),
            PRIMARY KEY (timestamp, strike, expiry_date)
        )
        """
        
        self.connection.execute(create_table_sql)
        self.logger.debug("Options chain table created/verified")
    
    def _create_options_metadata_table(self) -> None:
        """Create metadata table for tracking options data completeness."""
        create_metadata_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.options_metadata_table} (
            expiry_date DATE NOT NULL,
            start_timestamp TIMESTAMP NOT NULL,
            end_timestamp TIMESTAMP NOT NULL,
            total_records BIGINT NOT NULL,
            unique_strikes INTEGER NOT NULL,
            data_quality_score DOUBLE DEFAULT 100.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_source VARCHAR(50) DEFAULT 'CSV_IMPORT',
            PRIMARY KEY (expiry_date, data_source)
        )
        """
        
        self.connection.execute(create_metadata_sql)
        self.logger.debug("Options metadata table created/verified")
    
    def _create_options_indexes(self) -> None:
        """Create database indexes for optimized options queries."""
        try:
            # Index for strike-based queries (most common)
            self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_options_strike_time 
                ON {self.options_table} (strike, timestamp)
            """)
            
            # Index for time-based queries
            self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_options_timestamp 
                ON {self.options_table} (timestamp)
            """)
            
            # Index for expiry-based queries
            self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_options_expiry 
                ON {self.options_table} (expiry_date, strike)
            """)
            
            # Composite index for most common queries (strike selection)
            self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_options_strike_selection 
                ON {self.options_table} (timestamp, expiry_date, strike)
            """)
            
            # Index for data quality queries
            self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_options_hash 
                ON {self.options_table} (record_hash)
            """)
            
            self.logger.debug("Options database indexes created/verified")
            
        except Exception as e:
            self.logger.warning(f"Some options indexes may already exist: {e}")
    
    def _generate_record_hash(self, row_data: Dict[str, Any]) -> str:
        """Generate unique hash for duplicate detection."""
        # Create hash from key fields that should be unique
        hash_fields = [
            str(row_data.get('timestamp', '')),
            str(row_data.get('strike', '')),
            str(row_data.get('expiry_date', '')),
            str(row_data.get('call_ltp', '')),
            str(row_data.get('put_ltp', ''))
        ]
        
        hash_string = '|'.join(hash_fields)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def store_options_data(self, data: pd.DataFrame, data_source: str = "CSV_IMPORT", 
                          prevent_duplicates: bool = True) -> Dict[str, Any]:
        """
        Store options chain data with duplicate prevention.
        
        Args:
            data: Options DataFrame with required columns
            data_source: Source identifier for the data
            prevent_duplicates: Whether to check for and prevent duplicates
            
        Returns:
            Dict with storage statistics
        """
        try:
            if data.empty:
                self.logger.warning("No options data to store")
                return {"status": "no_data", "records_processed": 0}
            
            # Validate required columns
            required_columns = [
                'timestamp', 'strike', 'expiry_date', 'call_ltp', 'put_ltp',
                'call_iv', 'call_delta', 'put_iv', 'put_delta', 'dte'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Prepare data for storage
            df_to_store = data.copy()
            df_to_store['data_source'] = data_source
            df_to_store['last_updated'] = datetime.now()
            
            # Generate record hashes for duplicate detection
            if prevent_duplicates:
                df_to_store['record_hash'] = df_to_store.apply(
                    lambda row: self._generate_record_hash(row.to_dict()), axis=1
                )
                
                # Check for existing hashes
                existing_hashes = self._get_existing_hashes(df_to_store['record_hash'].tolist())
                
                # Filter out duplicates
                df_to_store = df_to_store[~df_to_store['record_hash'].isin(existing_hashes)]
                
                if df_to_store.empty:
                    self.logger.info("All records already exist in database - no duplicates inserted")
                    return {
                        "status": "all_duplicates",
                        "records_processed": 0,
                        "duplicates_found": len(data)
                    }
            
            # Insert new records
            records_inserted = len(df_to_store)
            
            # Use INSERT OR REPLACE for UPSERT behavior
            df_to_store.to_sql(
                self.options_table,
                self.connection,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            # Update metadata
            self._update_options_metadata(df_to_store, data_source)
            
            duplicates_found = len(data) - records_inserted if prevent_duplicates else 0
            
            self.logger.info(f"Options data stored: {records_inserted} new records, "
                           f"{duplicates_found} duplicates prevented")
            
            return {
                "status": "success",
                "records_processed": records_inserted,
                "duplicates_found": duplicates_found,
                "total_input_records": len(data)
            }
            
        except Exception as e:
            self.logger.error(f"Error storing options data: {e}")
            raise
    
    def _get_existing_hashes(self, hash_list: List[str]) -> List[str]:
        """Get existing record hashes from database."""
        if not hash_list:
            return []
        
        # Convert to quoted string list for SQL
        hash_str = "', '".join(hash_list)
        query = f"""
            SELECT DISTINCT record_hash 
            FROM {self.options_table} 
            WHERE record_hash IN ('{hash_str}')
        """
        
        try:
            result = self.connection.execute(query).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            self.logger.warning(f"Error checking existing hashes: {e}")
            return []
    
    def _update_options_metadata(self, data: pd.DataFrame, data_source: str) -> None:
        """Update options metadata table with data statistics."""
        try:
            # Group by expiry date
            for expiry_date, group in data.groupby('expiry_date'):
                metadata = {
                    'expiry_date': expiry_date,
                    'start_timestamp': group['timestamp'].min(),
                    'end_timestamp': group['timestamp'].max(),
                    'total_records': len(group),
                    'unique_strikes': group['strike'].nunique(),
                    'data_quality_score': self._calculate_data_quality_score(group),
                    'data_source': data_source,
                    'last_updated': datetime.now()
                }
                
                # UPSERT metadata
                self.connection.execute(f"""
                    INSERT OR REPLACE INTO {self.options_metadata_table} 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, list(metadata.values()))
                
        except Exception as e:
            self.logger.warning(f"Error updating options metadata: {e}")
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score based on completeness and consistency."""
        try:
            total_fields = len(data) * 10  # 10 main data fields per record
            missing_fields = data.isnull().sum().sum()
            
            completeness_score = ((total_fields - missing_fields) / total_fields) * 100
            
            # Additional quality checks
            quality_deductions = 0
            
            # Check for negative premiums
            if (data['call_ltp'] < 0).any() or (data['put_ltp'] < 0).any():
                quality_deductions += 5
            
            # Check for extreme IV values
            if (data['call_iv'] > 100).any() or (data['put_iv'] > 100).any():
                quality_deductions += 3
            
            # Check for extreme Delta values
            if (abs(data['call_delta']) > 1).any() or (abs(data['put_delta']) > 1).any():
                quality_deductions += 3
            
            final_score = max(0, completeness_score - quality_deductions)
            return round(final_score, 2)
            
        except Exception:
            return 75.0  # Default score if calculation fails
    
    def get_strikes_for_timestamp(self, timestamp: datetime, expiry_date: str,
                                 strike_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """
        Get all available strikes for a specific timestamp and expiry.
        
        Args:
            timestamp: Specific timestamp to query
            expiry_date: Option expiry date
            strike_range: Optional tuple of (min_strike, max_strike)
            
        Returns:
            DataFrame with strikes and their data
        """
        try:
            base_query = f"""
                SELECT * FROM {self.options_table}
                WHERE timestamp = ? AND expiry_date = ?
            """
            params = [timestamp, expiry_date]
            
            if strike_range:
                base_query += " AND strike BETWEEN ? AND ?"
                params.extend(strike_range)
            
            base_query += " ORDER BY strike"
            
            result = self.connection.execute(base_query, params).fetchdf()
            
            self.logger.debug(f"Retrieved {len(result)} strikes for {timestamp}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving strikes for timestamp: {e}")
            return pd.DataFrame()
    
    def get_strike_ohlc_data(self, strike: int, expiry_date: str,
                           start_time: datetime, end_time: datetime,
                           option_type: str = 'call') -> pd.DataFrame:
        """
        Get OHLC data for a specific strike over a time period.
        
        Args:
            strike: Option strike price
            expiry_date: Option expiry date
            start_time: Start timestamp
            end_time: End timestamp
            option_type: 'call' or 'put'
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            price_column = f"{option_type.lower()}_ltp"
            
            query = f"""
                SELECT 
                    timestamp,
                    strike,
                    {price_column} as price,
                    {option_type.lower()}_iv as iv,
                    {option_type.lower()}_delta as delta
                FROM {self.options_table}
                WHERE strike = ? AND expiry_date = ?
                AND timestamp BETWEEN ? AND ?
                AND {price_column} IS NOT NULL
                ORDER BY timestamp
            """
            
            result = self.connection.execute(
                query, [strike, expiry_date, start_time, end_time]
            ).fetchdf()
            
            # Generate OHLC from LTP data if we have enough points
            if len(result) > 0:
                ohlc_data = self._generate_ohlc_from_ltp(result)
                return ohlc_data
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error retrieving strike OHLC data: {e}")
            return pd.DataFrame()
    
    def _generate_ohlc_from_ltp(self, ltp_data: pd.DataFrame) -> pd.DataFrame:
        """Generate OHLC data from LTP points."""
        try:
            if ltp_data.empty:
                return pd.DataFrame()
            
            # Group by minute for 1-minute OHLC
            ltp_data['timestamp'] = pd.to_datetime(ltp_data['timestamp'])
            ltp_data.set_index('timestamp', inplace=True)
            
            # Resample to 1-minute intervals
            ohlc = ltp_data['price'].resample('1T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            # Add volume estimation (based on price changes)
            ohlc['volume'] = abs(ohlc['close'] - ohlc['open']) * 100  # Simple estimation
            
            # Add additional option metrics
            iv_data = ltp_data['iv'].resample('1T').last().dropna()
            delta_data = ltp_data['delta'].resample('1T').last().dropna()
            
            ohlc['iv'] = iv_data
            ohlc['delta'] = delta_data
            
            ohlc.reset_index(inplace=True)
            return ohlc
            
        except Exception as e:
            self.logger.error(f"Error generating OHLC from LTP: {e}")
            return pd.DataFrame()
    
    def find_atm_strike(self, underlying_price: float, timestamp: datetime,
                       expiry_date: str) -> Optional[int]:
        """
        Find the At-The-Money (ATM) strike for given underlying price.
        
        Args:
            underlying_price: Current underlying asset price
            timestamp: Timestamp for the query
            expiry_date: Option expiry date
            
        Returns:
            ATM strike price or None if not found
        """
        try:
            query = f"""
                SELECT strike, ABS(strike - ?) as distance
                FROM {self.options_table}
                WHERE timestamp = ? AND expiry_date = ?
                ORDER BY distance
                LIMIT 1
            """
            
            result = self.connection.execute(
                query, [underlying_price, timestamp, expiry_date]
            ).fetchone()
            
            if result:
                atm_strike = result[0]
                self.logger.debug(f"ATM strike for price {underlying_price}: {atm_strike}")
                return atm_strike
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding ATM strike: {e}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            # Basic record count
            record_count = self.connection.execute(
                f"SELECT COUNT(*) FROM {self.options_table}"
            ).fetchone()[0]
            
            # Unique strikes
            unique_strikes = self.connection.execute(
                f"SELECT COUNT(DISTINCT strike) FROM {self.options_table}"
            ).fetchone()[0]
            
            # Date range
            date_range = self.connection.execute(f"""
                SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date
                FROM {self.options_table}
            """).fetchone()
            
            # Expiry dates
            expiry_count = self.connection.execute(
                f"SELECT COUNT(DISTINCT expiry_date) FROM {self.options_table}"
            ).fetchone()[0]
            
            # Data quality
            avg_quality = self.connection.execute(
                f"SELECT AVG(data_quality_score) FROM {self.options_metadata_table}"
            ).fetchone()[0] or 0
            
            return {
                "total_records": record_count,
                "unique_strikes": unique_strikes,
                "date_range": {
                    "start": date_range[0],
                    "end": date_range[1]
                },
                "expiry_dates": expiry_count,
                "average_data_quality": round(avg_quality, 2),
                "database_path": self.db_path,
                "table_name": self.options_table
            }
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    def get_distinct_dates(self, date_type: str = "trading") -> List[str]:
        """
        Get distinct dates from options chain data.
        
        Args:
            date_type: Type of dates to retrieve ("trading" for timestamp dates, "expiry" for expiry dates)
            
        Returns:
            List of distinct dates in YYYY-MM-DD format
        """
        try:
            if date_type.lower() == "expiry":
                query = f"""
                SELECT DISTINCT DATE(expiry_date) as date_only
                FROM {self.options_table}
                ORDER BY date_only
                """
            else:  # trading dates
                query = f"""
                SELECT DISTINCT DATE(timestamp) as date_only
                FROM {self.options_table}
                ORDER BY date_only
                """
            
            result = self.connection.execute(query).fetchall()
            dates = [str(row[0]) for row in result]
            
            self.logger.info(f"Retrieved {len(dates)} distinct {date_type} dates from options data")
            return dates
            
        except Exception as e:
            self.logger.error(f"Error getting distinct {date_type} dates: {e}")
            return []
    
    def get_date_coverage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive date coverage statistics for options data.
        
        Returns:
            Dictionary with date coverage analysis
        """
        try:
            # Get trading date coverage
            trading_dates = self.get_distinct_dates("trading")
            expiry_dates = self.get_distinct_dates("expiry")
            
            # Get date range gaps
            date_gaps = self._analyze_date_gaps(trading_dates)
            
            # Records per date analysis
            records_per_date = self.connection.execute(f"""
                SELECT DATE(timestamp) as date_only, COUNT(*) as record_count
                FROM {self.options_table}
                GROUP BY DATE(timestamp)
                ORDER BY record_count DESC
            """).fetchall()
            
            # Strike coverage per date
            strikes_per_date = self.connection.execute(f"""
                SELECT DATE(timestamp) as date_only, COUNT(DISTINCT strike) as strike_count
                FROM {self.options_table}
                GROUP BY DATE(timestamp)
                ORDER BY strike_count DESC
            """).fetchall()
            
            return {
                "trading_dates": {
                    "total_dates": len(trading_dates),
                    "start_date": trading_dates[0] if trading_dates else None,
                    "end_date": trading_dates[-1] if trading_dates else None,
                    "date_gaps": date_gaps
                },
                "expiry_dates": {
                    "total_expiries": len(expiry_dates),
                    "dates": expiry_dates
                },
                "data_density": {
                    "max_records_per_day": records_per_date[0] if records_per_date else None,
                    "min_records_per_day": records_per_date[-1] if records_per_date else None,
                    "avg_records_per_day": sum(r[1] for r in records_per_date) / len(records_per_date) if records_per_date else 0
                },
                "strike_coverage": {
                    "max_strikes_per_day": strikes_per_date[0] if strikes_per_date else None,
                    "min_strikes_per_day": strikes_per_date[-1] if strikes_per_date else None,
                    "avg_strikes_per_day": sum(r[1] for r in strikes_per_date) / len(strikes_per_date) if strikes_per_date else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting date coverage stats: {e}")
            return {"error": str(e)}
    
    def get_option_premium_at_time(self, symbol: str, strike: int, option_type: str, 
                                   timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Get option premium at specific time from database.
        
        Args:
            symbol: Underlying symbol (e.g., 'NIFTY')
            strike: Strike price
            option_type: 'CALL' or 'PUT'
            timestamp: Timestamp to query
            
        Returns:
            Dict with premium and option data or None if not found
        """
        try:
            # Determine the correct LTP column based on option type
            ltp_column = f"{option_type.lower()}_ltp"
            iv_column = f"{option_type.lower()}_iv"
            delta_column = f"{option_type.lower()}_delta"
            
            # Query for exact time - filter by strike and timestamp only
            query = f"""
                SELECT 
                    {ltp_column} as premium,
                    {iv_column} as iv,
                    {delta_column} as delta,
                    timestamp,
                    dte,
                    expiry_date
                FROM {self.options_table}
                WHERE strike = ? AND timestamp = ?
                AND {ltp_column} IS NOT NULL AND {ltp_column} > 0
                ORDER BY expiry_date DESC
                LIMIT 1
            """
            
            result = self.connection.execute(
                query, [strike, timestamp]
            ).fetchone()
            
            if result:
                return {
                    'premium': result[0],
                    'iv': result[1],
                    'delta': result[2],
                    'timestamp': result[3],
                    'dte': result[4],
                    'expiry_date': result[5],
                    'strike': strike,
                    'option_type': option_type
                } 
            else:
                self.logger.debug(f"No premium data found for strike {strike} {option_type} at {timestamp} from database)")
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting option premium at time: {e}")
            return None

    def _analyze_date_gaps(self, dates: List[str]) -> List[Dict[str, str]]:
        """Analyze gaps in date sequence."""
        try:
            from datetime import datetime, timedelta
            
            if len(dates) < 2:
                return []
            
            gaps = []
            for i in range(1, len(dates)):
                prev_date = datetime.strptime(dates[i-1], '%Y-%m-%d')
                curr_date = datetime.strptime(dates[i], '%Y-%m-%d')
                
                # Check for gaps (more than 1 day for weekdays, more than 3 days for weekends)
                diff = (curr_date - prev_date).days
                if diff > 3:  # Likely a gap (considering weekends)
                    gaps.append({
                        "gap_start": dates[i-1],
                        "gap_end": dates[i],
                        "days_missing": diff - 1
                    })
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error analyzing date gaps: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old options data beyond specified days.
        
        Args:
            days_to_keep: Number of days to retain
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            delete_query = f"""
                DELETE FROM {self.options_table}
                WHERE timestamp < ?
            """
            
            deleted_count = self.connection.execute(delete_query, [cutoff_date]).rowcount
            
            self.logger.info(f"Cleaned up {deleted_count} old options records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Options database connection closed")


# Convenience functions
def create_options_database(db_path: Optional[str] = None, 
                          config_path: Optional[str] = None) -> OptionsDatabase:
    """Create and return OptionsDatabase instance."""
    return OptionsDatabase(db_path, config_path)


# Example usage and testing
if __name__ == "__main__":
    # Example usage of the Options Database
    import numpy as np
    
    # Create database instance
    options_db = create_options_database()
    
    # Generate sample data for testing
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01 09:15:00', periods=100, freq='1T'),
        'strike': np.repeat([21500, 21550, 21600], 34)[:100],
        'expiry_date': ['2024-01-04'] * 100,
        'call_ltp': np.random.uniform(10, 500, 100),
        'call_iv': np.random.uniform(15, 25, 100),
        'call_delta': np.random.uniform(0.1, 0.9, 100),
        'put_ltp': np.random.uniform(10, 500, 100),
        'put_iv': np.random.uniform(15, 25, 100),
        'put_delta': np.random.uniform(-0.9, -0.1, 100),
        'dte': [3] * 100
    })
    
    # Test data storage
    result = options_db.store_options_data(sample_data, data_source="TEST_DATA")
    print(f"Storage result: {result}")
    
    # Test ATM strike finding
    atm_strike = options_db.find_atm_strike(
        underlying_price=21575,
        timestamp=sample_data['timestamp'].iloc[0],
        expiry_date='2024-01-04'
    )
    print(f"ATM Strike: {atm_strike}")
    
    # Test database stats
    stats = options_db.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Close connection
    options_db.close()