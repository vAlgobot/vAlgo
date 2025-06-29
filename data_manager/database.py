"""
Database Manager for vAlgo Trading System

This module provides comprehensive database management for OHLCV market data
using DuckDB for high-performance analytics and backtesting operations.

Features:
- OHLCV data storage with proper schema and indexing
- Data loading methods for DataLoad mode
- Data retrieval methods for Backtesting mode
- Data validation and quality assurance
- Performance optimization for large datasets

Author: vAlgo Development Team
Created: June 27, 2025
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
import time

try:
    import duckdb
except ImportError:
    raise ImportError("DuckDB library not installed. Run: pip install duckdb")

from utils.logger import get_logger
from utils.config_loader import ConfigLoader


class DatabaseManager:
    """
    Professional database manager for OHLCV market data storage and retrieval.
    
    Uses DuckDB for high-performance analytical queries suitable for backtesting
    and market data analysis.
    """
    
    def __init__(self, db_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize Database Manager.
        
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
        
        # Database configuration
        self.table_name = "ohlcv_data"
        self.metadata_table = "data_metadata"
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info(f"DatabaseManager initialized with database: {self.db_path}")
    
    def _initialize_database(self) -> None:
        """Initialize database connection and create tables if they don't exist."""
        try:
            self.connection = duckdb.connect(self.db_path)
            
            # Create OHLCV data table
            self._create_ohlcv_table()
            
            # Create metadata table
            self._create_metadata_table()
            
            # Create indexes for better performance
            self._create_indexes()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_ohlcv_table(self) -> None:
        """Create the main OHLCV data table with optimized schema."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            symbol VARCHAR NOT NULL,
            exchange VARCHAR NOT NULL,
            timeframe VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume BIGINT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, exchange, timeframe, timestamp)
        )
        """
        
        self.connection.execute(create_table_sql)
        self.logger.debug("OHLCV table created/verified")
    
    def _create_metadata_table(self) -> None:
        """Create metadata table for tracking data completeness and quality."""
        # Drop existing table if it has old schema
        try:
            self.connection.execute(f"DROP TABLE IF EXISTS {self.metadata_table}")
        except Exception:
            pass  # Table might not exist
        
        create_metadata_sql = f"""
        CREATE TABLE {self.metadata_table} (
            symbol VARCHAR NOT NULL,
            exchange VARCHAR NOT NULL,
            timeframe VARCHAR NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            total_records BIGINT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            missing_days TEXT,
            PRIMARY KEY (symbol, exchange, timeframe)
        )
        """
        
        self.connection.execute(create_metadata_sql)
        self.logger.debug("Metadata table created with new schema")
    
    def _create_indexes(self) -> None:
        """Create database indexes for optimized query performance."""
        try:
            # Index on symbol for fast symbol-based queries
            self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_symbol 
                ON {self.table_name} (symbol)
            """)
            
            # Index on timestamp for time-based queries
            self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON {self.table_name} (timestamp)
            """)
            
            # Composite index for backtesting queries
            self.connection.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
                ON {self.table_name} (symbol, timeframe, timestamp)
            """)
            
            self.logger.debug("Database indexes created/verified")
            
        except Exception as e:
            self.logger.warning(f"Some indexes may already exist: {e}")
    
    def store_ohlcv_data(self, symbol: str, exchange: str, timeframe: str, 
                        data: pd.DataFrame, replace: bool = False) -> bool:
        """
        Store OHLCV data in the database.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Time interval
            data: OHLCV DataFrame with timestamp index
            replace: Whether to replace existing data
            
        Returns:
            bool: True if successful
        """
        try:
            if data.empty:
                self.logger.warning(f"No data to store for {symbol}")
                return False
            
            # Prepare data for insertion
            df_to_store = self._prepare_data_for_storage(symbol, exchange, timeframe, data)
            
            if df_to_store.empty:
                self.logger.warning(f"No valid data after preparation for {symbol}")
                return False
            
            # NUCLEAR CLEANUP: Force delete existing data and any epoch records
            if replace:
                # Delete existing data for this symbol
                self._delete_existing_data(symbol, exchange, timeframe)
                
                # FORCE DELETE any epoch timestamps from entire database (nuclear option)
                epoch_cleanup_sql = f"""
                DELETE FROM {self.table_name} 
                WHERE timestamp <= '1970-01-01 01:00:00' 
                   OR DATE(timestamp) = '1970-01-01'
                """
                self.connection.execute(epoch_cleanup_sql)
                self.logger.info("Force deleted any epoch timestamps from database")
            
            # Insert data
            insert_count = self._insert_ohlcv_data(df_to_store)
            
            # Update metadata
            self._update_metadata(symbol, exchange, timeframe, df_to_store)
            
            self.logger.info(f"Stored {insert_count} records for {symbol} ({exchange}, {timeframe})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store data for {symbol}: {e}")
            return False
    
    def _prepare_data_for_storage(self, symbol: str, exchange: str, timeframe: str, 
                                 data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for database storage with minimal timestamp modification.
        
        PRESERVE RAW API DATA: This method now prioritizes keeping exact timestamps
        from API responses without artificial generation or excessive conversion.
        """
        try:
            df = data.copy()
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Add metadata columns
            df['symbol'] = symbol.upper()
            df['exchange'] = exchange.upper()
            df['timeframe'] = timeframe.lower()
            
            # SIMPLIFIED TIMESTAMP HANDLING - Preserve raw API data
            self.logger.info(f"Raw data received: {len(df)} records")
            
            # Handle timestamp column/index  
            if isinstance(df.index, pd.DatetimeIndex):
                # We have DatetimeIndex - preserve it as timestamp column
                df = df.reset_index()
                # If index doesn't have a name, call it 'timestamp'
                if df.columns[0] == 0 or df.columns[0] == 'index':
                    df.columns = ['timestamp'] + list(df.columns[1:])
                self.logger.info("Preserved DatetimeIndex as timestamp column")
            elif df.index.name == 'timestamp':
                # Index is named timestamp - move to column
                df = df.reset_index()
                self.logger.info("Moved named timestamp index to column")
            elif 'timestamp' not in df.columns:
                # No DatetimeIndex and no timestamp column - use index values
                df['timestamp'] = df.index
                df = df.reset_index(drop=True)
                self.logger.info("Used index as timestamp column")
            
            # Log what we received from API
            if 'timestamp' in df.columns and not df.empty:
                sample = df['timestamp'].head(3).tolist()
                self.logger.info(f"API timestamp sample: {sample}")
                self.logger.info(f"API timestamp dtype: {df['timestamp'].dtype}")
            
            # MINIMAL conversion - only if absolutely necessary
            if 'timestamp' in df.columns:
                # Only convert if not already datetime
                if df['timestamp'].dtype != 'datetime64[ns]' and not isinstance(df['timestamp'].iloc[0] if not df.empty else None, pd.Timestamp):
                    self.logger.info("Converting timestamps to datetime format")
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                else:
                    self.logger.info("Timestamps already in correct format - preserving as-is")
                
                # Only remove truly invalid timestamps (NaT, null) - preserve everything else including holidays
                before_filter = len(df)
                invalid_mask = df['timestamp'].isna()
                df = df[~invalid_mask]
                removed_count = before_filter - len(df)
                
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} records with invalid/null timestamps")
                
                # Log final timestamp range
                if not df.empty:
                    self.logger.info(f"Final timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Basic OHLCV validation (simplified)
            df = self._validate_ohlcv_data(df)
            
            # Select final columns in correct order
            final_columns = [
                'symbol', 'exchange', 'timeframe', 'timestamp',
                'open', 'high', 'low', 'close', 'volume'
            ]
            
            df = df[final_columns]
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            self.logger.info(f"Prepared {len(df)} records for storage (preserving API timestamps)")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return pd.DataFrame()
    
    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        try:
            timeframe = timeframe.lower().strip()
            if timeframe.endswith('m'):
                return int(timeframe[:-1])
            elif timeframe.endswith('h'):
                return int(timeframe[:-1]) * 60
            elif timeframe.endswith('d'):
                return int(timeframe[:-1]) * 60 * 24
            else:
                # Default to 5 minutes for 5m timeframe
                return 5
        except Exception as e:
            self.logger.error(f"Error parsing timeframe {timeframe}: {e}")
            return 5

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified OHLCV validation - minimal processing to preserve API data.
        """
        try:
            original_count = len(df)
            
            if df.empty:
                return df
            
            
            # Basic OHLC relationship validation (preserve price data as much as possible)
            valid_mask = (
                (df['high'] >= df['low']) &
                (df['volume'] >= 0) &  # Allow zero volume (API may provide it)
                (df['open'] > 0) &
                (df['high'] > 0) &
                (df['low'] > 0) &
                (df['close'] > 0)
            )
            
            invalid_ohlc = len(df) - valid_mask.sum()
            df = df[valid_mask]
            
            if invalid_ohlc > 0:
                self.logger.warning(f"Removed {invalid_ohlc} records with invalid OHLC relationships")
            
            # Remove duplicate timestamps (keep first occurrence)
            before_dedup = len(df)
            if 'timestamp' in df.columns:
                df = df.drop_duplicates(subset=['timestamp'], keep='first')
                duplicates_removed = before_dedup - len(df)
                
                if duplicates_removed > 0:
                    self.logger.info(f"Removed {duplicates_removed} duplicate timestamps")
            
            total_removed = original_count - len(df)
            if total_removed > 0:
                self.logger.info(f"Validation summary: {len(df)}/{original_count} records retained ({total_removed} removed)")
            else:
                self.logger.debug(f"All {len(df)} records passed validation")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating OHLCV data: {e}")
            return df
    
    
    def _delete_existing_data(self, symbol: str, exchange: str, timeframe: str) -> None:
        """Delete existing data for symbol/exchange/timeframe combination."""
        delete_sql = f"""
        DELETE FROM {self.table_name} 
        WHERE symbol = ? AND exchange = ? AND timeframe = ?
        """
        
        self.connection.execute(delete_sql, [symbol.upper(), exchange.upper(), timeframe.lower()])
        self.logger.debug(f"Deleted existing data for {symbol} ({exchange}, {timeframe})")
    
    def _insert_ohlcv_data(self, df: pd.DataFrame) -> int:
        """
        Insert OHLCV data into database with minimal filtering.
        
        PRESERVE API DATA: Only remove truly invalid records (null timestamps).
        """
        try:
            if df.empty:
                self.logger.warning("No data to insert")
                return 0
            
            # Log what we're inserting
            self.logger.info(f"Inserting {len(df)} records to database")
            if 'timestamp' in df.columns and not df.empty:
                timestamp_range = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
                self.logger.info(f"Timestamp range: {timestamp_range}")
            
            # Minimal validation - only remove null timestamps
            valid_mask = df['timestamp'].notna()
            df_clean = df[valid_mask].copy()
            
            removed_count = len(df) - len(df_clean)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} records with null timestamps")
            
            if df_clean.empty:
                self.logger.warning("No valid data to insert after removing null timestamps")
                return 0
            
            # Use DuckDB's efficient DataFrame insertion
            self.connection.register('temp_df', df_clean)
            
            insert_sql = f"""
            INSERT INTO {self.table_name} 
            (symbol, exchange, timeframe, timestamp, open, high, low, close, volume)
            SELECT symbol, exchange, timeframe, timestamp, open, high, low, close, volume 
            FROM temp_df
            """
            
            self.connection.execute(insert_sql)
            
            # Clean up temporary registration
            self.connection.unregister('temp_df')
            
            return len(df_clean)
            
        except Exception as e:
            self.logger.error(f"Error inserting data: {e}")
            raise
    
    def _update_metadata(self, symbol: str, exchange: str, timeframe: str, df: pd.DataFrame) -> None:
        """Update metadata table with data information."""
        try:
            start_date = df['timestamp'].min().date()
            end_date = df['timestamp'].max().date()
            total_records = len(df)
            
            # Check for missing days (simplified)
            expected_days = (end_date - start_date).days + 1
            actual_days = df['timestamp'].dt.date.nunique()
            missing_days_count = expected_days - actual_days
            
            missing_days = f"{missing_days_count} potential missing days" if missing_days_count > 0 else "None"
            
            # Upsert metadata
            upsert_sql = f"""
            INSERT OR REPLACE INTO {self.metadata_table} 
            (symbol, exchange, timeframe, start_date, end_date, total_records, 
             missing_days, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            self.connection.execute(upsert_sql, [
                symbol.upper(), exchange.upper(), timeframe.lower(),
                start_date, end_date, total_records, missing_days
            ])
            
            self.logger.debug(f"Updated metadata for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error updating metadata: {e}")
    
    def load_data_from_market_data_manager(self, market_data_dict: Dict[str, pd.DataFrame],
                                         instruments_config: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Load data from MarketDataManager results into database.
        
        Args:
            market_data_dict: Dictionary of symbol -> DataFrame from MarketDataManager
            instruments_config: List of instrument configurations
            
        Returns:
            Dict mapping symbols to success status
        """
        results = {}
        
        self.logger.info(f"Loading data for {len(market_data_dict)} instruments into database")
        
        # Create mapping of symbols to their config
        symbol_config_map = {
            config['symbol'].upper(): config 
            for config in instruments_config
        }
        
        for symbol, df in market_data_dict.items():
            try:
                # Get instrument configuration
                config = symbol_config_map.get(symbol.upper())
                if not config:
                    self.logger.warning(f"No configuration found for {symbol}")
                    results[symbol] = False
                    continue
                
                exchange = config.get('exchange', 'NSE')
                timeframe = config.get('timeframe', '5m')
                
                # Store in database
                success = self.store_ohlcv_data(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data=df,
                    replace=True  # Replace existing data during DataLoad
                )
                
                results[symbol] = success
                
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")
                results[symbol] = False
        
        successful_loads = sum(results.values())
        self.logger.info(f"Successfully loaded {successful_loads}/{len(results)} instruments")
        
        return results
    
    def get_ohlcv_data(self, symbol: str, exchange: str, timeframe: str,
                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve OHLCV data for backtesting.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Time interval
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        try:
            # Build query
            where_conditions = [
                "symbol = ?",
                "exchange = ?", 
                "timeframe = ?"
            ]
            
            params = [symbol.upper(), exchange.upper(), timeframe.lower()]
            
            if start_date:
                where_conditions.append("timestamp >= ?")
                params.append(start_date)
            
            if end_date:
                where_conditions.append("timestamp <= ?")
                params.append(end_date)
            
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM {self.table_name}
            WHERE {' AND '.join(where_conditions)}
            ORDER BY timestamp ASC
            """
            
            # Execute query
            df = self.connection.execute(query, params).df()
            
            if df.empty:
                self.logger.warning(f"No data found for {symbol} ({exchange}, {timeframe})")
                return pd.DataFrame()
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            self.logger.info(f"Retrieved {len(df)} records for {symbol} ({exchange}, {timeframe})")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_instruments_data(self, instruments: List[Dict[str, str]],
                                    start_date: Optional[str] = None, 
                                    end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Retrieve data for multiple instruments (for backtesting).
        
        Args:
            instruments: List of dicts with symbol, exchange, timeframe
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            Dict mapping symbols to DataFrames
        """
        results = {}
        
        self.logger.info(f"Retrieving data for {len(instruments)} instruments")
        
        for instrument in instruments:
            symbol = instrument['symbol']
            exchange = instrument['exchange']
            timeframe = instrument['timeframe']
            
            df = self.get_ohlcv_data(symbol, exchange, timeframe, start_date, end_date)
            
            if not df.empty:
                results[symbol] = df
        
        self.logger.info(f"Successfully retrieved data for {len(results)} instruments")
        return results
    
    def get_backtest_data_from_config(self, config_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for all active instruments using config dates (for backtesting).
        
        Args:
            config_path: Path to Excel configuration file
            
        Returns:
            Dict mapping symbols to DataFrames
        """
        try:
            config_path = config_path or self.config_path
            config_loader = ConfigLoader(config_path)
            
            if not config_loader.load_config():
                self.logger.error("Failed to load configuration")
                return {}
            
            # Get date range from config
            init_config = config_loader.get_initialize_config()
            start_date = init_config.get('start date')
            end_date = init_config.get('end date')
            
            # Get active instruments
            instruments = config_loader.get_active_instruments()
            
            self.logger.info(f"Loading backtest data from {start_date} to {end_date}")
            
            return self.get_multiple_instruments_data(instruments, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Error getting backtest data from config: {e}")
            return {}
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of available data in the database."""
        try:
            query = f"""
            SELECT 
                symbol,
                exchange,
                timeframe,
                COUNT(*) as total_records,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM {self.table_name}
            GROUP BY symbol, exchange, timeframe
            ORDER BY symbol, timeframe
            """
            
            df = self.connection.execute(query).df()
            
            # Format dates
            if not df.empty:
                df['start_date'] = pd.to_datetime(df['start_date']).dt.date
                df['end_date'] = pd.to_datetime(df['end_date']).dt.date
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return pd.DataFrame()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Total records
            total_records = self.connection.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()[0]
            
            # Unique symbols
            unique_symbols = self.connection.execute(
                f"SELECT COUNT(DISTINCT symbol) FROM {self.table_name}"
            ).fetchone()[0]
            
            # Date range
            date_range = self.connection.execute(f"""
                SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
                FROM {self.table_name}
            """).fetchone()
            
            # Database file size
            db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            
            stats = {
                'total_records': total_records,
                'unique_symbols': unique_symbols,
                'start_date': date_range[0] if date_range[0] else None,
                'end_date': date_range[1] if date_range[1] else None,
                'database_size_mb': round(db_size_mb, 2),
                'database_path': self.db_path
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_invalid_timestamps(self) -> int:
        """
        Remove epoch and other obviously invalid timestamps from database.
        
        Returns:
            int: Number of records deleted
        """
        try:
            # Query to identify invalid timestamps before deletion
            count_query = f"""
            SELECT COUNT(*) FROM {self.table_name}
            WHERE timestamp <= '1970-01-01 01:00:00'
               OR timestamp IS NULL
               OR timestamp > '2040-01-01'
            """
            
            invalid_count = self.connection.execute(count_query).fetchone()[0]
            
            if invalid_count > 0:
                # Delete invalid timestamps
                cleanup_query = f"""
                DELETE FROM {self.table_name}
                WHERE timestamp <= '1970-01-01 01:00:00'
                   OR timestamp IS NULL
                   OR timestamp > '2040-01-01'
                """
                
                self.connection.execute(cleanup_query)
                
                # Also clean up metadata for invalid timestamp ranges
                metadata_cleanup = f"""
                DELETE FROM {self.metadata_table}
                WHERE start_date <= '1970-01-01'
                   OR end_date <= '1970-01-01'
                   OR start_date > '2040-01-01'
                """
                
                self.connection.execute(metadata_cleanup)
                
                self.logger.info(f"Cleaned up {invalid_count} records with invalid timestamps (epoch, null, far future)")
            else:
                self.logger.info("No invalid timestamps found - database is clean")
            
            return invalid_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up invalid timestamps: {e}")
            return 0
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """
        Clean up old data beyond specified days.
        
        Args:
            days_to_keep: Number of days to retain
            
        Returns:
            int: Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            delete_query = f"""
            DELETE FROM {self.table_name}
            WHERE timestamp < ?
            """
            
            # Get count before deletion
            count_before = self.connection.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE timestamp < ?",
                [cutoff_date]
            ).fetchone()[0]
            
            # Execute deletion
            self.connection.execute(delete_query, [cutoff_date])
            
            self.logger.info(f"Cleaned up {count_before} old records (older than {days_to_keep} days)")
            return count_before
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def export_data(self, symbol: str, exchange: str, timeframe: str, 
                   output_path: str, format: str = 'csv',
                   start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """
        Export data for a specific instrument.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Time interval
            output_path: Output file path
            format: Export format ('csv' or 'parquet')
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            df = self.get_ohlcv_data(symbol, exchange, timeframe, start_date, end_date)
            
            if df.empty:
                self.logger.warning(f"No data to export for {symbol}")
                return False
            
            if format.lower() == 'csv':
                df.to_csv(output_path)
            elif format.lower() == 'parquet':
                df.to_parquet(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(df)} records to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
    
    def close(self) -> None:
        """Close database connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __str__(self) -> str:
        """String representation."""
        return f"DatabaseManager(db_path={self.db_path})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        stats = self.get_database_stats()
        return (f"DatabaseManager(db_path={self.db_path}, "
                f"records={stats.get('total_records', 0)}, "
                f"symbols={stats.get('unique_symbols', 0)})")