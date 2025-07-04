#!/usr/bin/env python3
"""
vAlgo Data Loading Script
========================

This script loads historical data for all configured instruments using the DataLoad mode.
It reads configuration from Excel, fetches data via OpenAlgo, and stores in DuckDB database.

Usage:
    python main_data_load.py

Requirements:
    - Excel config with Mode="dataload"
    - .env file with OPENALGO_API_KEY
    - OpenAlgo server running with broker connected
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any

# Add vAlgo to path
sys.path.append(str(Path(__file__).parent))

# Import vAlgo modules
from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.env_loader import get_all_config
from utils.constants import TradingMode
from data_manager.openalgo import OpenAlgo
from data_manager.market_data import MarketDataManager
from data_manager.database import DatabaseManager


class DataLoadManager:
    """
    Manages the complete data loading process for vAlgo system.
    
    Orchestrates fetching historical data for all configured instruments
    and storing them in the database for backtesting and analysis.
    """
    
    def __init__(self, config_path: str = "config/config.xlsx"):
        """Initialize DataLoadManager with configuration."""
        self.config_path = config_path
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.config_loader = None
        self.openalgo_client = None
        self.market_data_manager = None
        self.database_manager = None
        
        # Statistics tracking
        self.stats = {
            'total_instruments': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None,
            'instruments_status': {}
        }
    
    def run_data_load(self) -> bool:
        """
        Execute the complete data loading process.
        
        Returns:
            bool: True if data load completed successfully
        """
        self.stats['start_time'] = time.time()
        
        try:
            self.logger.info("Starting vAlgo Data Loading Process")
            print("🚀 vAlgo Data Loading Process")
            print("=" * 60)
            
            # Step 1: Initialize configuration
            if not self._initialize_configuration():
                return False
            
            # Step 2: Initialize OpenAlgo connection
            if not self._initialize_openalgo():
                return False
            
            # Step 3: Initialize market data manager
            if not self._initialize_market_data_manager():
                return False
            
            # Step 4: Initialize database
            if not self._initialize_database():
                return False
            
            # Step 5: Prepare for incremental data loading
            print("\n=== PREPARING FOR INCREMENTAL DATA LOADING ===")
            print("📊 Checking existing data coverage...")
            existing_data_summary = self._get_existing_data_summary()
            if existing_data_summary:
                print(f"✅ Found existing data: {existing_data_summary}")
            else:
                print("✅ No existing data found - fresh data loading")
            
            # Step 6: Load data for all instruments
            if not self._load_all_instruments():
                return False
            
            # Step 7: Generate summary report
            self._generate_summary_report()
            
            self.logger.info("Data loading process completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data loading process failed: {e}")
            print(f"❌ Data loading failed: {e}")
            return False
        
        finally:
            self.stats['end_time'] = time.time()
            self._cleanup()
    
    def _initialize_configuration(self) -> bool:
        """Initialize and validate configuration."""
        try:
            print("\n=== INITIALIZING CONFIGURATION ===")
            
            # Check if config file exists
            if not Path(self.config_path).exists():
                print(f"❌ Configuration file not found: {self.config_path}")
                return False
            
            # Load configuration
            self.config_loader = ConfigLoader(self.config_path)
            
            # Explicitly load the config data
            if not self.config_loader.load_config():
                print("❌ Failed to load configuration from Excel file")
                return False
            
            # Validate trading mode
            initialize_config = self.config_loader.get_initialize_config()
            mode = initialize_config.get('mode', '').lower()
            
            if mode != TradingMode.DATALOAD.value:
                print(f"❌ Invalid mode: {mode}. Expected: {TradingMode.DATALOAD.value}")
                print("Please set Mode='dataload' in Initialize sheet")
                return False
            
            # Debug: Show available fields
            print(f"   Available fields: {list(initialize_config.keys())}")
            
            # Get date range - try multiple possible field names
            self.start_date = (
                initialize_config.get('start_date') or 
                initialize_config.get('Start_Date') or 
                initialize_config.get('StartDate') or
                initialize_config.get('From_Date') or
                initialize_config.get('start')
            )
            
            self.end_date = (
                initialize_config.get('end_date') or 
                initialize_config.get('End_Date') or 
                initialize_config.get('EndDate') or
                initialize_config.get('To_Date') or
                initialize_config.get('end')
            )
            
            # Check for data export flag
            self.data_export_enabled = self._check_data_export_flag(initialize_config)
            
            if not self.start_date or not self.end_date:
                print("❌ Start date and end date must be configured in Initialize sheet")
                print(f"   Looking for fields: start_date, Start_Date, StartDate, From_Date, start")
                print(f"   Looking for fields: end_date, End_Date, EndDate, To_Date, end")
                print(f"   Found start_date: {self.start_date}")
                print(f"   Found end_date: {self.end_date}")
                return False
            
            print(f"✅ Configuration loaded: {self.config_path}")
            print(f"   Mode: {mode}")
            print(f"   Date range: {self.start_date} to {self.end_date}")
            print(f"   Data export: {'enabled' if self.data_export_enabled else 'disabled'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration: {e}")
            print(f"❌ Configuration error: {e}")
            return False
    
    def _check_data_export_flag(self, initialize_config: Dict[str, Any]) -> bool:
        """Check if data export is enabled in configuration."""
        try:
            export_flag = (
                initialize_config.get('data_export') or 
                initialize_config.get('Data_Export') or 
                initialize_config.get('DataExport') or
                initialize_config.get('export_data') or
                initialize_config.get('Export_Data')
            )
            
            if export_flag is None:
                return False
            
            # Convert to boolean
            if isinstance(export_flag, str):
                return export_flag.lower() in ['true', 'yes', '1', 'on', 'enabled']
            elif isinstance(export_flag, (int, float)):
                return bool(export_flag)
            else:
                return bool(export_flag)
                
        except Exception as e:
            self.logger.warning(f"Error checking data export flag: {e}")
            return False
    
    def _initialize_openalgo(self) -> bool:
        """Initialize OpenAlgo connection."""
        try:
            print("\n=== INITIALIZING OPENALGO CONNECTION ===")
            
            # Load API key from .env
            env_config = get_all_config()
            api_key = env_config.get('openalgo_api_key') or env_config.get('OPENALGO_API_KEY')
            
            if not api_key:
                print("❌ OPENALGO_API_KEY not found in .env file")
                return False
            
            print(f"✅ API key loaded: {api_key[:20]}...")
            
            # Initialize OpenAlgo client
            self.openalgo_client = OpenAlgo(
                config_path=self.config_path,
                api_key=api_key
            )
            
            # Test connection
            if not self.openalgo_client.connect():
                print("❌ Failed to connect to OpenAlgo")
                return False
            
            print("✅ OpenAlgo connection successful")
            
            # Test basic API call
            funds_response = self.openalgo_client.get_funds()
            if funds_response['status'] == 'success':
                print("✅ OpenAlgo API validation successful")
            else:
                print(f"⚠️  OpenAlgo API warning: {funds_response}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAlgo: {e}")
            print(f"❌ OpenAlgo error: {e}")
            return False
    
    def _initialize_market_data_manager(self) -> bool:
        """Initialize market data manager."""
        try:
            print("\n=== INITIALIZING MARKET DATA MANAGER ===")
            
            self.market_data_manager = MarketDataManager(
                openalgo_client=self.openalgo_client,
                config_path=self.config_path
            )
            
            # Get active instruments
            self.active_instruments = self.config_loader.get_active_instruments()
            self.stats['total_instruments'] = len(self.active_instruments)
            
            if not self.active_instruments:
                print("❌ No active instruments found in configuration")
                return False
            
            print(f"✅ Market data manager initialized")
            print(f"   Active instruments: {self.stats['total_instruments']}")
            
            # Display instruments
            print("   Configured instruments:")
            for i, instrument in enumerate(self.active_instruments, 1):
                symbol = instrument.get('symbol')
                exchange = instrument.get('exchange')
                timeframe = instrument.get('timeframe')
                print(f"   {i:2d}. {symbol} ({exchange}, {timeframe})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize market data manager: {e}")
            print(f"❌ Market data manager error: {e}")
            return False
    
    def _initialize_database(self) -> bool:
        """Initialize database manager."""
        try:
            print("\n=== INITIALIZING DATABASE ===")
            
            self.database_manager = DatabaseManager()
            
            # Get initial database stats
            initial_stats = self.database_manager.get_database_stats()
            print(f"✅ Database initialized: {initial_stats['database_path']}")
            print(f"   Initial records: {initial_stats['total_records']}")
            print(f"   Database size: {initial_stats['database_size_mb']:.2f} MB")
            
            # Clean up any invalid timestamps (epoch, null, far future)
            print("   🧹 Cleaning up invalid timestamps...")
            invalid_cleaned = self.database_manager.cleanup_invalid_timestamps()
            if invalid_cleaned > 0:
                print(f"   ✅ Cleaned up {invalid_cleaned} invalid timestamp records")
                # Update stats after cleanup
                updated_stats = self.database_manager.get_database_stats()
                print(f"   📊 Records after cleanup: {updated_stats['total_records']}")
            else:
                print("   ✅ No invalid timestamps found - database is clean")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            print(f"❌ Database error: {e}")
            return False
    
    def _load_all_instruments(self) -> bool:
        """Load historical data for all configured instruments."""
        try:
            print("\n=== LOADING HISTORICAL DATA ===")
            
            # Format dates for API calls
            def format_date_for_api(date_value):
                """Format date for API calls (YYYY-MM-DD). Expects dd-MM-yyyy format from user."""
                date_str = str(date_value).strip()
                
                # Handle datetime objects first
                if hasattr(date_value, 'strftime'):
                    return date_value.strftime('%Y-%m-%d')
                
                # Standard format: dd-MM-yyyy (e.g., 01-01-2024)
                try:
                    # Remove any time component if present
                    if ' ' in date_str:
                        date_str = date_str.split(' ')[0]
                    
                    dt = datetime.strptime(date_str, '%d-%m-%Y')
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    # Fallback for edge cases, but prefer standard format
                    try:
                        # Try YYYY-MM-DD format (might come from API)
                        dt = datetime.strptime(date_str, '%Y-%m-%d')
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        pass
                
                raise ValueError(f"Invalid date format: {date_value}. Expected dd-MM-yyyy format (e.g., 01-01-2024)")
            
            # Format dates for API
            api_start_date = format_date_for_api(self.start_date)
            api_end_date = format_date_for_api(self.end_date)
            
            # Calculate days for info display
            start_dt = datetime.strptime(api_start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(api_end_date, '%Y-%m-%d')
            days_span = (end_dt - start_dt).days + 1
            
            print(f"Date range: {api_start_date} to {api_end_date} ({days_span} days)")
            print(f"Expected API calls per instrument: ~{days_span // 30 + 1}")
            print()
            
            # Process each instrument
            for i, instrument in enumerate(self.active_instruments, 1):
                success = self._load_single_instrument(i, instrument, api_start_date, api_end_date)
                
                if success:
                    self.stats['successful_loads'] += 1
                else:
                    self.stats['failed_loads'] += 1
            
            print(f"\n📊 Loading Summary:")
            print(f"   Successful: {self.stats['successful_loads']}/{self.stats['total_instruments']}")
            print(f"   Failed: {self.stats['failed_loads']}/{self.stats['total_instruments']}")
            print(f"   Total records loaded: {self.stats['total_records']:,}")
            
            return self.stats['successful_loads'] > 0
            
        except Exception as e:
            self.logger.error(f"Failed to load instruments: {e}")
            print(f"❌ Data loading error: {e}")
            return False
    
    def _load_single_instrument(self, index: int, instrument: Dict[str, Any], start_date: str, end_date: str) -> bool:
        """Load data for a single instrument."""
        symbol = instrument.get('symbol')
        exchange = instrument.get('exchange')
        timeframe = instrument.get('timeframe')
        
        try:
            print(f"{index:2d}. Loading {symbol} ({exchange}, {timeframe})...")
            
            # Fetch historical data
            data = self.market_data_manager.fetch_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                print(f"    ❌ No data received for {symbol}")
                self.stats['instruments_status'][symbol] = {
                    'status': 'failed',
                    'records': 0,
                    'error': 'No data received'
                }
                return False
            
            # Store in database with incremental loading (replace=False)
            success = self.database_manager.store_ohlcv_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data=data,
                replace=False  # Incremental loading - only insert new records
            )
            
            if success:
                record_count = len(data)
                self.stats['total_records'] += record_count
                
                # Safe date range calculation
                try:
                    if not data.empty and hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                        date_range = f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"
                    else:
                        date_range = f"{start_date} to {end_date}"
                except (AttributeError, ValueError):
                    date_range = f"{start_date} to {end_date}"
                
                print(f"    ✅ {record_count:,} records ({date_range})")
                
                self.stats['instruments_status'][symbol] = {
                    'status': 'success',
                    'records': record_count,
                    'date_range': date_range
                }
                
                # Export to CSV if enabled
                if self.data_export_enabled:
                    self._export_instrument_to_csv(symbol, exchange, timeframe, start_date, end_date)
                
                return True
            else:
                print(f"    ❌ Failed to store {symbol} data")
                self.stats['instruments_status'][symbol] = {
                    'status': 'failed',
                    'records': 0,
                    'error': 'Database storage failed'
                }
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to load {symbol}: {e}")
            print(f"    ❌ Error loading {symbol}: {e}")
            self.stats['instruments_status'][symbol] = {
                'status': 'failed',
                'records': 0,
                'error': str(e)
            }
            return False
    
    def _export_instrument_to_csv(self, symbol: str, exchange: str, timeframe: str, 
                                 start_date: str, end_date: str) -> None:
        """Export ALL available instrument data from database to CSV file."""
        try:
            # Create exports directory
            export_dir = Path("outputs/data_exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename: SYMBOL_TIMEFRAME.csv
            filename = f"{symbol}_{timeframe}.csv"
            export_path = export_dir / filename
            
            # Get actual database coverage for this instrument
            try:
                coverage_query = """
                SELECT MIN(DATE(timestamp)) as min_date, MAX(DATE(timestamp)) as max_date, COUNT(*) as total_records
                FROM ohlcv_data 
                WHERE symbol = ? AND exchange = ? AND timeframe = ?
                """
                coverage_result = self.database_manager.connection.execute(coverage_query, [
                    symbol.upper(), exchange.upper(), timeframe.lower()
                ]).fetchone()
                
                if coverage_result and coverage_result[0]:
                    db_start, db_end, db_records = coverage_result
                    coverage_info = f"all {db_records:,} records ({db_start} to {db_end})"
                else:
                    coverage_info = "no data found in database"
            except Exception:
                coverage_info = "all available records"
            
            print(f"    📄 Exporting to CSV: {filename} ({coverage_info})...")
            
            # Export ALL data from database (no date range filter)
            success = self.database_manager.export_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                output_path=str(export_path),
                format="csv"
                # No start_date/end_date = exports all database records
            )
            
            if success and export_path.exists():
                # Get file size and record count
                file_size_kb = export_path.stat().st_size / 1024
                
                # Get actual exported record count from database
                try:
                    count_query = """
                    SELECT COUNT(*) FROM ohlcv_data 
                    WHERE symbol = ? AND exchange = ? AND timeframe = ?
                    """
                    record_count = self.database_manager.connection.execute(count_query, [
                        symbol.upper(), exchange.upper(), timeframe.lower()
                    ]).fetchone()[0]
                    
                    print(f"    ✅ CSV exported: {export_path.name} ({record_count:,} records, {file_size_kb:.1f} KB)")
                except Exception:
                    print(f"    ✅ CSV exported: {export_path.name} ({file_size_kb:.1f} KB)")
                
                # Update stats
                if symbol not in self.stats['instruments_status']:
                    self.stats['instruments_status'][symbol] = {}
                self.stats['instruments_status'][symbol]['csv_export'] = {
                    'path': str(export_path),
                    'size_kb': round(file_size_kb, 1)
                }
            else:
                print(f"    ⚠️  CSV export failed for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error exporting {symbol} to CSV: {e}")
            print(f"    ⚠️  CSV export error for {symbol}: {e}")
    
    def _generate_summary_report(self) -> None:
        """Generate final summary report."""
        # Ensure end_time is set
        if self.stats['end_time'] is None:
            self.stats['end_time'] = time.time()
        
        elapsed_time = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "=" * 60)
        print("📊 DATA LOADING SUMMARY REPORT")
        print("=" * 60)
        
        # Overall statistics
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"📈 Instruments processed: {self.stats['total_instruments']}")
        print(f"✅ Successful loads: {self.stats['successful_loads']}")
        print(f"❌ Failed loads: {self.stats['failed_loads']}")
        print(f"📊 Total records: {self.stats['total_records']:,}")
        
        # Database statistics
        final_stats = self.database_manager.get_database_stats()
        print(f"🗄️  Database size: {final_stats['database_size_mb']:.2f} MB")
        print(f"📅 Date range: {final_stats['start_date']} to {final_stats['end_date']}")
        
        # Per-instrument details
        print(f"\n📋 Per-Instrument Status:")
        for symbol, status in self.stats['instruments_status'].items():
            status_icon = "✅" if status['status'] == 'success' else "❌"
            if status['status'] == 'success':
                base_info = f"   {status_icon} {symbol:12s}: {status['records']:6,} records ({status['date_range']})"
                if 'csv_export' in status:
                    csv_info = status['csv_export']
                    base_info += f" | CSV: {csv_info['size_kb']} KB"
                print(base_info)
            else:
                print(f"   {status_icon} {symbol:12s}: {status['error']}")
        
        # CSV Export summary
        if self.data_export_enabled:
            exported_files = [status.get('csv_export') for status in self.stats['instruments_status'].values() 
                            if status.get('csv_export')]
            if exported_files:
                print(f"\n📄 CSV Exports:")
                total_size_kb = sum(export['size_kb'] for export in exported_files)
                print(f"   Files exported: {len(exported_files)}")
                print(f"   Total size: {total_size_kb:.1f} KB")
                print(f"   Location: outputs/data_exports/")
                for status in self.stats['instruments_status'].values():
                    if 'csv_export' in status:
                        csv_info = status['csv_export']
                        filename = Path(csv_info['path']).name
                        print(f"   • {filename} ({csv_info['size_kb']} KB)")
        
        # Success rate
        success_rate = (self.stats['successful_loads'] / self.stats['total_instruments']) * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("🎉 ALL INSTRUMENTS LOADED SUCCESSFULLY!")
            print("\n✅ Your vAlgo system is ready for:")
            print("   - Backtesting operations")
            print("   - Indicator calculations")
            print("   - Strategy development")
            print("   - Live trading preparation")
            if self.data_export_enabled:
                print("   - Manual data validation (CSV files exported)")
        elif success_rate >= 80:
            print("✅ Most instruments loaded successfully!")
            print("⚠️  Review failed instruments and retry if needed")
        else:
            print("⚠️  Many instruments failed to load")
            print("Please check configuration and OpenAlgo connection")
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.openalgo_client:
                self.openalgo_client.disconnect()
            
            if self.database_manager:
                self.database_manager.close()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _get_existing_data_summary(self) -> str:
        """Get summary of existing data in database."""
        try:
            if not self.database_manager:
                return ""
            
            # Get data summary from database
            summary_query = """
            SELECT symbol, exchange, timeframe, 
                   MIN(DATE(timestamp)) as start_date,
                   MAX(DATE(timestamp)) as end_date,
                   COUNT(*) as records
            FROM ohlcv_data 
            GROUP BY symbol, exchange, timeframe
            ORDER BY symbol
            """
            
            result = self.database_manager.connection.execute(summary_query).fetchall()
            
            if not result:
                return ""
            
            summaries = []
            for row in result:
                symbol, exchange, timeframe, start_date, end_date, records = row
                summaries.append(f"{symbol} ({timeframe}): {records:,} records ({start_date} to {end_date})")
            
            return "; ".join(summaries)
            
        except Exception as e:
            self.logger.error(f"Error getting existing data summary: {e}")
            return ""


def main():
    """Main entry point for data loading script."""
    try:
        # Initialize and run data loading
        data_loader = DataLoadManager()
        success = data_loader.run_data_load()
        
        # Exit with appropriate code
        if success:
            print(f"\n🚀 Data loading completed successfully!")
            print(f"Database ready for backtesting and strategy development.")
            sys.exit(0)
        else:
            print(f"\n❌ Data loading failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Data loading interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()