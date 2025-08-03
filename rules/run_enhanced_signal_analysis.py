#!/usr/bin/env python3
"""
Enhanced Signal Analysis Runner for vAlgo Rule Engine

Runs Excel-driven strategy analysis with:
- Row-by-row strategy processing (1:1 entry-exit pairing)
- Active status filtering within rules
- condition_order based AND/OR logic
- Entry-exit state machine tracking
- Complete configuration-driven approach

Usage:
    python rules/run_enhanced_signal_analysis.py
    python rules/run_enhanced_signal_analysis.py --config config/config.xlsx
    python rules/run_enhanced_signal_analysis.py --output enhanced_analysis.csv
"""

import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rules import EnhancedRuleEngine, EnhancedSignalProcessor
from utils.initialize_config_loader import InitializeConfigLoader


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main function for enhanced signal analysis"""
    parser = argparse.ArgumentParser(
        description="Run enhanced signal analysis using vAlgo Enhanced Rule Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Features:
  • Row-by-row strategy processing (1:1 entry-exit pairing)
  • Active status filtering within rules  
  • condition_order based AND/OR logic evaluation
  • Entry-exit state machine for trade lifecycle tracking
  • Complete Excel-driven configuration

Examples:
  %(prog)s                                    # Use config from Initialize sheet
  %(prog)s --config config/my_config.xlsx    # Use custom config file
  %(prog)s --output enhanced_results.csv     # Custom output file
  %(prog)s --start-date 2024-01-01 --end-date 2024-12-31  # Custom date range
  %(prog)s --dry-run --verbose               # Validate configuration only
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.xlsx',
        help='Path to Excel configuration file (default: config/config.xlsx)'
    )
    
    parser.add_argument(
        '--signal-data',
        type=str,
        help='Path to signal data CSV file (overrides config)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for analysis (YYYY-MM-DD, overrides config)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for analysis (YYYY-MM-DD, overrides config)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'excel'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without processing data'
    )
    
    parser.add_argument(
        '--show-lifecycle',
        action='store_true',
        help='Show detailed trade lifecycle report'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== vAlgo Enhanced Signal Analysis Runner ===")
        logger.info(f"Configuration file: {args.config}")
        
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        # Load and validate initialization config
        logger.info("Loading initialization configuration...")
        init_config = InitializeConfigLoader(str(config_path))
        if not init_config.load_config():
            logger.error("Failed to load initialization configuration")
            return 1
        
        # Get configuration parameters
        init_params = init_config.get_initialize_parameters()
        signal_data_path = args.signal_data or init_params.get('signal_data_path', '')
        start_date = args.start_date or init_params.get('start_date', '')
        end_date = args.end_date or init_params.get('end_date', '')
        
        logger.info(f"Signal data path: {signal_data_path}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Create enhanced rule engine
        logger.info("Initializing Enhanced Rule Engine...")
        engine = EnhancedRuleEngine(config_path=str(config_path))
        
        # Get engine state
        engine_state = engine.get_engine_state()
        logger.info(f"Enhanced Rule Engine loaded:")
        logger.info(f"  - Strategy rows: {engine_state['strategy_rows_count']}")
        logger.info(f"  - Entry rules: {engine_state['entry_rules_count']}")
        logger.info(f"  - Exit rules: {engine_state['exit_rules_count']}")
        
        # Show strategy summary
        if args.verbose:
            strategy_summary = engine.get_strategy_summary()
            logger.debug("Strategy breakdown:")
            for strategy in strategy_summary:
                logger.debug(f"  - {strategy['strategy_name']}: "
                           f"{strategy['entry_rule']}({strategy['entry_conditions_count']}) → "
                           f"{strategy['exit_rule']}({strategy['exit_conditions_count']})")
        
        # Validate initialization parameters (including date range)
        init_validation_issues = init_config.validate_parameters()
        if init_validation_issues:
            logger.info("Configuration validation results:")
            for issue in init_validation_issues:
                if issue.startswith("INFO:"):
                    logger.info(f"  ℹ️  {issue[5:].strip()}")
                elif issue.startswith("WARNING:"):
                    logger.warning(f"  ⚠️  {issue[8:].strip()}")
                else:
                    logger.warning(f"  ⚠️  {issue}")
        
        # Validate rule engine configuration
        engine_validation_issues = engine.validate_configuration()
        if engine_validation_issues:
            logger.warning(f"Rule engine validation issues: {engine_validation_issues}")
        else:
            logger.info("Rule engine validation passed")
        
        # Dry run mode - just validate configuration
        if args.dry_run:
            logger.info("=== DRY RUN MODE - Enhanced Configuration Validation ===")
            
            # Show state machine state
            state_machine_state = engine_state['state_machine_state']
            logger.info(f"State machine initialized: {state_machine_state['waiting_entries_count']} waiting entries")
            
            # Show strategy rows details
            strategy_rows = engine.get_strategy_rows()
            logger.info(f"Active strategy rows: {len(strategy_rows)}")
            for row in strategy_rows[:5]:  # Show first 5
                logger.info(f"  - {row.strategy_name}: {row.entry_rule} → {row.exit_rule} "
                           f"(Position: {row.position_size})")
            
            logger.info("Enhanced dry run completed successfully")
            return 0
        
        # Validate signal data
        if not signal_data_path:
            logger.error("Signal data path not configured. Set 'signal_data_path' in Initialize sheet or use --signal-data")
            return 1
        
        if not Path(signal_data_path).exists():
            logger.error(f"Signal data file not found: {signal_data_path}")
            return 1
        
        # Load signal data
        logger.info("Loading signal data...")
        df = pd.read_csv(signal_data_path)
        logger.info(f"Loaded signal data: {len(df)} rows, {len(df.columns)} columns")
        
        # Show actual date range in signal data
        timestamp_columns = ['timestamp', 'datetime', 'date_time', 'time']
        timestamp_col = None
        for col in timestamp_columns:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            actual_start = df[timestamp_col].min().strftime('%Y-%m-%d')
            actual_end = df[timestamp_col].max().strftime('%Y-%m-%d')
            logger.info(f"Signal data date range: {actual_start} to {actual_end}")
            
            # Filter by date range if configured
            if start_date and end_date:
                logger.info(f"Applying date filter: {start_date} to {end_date}")
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                mask = (df[timestamp_col].dt.date >= start_dt.date()) & \
                       (df[timestamp_col].dt.date <= end_dt.date())
                original_rows = len(df)
                df = df[mask]
                filtered_rows = len(df)
                
                if filtered_rows == 0:
                    logger.error(f"No data found for date range {start_date} to {end_date}")
                    logger.error(f"Available data: {actual_start} to {actual_end}")
                    return 1
                elif filtered_rows < original_rows:
                    logger.info(f"Date filtering: {filtered_rows}/{original_rows} rows kept")
                else:
                    logger.info(f"Date filtering: All {filtered_rows} rows within range")
        else:
            logger.warning("No timestamp column found - skipping date filtering")
        
        # Generate output file path if not provided
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs/enhanced_signal_analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if args.format == 'excel':
                args.output = str(output_dir / f"enhanced_analysis_{timestamp}.xlsx")
            else:
                args.output = str(output_dir / f"enhanced_analysis_{timestamp}.csv")
        
        logger.info(f"Output file: {args.output}")
        
        # Process signal data using enhanced processor
        logger.info("Processing signal data with enhanced rule engine...")
        processor = EnhancedSignalProcessor(str(config_path))
        
        # Process signal data
        success = processor.process_signal_data_batch(df)
        
        if not success:
            logger.error("Enhanced signal data processing failed")
            return 1
        
        # Get results
        results = processor.get_processing_results()
        logger.info(f"Processed {len(results)} signals with enhanced rule engine")
        
        # Export results
        logger.info("Exporting enhanced results...")
        if not processor.export_results(args.output, args.format):
            logger.error("Failed to export enhanced results")
            return 1
        
        # Calculate and display enhanced summary statistics
        logger.info("=== Enhanced Analysis Summary ===")
        
        total_entries = sum(len(r.get('entries_triggered', [])) for r in results)
        total_exits = sum(len(r.get('exits_triggered', [])) for r in results)
        total_errors = sum(len(r.get('errors', [])) for r in results)
        
        entry_rate = total_entries / len(results) if results else 0
        exit_rate = total_exits / len(results) if results else 0
        
        logger.info(f"Signals processed: {len(results)}")
        logger.info(f"Entry signals: {total_entries} ({entry_rate:.1%} rate)")
        logger.info(f"Exit signals: {total_exits} ({exit_rate:.1%} rate)")
        logger.info(f"Errors: {total_errors}")
        
        # Get trade lifecycle report
        if args.show_lifecycle:
            logger.info("=== Trade Lifecycle Report ===")
            lifecycle_report = processor.rule_engine.get_trade_lifecycle_report()
            
            logger.info(f"Total trade trackers: {lifecycle_report['total_trackers']}")
            logger.info(f"Entry trigger rate: {lifecycle_report['entry_trigger_rate']:.1%}")
            logger.info(f"Exit trigger rate: {lifecycle_report['exit_trigger_rate']:.1%}")
            logger.info(f"Completion rate: {lifecycle_report['completion_rate']:.1%}")
            
            # Strategy breakdown
            if 'strategy_breakdown' in lifecycle_report:
                logger.info("Strategy performance:")
                for strategy, stats in lifecycle_report['strategy_breakdown'].items():
                    completion = stats['completed'] / stats['total'] if stats['total'] > 0 else 0
                    logger.info(f"  - {strategy}: {stats['total']} total, "
                               f"{stats['entries']} entries, {stats['completed']} completed ({completion:.1%})")
        
        logger.info(f"Enhanced results exported to: {args.output}")
        logger.info("=== Enhanced Signal Analysis Completed Successfully ===")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Enhanced analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error in enhanced analysis: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())