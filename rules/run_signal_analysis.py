#!/usr/bin/env python3
"""
Signal Analysis Runner for vAlgo Rule Engine

Standalone script to run signal data analysis using the configured strategy pipeline:
Strategy_Config → Entry_Conditions → Exit_Conditions

Usage:
    python rules/run_signal_analysis.py
    python rules/run_signal_analysis.py --config config/config.xlsx
    python rules/run_signal_analysis.py --output outputs/signal_analysis.csv
    python rules/run_signal_analysis.py --start-date 2024-01-01 --end-date 2024-12-31
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rules import RuleEngine
from rules.signal_processor import SignalDataProcessor
from utils.initialize_config_loader import InitializeConfigLoader


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/signal_analysis.log', mode='a')
        ]
    )


def main():
    """Main function for signal analysis"""
    parser = argparse.ArgumentParser(
        description="Run signal data analysis using vAlgo Rule Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use config from Initialize sheet
  %(prog)s --config config/my_config.xlsx    # Use custom config file
  %(prog)s --output results/analysis.csv     # Custom output file
  %(prog)s --start-date 2024-01-01 --end-date 2024-12-31  # Custom date range
  %(prog)s --verbose                          # Enable debug logging
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
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== vAlgo Signal Analysis Runner ===")
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
        
        # Validate parameters
        issues = init_config.validate_parameters()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
        
        # Get configuration parameters
        init_params = init_config.get_initialize_parameters()
        signal_data_path = args.signal_data or init_params.get('signal_data_path', '')
        start_date = args.start_date or init_params.get('start_date', '')
        end_date = args.end_date or init_params.get('end_date', '')
        
        logger.info(f"Signal data path: {signal_data_path}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Validate signal data path
        if not signal_data_path:
            logger.error("Signal data path not configured. Set 'signal_data_path' in Initialize sheet or use --signal-data")
            return 1
        
        if not Path(signal_data_path).exists():
            logger.error(f"Signal data file not found: {signal_data_path}")
            return 1
        
        # Create rule engine
        logger.info("Initializing Rule Engine...")
        rule_engine = RuleEngine(config_path=str(config_path))
        
        # Get engine info
        engine_info = rule_engine.get_engine_info()
        logger.info(f"Rule Engine loaded: {engine_info['entry_rules_count']} entry rules, "
                   f"{engine_info['exit_rules_count']} exit rules, "
                   f"{engine_info['strategies_count']} strategies")
        
        if args.verbose:
            logger.debug(f"Available strategies: {engine_info['strategies']}")
            logger.debug(f"Available entry rules: {engine_info['entry_rules']}")
            logger.debug(f"Available exit rules: {engine_info['exit_rules']}")
        
        # Dry run mode - just validate configuration
        if args.dry_run:
            logger.info("=== DRY RUN MODE - Configuration Validation ===")
            
            # Validate all rules
            validation_results = rule_engine.validate_all_rules()
            if validation_results:
                logger.warning(f"Rule validation issues: {validation_results}")
            else:
                logger.info("All rules validated successfully")
            
            # Check strategies
            strategies = rule_engine.get_available_strategies()
            for strategy_name in strategies:
                strategy_config = rule_engine.get_strategy_config(strategy_name)
                if strategy_config:
                    status = strategy_config.get('status', 'Unknown')
                    entry_rules = strategy_config.get('entry_rules', [])
                    exit_rules = strategy_config.get('exit_rules', [])
                    logger.info(f"Strategy '{strategy_name}': {status}, "
                               f"{len(entry_rules)} entry rules, {len(exit_rules)} exit rules")
            
            logger.info("Dry run completed successfully")
            return 0
        
        # Generate output file path if not provided
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs/signal_analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if args.format == 'excel':
                args.output = str(output_dir / f"signal_analysis_{timestamp}.xlsx")
            else:
                args.output = str(output_dir / f"signal_analysis_{timestamp}.csv")
        
        logger.info(f"Output file: {args.output}")
        
        # Create signal processor
        logger.info("Initializing Signal Data Processor...")
        processor = SignalDataProcessor(str(config_path))
        processor.set_rule_engine(rule_engine)
        
        # Override configuration if provided
        if args.signal_data:
            processor.signal_data_path = args.signal_data
        if args.start_date:
            processor.start_date = args.start_date
        if args.end_date:
            processor.end_date = args.end_date
        
        # Process signal data
        logger.info("Processing signal data...")
        if not processor.process_signal_data():
            logger.error("Signal data processing failed")
            return 1
        
        # Export results
        logger.info("Exporting results...")
        if not processor.export_results(args.output, args.format):
            logger.error("Failed to export results")
            return 1
        
        # Get and display summary statistics
        stats = processor.get_summary_statistics()
        logger.info("=== Analysis Summary ===")
        logger.info(f"Total signals processed: {stats['total_signals_processed']}")
        logger.info(f"Entry signals triggered: {stats['entry_signals_triggered']} ({stats['entry_signal_rate']:.1%})")
        logger.info(f"Exit signals triggered: {stats['exit_signals_triggered']} ({stats['exit_signal_rate']:.1%})")
        logger.info(f"No signals: {stats['no_signals']}")
        logger.info(f"Date range: {stats['date_range']}")
        
        # Strategy breakdown
        if args.verbose and 'strategy_breakdown' in stats:
            logger.info("=== Strategy Breakdown ===")
            for strategy, strategy_stats in stats['strategy_breakdown'].items():
                logger.info(f"  {strategy}: {strategy_stats['total']} total, "
                           f"{strategy_stats['entry']} entry, {strategy_stats['exit']} exit")
        
        logger.info(f"Results exported to: {args.output}")
        logger.info("=== Signal Analysis Completed Successfully ===")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())