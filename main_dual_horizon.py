"""
Enhanced Main Entry Point for Dual-Horizon Crypto Trading RL System

Integrates the dual-horizon reward system with enhanced configuration,
training, and evaluation capabilities.
"""

import logging
import sys
import argparse
from pathlib import Path
import multiprocessing as mp

# --- Initial Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

def setup_enhanced_cli():
    """Configures the enhanced command-line interface with dual-horizon support."""
    parser = argparse.ArgumentParser(
        description="Enhanced Zero1: Dual-Horizon Reinforcement Learning System for Crypto Trading.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # --- Process Command ---
    process_parser = subparsers.add_parser(
        'process',
        help="Process raw trade data into enriched Parquet files.",
        description="Ingests raw trade data (zip files) and converts it into a clean, processed format with advanced features."
    )

    process_parser.add_argument(
        '--period',
        type=str,
        default='in_sample',
        choices=['in_sample', 'out_of_sample'],
        help="The data period to process (default: in_sample)."
    )

    process_parser.add_argument(
        '--force',
        action='store_true',
        help="Force reprocessing of files even if they already exist."
    )

    # --- Enhanced Train Command ---
    train_parser = subparsers.add_parser(
        'train',
        help="Train a new PPO agent using the dual-horizon reward system.",
        description="Enhanced training pipeline with dual-horizon rewards, hyperparameter optimization, and advanced monitoring."
    )

    train_parser.add_argument(
        '--trials',
        type=int,
        default=20,
        help="Number of Optuna hyperparameter optimization trials to run (set to 0 to skip)."
    )

    train_parser.add_argument(
        '--steps',
        type=int,
        default=500000,
        help="Number of timesteps for the final training phase after optimization."
    )

    train_parser.add_argument(
        '--wandb',
        action='store_true',
        help="Enable experiment tracking with Weights & Biases (requires WANDB_API_KEY)."
    )

    # ✅ NEW: Dual-horizon specific training parameters
    train_parser.add_argument(
        '--reward-horizon',
        type=int,
        default=9,
        help="Reward horizon in steps for delayed evaluation (default: 9 = 3 minutes)."
    )

    train_parser.add_argument(
        '--immediate-weight',
        type=float,
        default=0.3,
        help="Weight for immediate realized PnL rewards (default: 0.3)."
    )

    train_parser.add_argument(
        '--delayed-weight', 
        type=float,
        default=0.7,
        help="Weight for delayed future-looking rewards (default: 0.7)."
    )

    train_parser.add_argument(
        '--leverage',
        type=float,
        default=10.0,
        help="Trading leverage for reward scaling (default: 10.0)."
    )

    # --- Enhanced Backtest Command ---
    backtest_parser = subparsers.add_parser(
        'backtest',
        help="Evaluate a trained dual-horizon model's performance.",
        description="Comprehensive backtesting with dual-horizon reward analysis and advanced metrics."
    )

    backtest_parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to the model .zip file to evaluate (defaults to the path in config)."
    )

    backtest_parser.add_argument(
        '--period',
        type=str,
        default='out_of_sample',
        choices=['in_sample', 'out_of_sample'],
        help="The data period to backtest on (default: out_of_sample)."
    )

    backtest_parser.add_argument(
        '--analyze-promises',
        action='store_true',
        help="Enable detailed promise fulfillment analysis."
    )

    # --- Enhanced Pipeline Command ---
    pipeline_parser = subparsers.add_parser(
        'run-pipeline',
        help="Run the enhanced end-to-end dual-horizon pipeline: process -> train -> backtest.",
        description="Complete workflow with dual-horizon reward system and advanced analytics."
    )

    pipeline_parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help="Number of optimization trials for the training step."
    )

    pipeline_parser.add_argument(
        '--steps',
        type=int,
        default=200000,
        help="Number of timesteps for the final training step."
    )

    pipeline_parser.add_argument(
        '--leverage',
        type=float,
        default=10.0,
        help="Trading leverage for training."
    )

    pipeline_parser.add_argument(
        '--reward-profile',
        type=str,
        default='balanced',
        choices=['aggressive', 'balanced', 'conservative'],
        help="Reward profile: aggressive (0.2/0.8), balanced (0.3/0.7), conservative (0.5/0.5)."
    )

    # ✅ NEW: Analysis Command
    analysis_parser = subparsers.add_parser(
        'analyze',
        help="Analyze dual-horizon reward performance and promise fulfillment.",
        description="Advanced analysis of reward components, promise success rates, and trading patterns."
    )

    analysis_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help="Path to the trained model for analysis."
    )

    analysis_parser.add_argument(
        '--period',
        type=str,
        default='out_of_sample',
        choices=['in_sample', 'out_of_sample'],
        help="Data period for analysis."
    )

    analysis_parser.add_argument(
        '--output-dir',
        type=str,
        default='./analysis_results',
        help="Directory to save analysis results."
    )

    return parser

def main():
    """Enhanced main function with dual-horizon system integration."""
    parser = setup_enhanced_cli()
    args = parser.parse_args()

    # --- System Initialization ---
    try:
        print("🚀 Starting Enhanced Zero1 Dual-Horizon Crypto Trading RL System 🚀")
        print("✨ NEW: Forward-looking delayed rewards + Immediate realized PnL tracking")
        
        # Set multiprocessing start method for compatibility
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        # Import and setup enhanced configuration
        from config_enhanced import SETTINGS, setup_environment, validate_configuration

        # Configure enhanced logging
        log_file = Path(SETTINGS.get_logs_path()) / "dual_horizon_system.log"
        log_file.parent.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

        # Setup environment and validate configuration
        setup_environment()
        warnings = validate_configuration(SETTINGS)
        
        if warnings:
            logger.warning("Configuration warnings detected:")
            for warning in warnings:
                logger.warning(f" - {warning}")

        logger.info(f"Environment: {SETTINGS.environment.value}")
        logger.info(f"Primary Asset: {SETTINGS.primary_asset}")
        logger.info(f"Dual-Horizon Config:")
        logger.info(f"  - Reward Horizon: {getattr(SETTINGS.strategy, 'reward_horizon_steps', 9)} steps")
        logger.info(f"  - Immediate Weight: {getattr(SETTINGS.strategy, 'immediate_reward_weight', 0.3):.1%}")
        logger.info(f"  - Delayed Weight: {getattr(SETTINGS.strategy, 'delayed_reward_weight', 0.7):.1%}")

    except Exception as e:
        logger.error(f"Fatal error during system initialization: {e}", exc_info=True)
        sys.exit(1)

    # --- Command Dispatcher ---
    if args.command == 'process':
        run_processing(args)
    elif args.command == 'train':
        run_enhanced_training(args)
    elif args.command == 'backtest':
        run_enhanced_evaluation(args)
    elif args.command == 'analyze':
        run_dual_horizon_analysis(args)
    elif args.command == 'run-pipeline':
        logger.info("🔄 Starting Enhanced End-to-End Dual-Horizon Pipeline...")
        run_processing(argparse.Namespace(period='in_sample', force=True))
        
        # Convert reward profile to weights
        profile_weights = get_reward_profile_weights(args.reward_profile)
        enhanced_args = argparse.Namespace(
            trials=args.trials,
            steps=args.steps,
            wandb=False,
            reward_horizon=9,
            immediate_weight=profile_weights['immediate'],
            delayed_weight=profile_weights['delayed'],
            leverage=args.leverage
        )
        
        run_enhanced_training(enhanced_args)
        run_enhanced_evaluation(argparse.Namespace(
            model_path=None, 
            period='out_of_sample',
            analyze_promises=True
        ))
        
        logger.info("✅ Enhanced End-to-End Dual-Horizon Pipeline Completed!")

def get_reward_profile_weights(profile: str) -> Dict[str, float]:
    """Convert reward profile to weight settings."""
    profiles = {
        'aggressive': {'immediate': 0.2, 'delayed': 0.8},    # Focus on strategic entries
        'balanced': {'immediate': 0.3, 'delayed': 0.7},      # Default balanced approach  
        'conservative': {'immediate': 0.5, 'delayed': 0.5}   # Equal weighting
    }
    return profiles.get(profile, profiles['balanced'])

def run_processing(args):
    """Handles the data processing command."""
    try:
        logger.info(f"--- Starting Enhanced Data Processing for '{args.period}' period ---")
        from processor import process_trades_for_period
        
        process_trades_for_period(args.period, force_reprocess=args.force)
        logger.info(f"✅ Enhanced data processing for '{args.period}' completed successfully.")
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        sys.exit(1)

def run_enhanced_training(args):
    """Handles the enhanced dual-horizon model training."""
    try:
        logger.info("--- Starting Enhanced Dual-Horizon Model Training ---")
        logger.info(f"🎯 Reward Configuration:")
        logger.info(f"   - Horizon: {getattr(args, 'reward_horizon', 9)} steps")
        logger.info(f"   - Immediate: {getattr(args, 'immediate_weight', 0.3):.1%}")
        logger.info(f"   - Delayed: {getattr(args, 'delayed_weight', 0.7):.1%}")
        logger.info(f"   - Leverage: {getattr(args, 'leverage', 10.0)}x")
        logger.info(f"📊 Training: {args.trials} trials, {args.steps:,} steps, W&B: {args.wandb}")
        
        # Import enhanced trainer
        from trainer_dual_horizon import train_dual_horizon_model_advanced
        
        # Prepare dual-horizon configuration
        dual_horizon_config = {
            'reward_horizon_steps': getattr(args, 'reward_horizon', 9),
            'immediate_reward_weight': getattr(args, 'immediate_weight', 0.3),
            'delayed_reward_weight': getattr(args, 'delayed_weight', 0.7),
            'leverage': getattr(args, 'leverage', 10.0)
        }
        
        train_dual_horizon_model_advanced(
            optimization_trials=args.trials,
            final_training_steps=args.steps,
            use_wandb=args.wandb,
            dual_horizon_config=dual_horizon_config
        )
        
        logger.info("✅ Enhanced dual-horizon model training completed successfully.")
        
    except Exception as e:
        logger.error(f"Enhanced model training failed: {e}", exc_info=True)
        sys.exit(1)

def run_enhanced_evaluation(args):
    """Handles the enhanced dual-horizon model evaluation."""
    try:
        logger.info(f"--- Starting Enhanced Dual-Horizon Evaluation on '{args.period}' period ---")
        
        # Import enhanced evaluator
        from evaluator_dual_horizon import run_dual_horizon_backtest
        
        # Check if model exists
        from config_enhanced import SETTINGS
        model_path = args.model_path or SETTINGS.get_model_path()
        
        if not Path(model_path).exists():
            logger.error(f"Model file not found at '{model_path}'.")
            logger.error("Please train a model first using the 'train' command or provide a valid path with --model-path.")
            sys.exit(1)

        logger.info(f"Using model: {model_path}")
        
        # Run enhanced backtest with promise analysis
        results = run_dual_horizon_backtest(
            model_path=model_path,
            period=args.period,
            analyze_promises=getattr(args, 'analyze_promises', False)
        )
        
        # Display dual-horizon specific results
        if results and 'dual_horizon_metrics' in results:
            metrics = results['dual_horizon_metrics']
            logger.info("🎯 DUAL-HORIZON PERFORMANCE SUMMARY:")
            logger.info(f"   - Promise Success Rate: {metrics.get('promise_success_rate', 0):.1%}")
            logger.info(f"   - Avg Delayed Reward: {metrics.get('avg_delayed_reward', 0):.4f}")
            logger.info(f"   - Total Promises: {metrics.get('successful_promises', 0) + metrics.get('failed_promises', 0)}")
            logger.info(f"   - Realized PnL Entries: {metrics.get('realized_pnl_entries', 0)}")
            logger.info(f"   - Delayed Component: {metrics.get('total_delayed_rewards', 0):.4f}")
            logger.info(f"   - Immediate Component: {metrics.get('total_immediate_rewards', 0):.4f}")
        
        logger.info("✅ Enhanced dual-horizon evaluation completed successfully.")
        
    except Exception as e:
        logger.error(f"Enhanced model evaluation failed: {e}", exc_info=True)
        sys.exit(1)

def run_dual_horizon_analysis(args):
    """Handles the dual-horizon specific analysis command."""
    try:
        logger.info(f"--- Starting Dual-Horizon Reward Analysis ---")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Period: {args.period}")
        logger.info(f"Output: {args.output_dir}")
        
        # Import analysis module
        from analysis_dual_horizon import DualHorizonAnalyzer
        
        # Create analyzer and run comprehensive analysis
        analyzer = DualHorizonAnalyzer(
            model_path=args.model_path,
            output_dir=args.output_dir
        )
        
        analysis_results = analyzer.run_comprehensive_analysis(
            period=args.period
        )
        
        # Display key findings
        logger.info("🔍 ANALYSIS RESULTS:")
        for key, value in analysis_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"   - {key}: {value}")
            elif isinstance(value, dict):
                logger.info(f"   - {key}: {len(value)} items")
        
        logger.info(f"📊 Detailed results saved to: {args.output_dir}")
        logger.info("✅ Dual-horizon analysis completed successfully.")
        
    except ImportError:
        logger.error("Analysis module not available. Implement analysis_dual_horizon.py for detailed analysis.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Dual-horizon analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()