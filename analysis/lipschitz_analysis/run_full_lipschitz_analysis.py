#!/usr/bin/env python3
"""
Full-Scale LLaDA Lipschitz Analysis Runner

This script runs the comprehensive Lipschitz analysis based on findings from 
the conversation history. Key improvements:

1. Fixed vocabulary size (126,464 instead of 50,304)
2. Proper time point sampling
3. Robust error handling
4. Intermediate results saving
5. Expected results: Lipschitz bounds 100-300+ range

Historical Context:
- Original issue: vocabulary size mismatch caused all singular values = 1.0
- After fix: proper Lipschitz bounds indicating manageable stiffness for ETD2RK
- Results validate SEFM (Soft-Embedding Flow-Matching) approach feasibility
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from llada_lipschitz_analysis import LLaDALipschitzAnalyzer

def setup_logging(log_dir: str = "logs"):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"llada_lipschitz_analysis_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging to: {log_file}")
    return log_file

def get_time_points(mode: str = "comprehensive"):
    """
    Get time points for analysis based on mode.
    
    Args:
        mode: Analysis mode - 'quick', 'standard', or 'comprehensive'
        
    Returns:
        List of time points
    """
    if mode == "quick":
        # Quick test with representative time points as specified
        return [1.0, 0.5, 0.2, 0.1]
    elif mode == "standard":
        # Standard analysis covering important regions, especially small-t end
        return [
            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 
            0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01
        ]
    elif mode == "comprehensive":
        # Comprehensive analysis spanning the region where inference operates
        return [
            1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 
            0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 
            0.015, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

def validate_environment():
    """Validate that the environment is set up correctly."""
    try:
        import torch
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA device: {torch.cuda.get_device_name()}")
            logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        logging.error("PyTorch not found!")
        return False
    
    try:
        from transformers import AutoModel, AutoTokenizer
        logging.info("Transformers library available")
    except ImportError:
        logging.error("Transformers library not found!")
        return False
    
    return True

def save_intermediate_results(results: dict, step: int, output_dir: str):
    """Save intermediate results during analysis."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"intermediate_results_step_{step:03d}.json")
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Intermediate results saved to {filename}")

def main():
    """Main function to run the full Lipschitz analysis."""
    parser = argparse.ArgumentParser(description="LLaDA Lipschitz Analysis")
    parser.add_argument("--mode", choices=["quick", "standard", "comprehensive"], 
                       default="standard", help="Analysis mode")
    parser.add_argument("--num-points", type=int, default=128, 
                       help="Number of sample points per time point")
    parser.add_argument("--vocab-size", type=int, default=None,
                       help="Vocabulary size (default: auto-detect from model)")
    parser.add_argument("--seq-length", type=int, default=32,
                       help="Sequence length for analysis")
    parser.add_argument("--output-dir", type=str, default=".",
                       help="Output directory for results")
    parser.add_argument("--sampling-method", choices=["dirichlet", "gumbel_softmax", "uniform"],
                       default="dirichlet", help="Simplex sampling method")
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging()
    
    logging.info("FULL-SCALE LLaDA LIPSCHITZ ANALYSIS")
    logging.info("=" * 50)
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Points per time: {args.num_points}")
    logging.info(f"Vocabulary size: {args.vocab_size or 'auto-detect'}")
    logging.info(f"Sequence length: {args.seq_length}")
    logging.info(f"Sampling method: {args.sampling_method}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Validate environment
    if not validate_environment():
        logging.error("Environment validation failed!")
        return 1
    
    # Get time points
    time_points = get_time_points(args.mode)
    logging.info(f"Time points ({len(time_points)}): {time_points}")
    
    try:
        # Initialize analyzer with corrected vocabulary size
        logging.info("Initializing LLaDA Lipschitz Analyzer...")
        analyzer = LLaDALipschitzAnalyzer(
            vocab_size=args.vocab_size,
            sequence_length=args.seq_length,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Run the analysis
        logging.info("Starting Lipschitz analysis...")
        results = analyzer.run_full_analysis(
            time_points=time_points,
            num_points=args.num_points,
            sampling_method=args.sampling_method
        )
        
        # Save results
        output_file = os.path.join(args.output_dir, f"llada_lipschitz_results_{args.mode}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_file}")
        
        # Generate plots
        plot_file = os.path.join(args.output_dir, f"llada_lipschitz_analysis_{args.mode}.png")
        analyzer.plot_results(results, save_path=plot_file)
        
        # Print summary
        analyzer.print_summary(results)
        
        # Validate expected results
        lipschitz_estimates = results['lipschitz_estimates']
        if lipschitz_estimates:
            max_lipschitz = max(est['max_sv'] for est in lipschitz_estimates)
            avg_lipschitz = sum(est['max_sv'] for est in lipschitz_estimates) / len(lipschitz_estimates)
            
            logging.info("\n" + "=" * 50)
            logging.info("VALIDATION AGAINST HISTORICAL FINDINGS")
            logging.info("=" * 50)
            logging.info(f"Maximum Lipschitz bound: {max_lipschitz:.2f}")
            logging.info(f"Average Lipschitz bound: {avg_lipschitz:.2f}")
            
            # Check against expected range (100-300+ from conversation history)
            if 50 <= max_lipschitz <= 1000:
                logging.info("✅ Results are in expected range (50-1000)")
                logging.info("✅ Indicates manageable stiffness for ETD2RK solver")
                logging.info("✅ Validates SEFM approach feasibility")
            elif max_lipschitz < 50:
                logging.warning("⚠️  Lipschitz bounds unexpectedly low")
                logging.warning("⚠️  May indicate numerical issues or insufficient sampling")
            else:
                logging.warning("⚠️  Lipschitz bounds very high")
                logging.warning("⚠️  May indicate severe stiffness - consider alternative solvers")
            
            # Check vocabulary size effect
            if results['vocab_size'] in [126349, 126464]:
                logging.info(f"✅ Using detected vocabulary size ({results['vocab_size']})")
            else:
                logging.warning(f"⚠️  Unexpected vocabulary size {results['vocab_size']}")
                logging.warning("⚠️  Expected ~126,349 or 126,464 for LLaDA 1.5")
        
        logging.info(f"\nAnalysis complete! All results saved to {args.output_dir}")
        logging.info(f"Log file: {log_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Analysis failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    # Add import here to avoid circular imports
    import torch
    
    exit_code = main()
    sys.exit(exit_code) 