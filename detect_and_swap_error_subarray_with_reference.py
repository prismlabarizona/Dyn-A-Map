import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class SwapConfig:
    """Configuration for error detection and swapping"""
    error_threshold: float = 0.1  # Minimum error difference to trigger swap
    max_swaps_per_run: int = 3    # Maximum number of swaps allowed
    swap_strategy: str = "best_worst"  # "best_worst", "random", "sequential"
    enable_adaptive_threshold: bool = True
    adaptive_factor: float = 1.5
    min_block_size: int = 8
    enable_validation: bool = True
    save_intermediate: bool = False
    swap_within_affected_file: bool = True  # NEW: Swap within affected file, not with reference

class ErrorDetector:
    """Advanced error detection and correction for memristive crossbar arrays"""
    
    def __init__(self, config: SwapConfig = None):
        self.config = config or SwapConfig()
        self.setup_logging()
        self.swap_history = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('error_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_error_metrics(self, weights: np.ndarray, reference: np.ndarray, 
                              col_blocks: List[int]) -> Dict[str, Any]:
        """Calculate comprehensive error metrics for each block"""
        num_cols, num_rows = weights.shape
        metrics = {
            'block_errors': [],
            'block_stats': [],
            'overall_stats': {},
            'error_distribution': []
        }
        
        start = 0
        for i, cb in enumerate(col_blocks):
            block = weights[start:start+cb, :]
            ref_block = reference[start:start+cb, :]
            
            # Calculate multiple error metrics
            mae = np.mean(np.abs(block - ref_block))
            mse = np.mean((block - ref_block) ** 2)
            rmse = np.sqrt(mse)
            max_error = np.max(np.abs(block - ref_block))
            error_std = np.std(np.abs(block - ref_block))
            
            # Calculate resistance drift (assuming weights represent resistances)
            resistance_drift = np.mean(np.abs(block - ref_block) / (ref_block + 1e-9))
            
            block_metrics = {
                'block_id': i + 1,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'max_error': max_error,
                'error_std': error_std,
                'resistance_drift': resistance_drift,
                'start_col': start,
                'end_col': start + cb,
                'size': cb
            }
            
            metrics['block_errors'].append(mae)
            metrics['block_stats'].append(block_metrics)
            metrics['error_distribution'].extend(np.abs(block - ref_block).flatten())
            start += cb
        
        # Overall statistics
        metrics['overall_stats'] = {
            'total_mae': np.mean(metrics['block_errors']),
            'total_std': np.std(metrics['block_errors']),
            'worst_block': np.argmax(metrics['block_errors']),
            'best_block': np.argmin(metrics['block_errors']),
            'error_range': np.max(metrics['block_errors']) - np.min(metrics['block_errors'])
        }
        
        return metrics
    
    def select_swap_strategy(self, metrics: Dict[str, Any], 
                           col_blocks: List[int]) -> Tuple[int, int]:
        """Select swap strategy based on configuration"""
        if self.config.swap_strategy == "best_worst":
            worst_idx = metrics['overall_stats']['worst_block']
            best_idx = metrics['overall_stats']['best_block']
            return worst_idx, best_idx
        
        elif self.config.swap_strategy == "random":
            import random
            n_blocks = len(col_blocks)
            idx1, idx2 = random.sample(range(n_blocks), 2)
            return idx1, idx2
        
        elif self.config.swap_strategy == "sequential":
            # Swap adjacent blocks with high error
            block_errors = metrics['block_errors']
            worst_idx = np.argmax(block_errors)
            if worst_idx > 0 and block_errors[worst_idx - 1] > np.mean(block_errors):
                return worst_idx, worst_idx - 1
            elif worst_idx < len(block_errors) - 1 and block_errors[worst_idx + 1] > np.mean(block_errors):
                return worst_idx, worst_idx + 1
            else:
                return worst_idx, metrics['overall_stats']['best_block']
        
        else:
            raise ValueError(f"Unknown swap strategy: {self.config.swap_strategy}")
    
    def should_swap(self, metrics: Dict[str, Any]) -> bool:
        """Determine if swapping should be performed"""
        total_mae = metrics['overall_stats']['total_mae']
        error_range = metrics['overall_stats']['error_range']
        
        # Adaptive threshold
        if self.config.enable_adaptive_threshold:
            threshold = self.config.error_threshold * self.config.adaptive_factor
        else:
            threshold = self.config.error_threshold
        
        # Check if error is significant enough to warrant swapping
        if total_mae < threshold:
            self.logger.info(f"Error {total_mae:.6f} below threshold {threshold:.6f}, skipping swap")
            return False
        
        # Check if error range is significant
        if error_range < threshold * 0.5:
            self.logger.info(f"Error range {error_range:.6f} too small, skipping swap")
            return False
        
        return True
    
    def perform_swap(self, weights: np.ndarray, error_idx: int, best_idx: int,
                    col_blocks: List[int], label: str = "") -> np.ndarray:
        """Perform the actual swap operation with validation"""
        starts = np.cumsum([0] + col_blocks[:-1])
        e_start, e_end = starts[error_idx], starts[error_idx] + col_blocks[error_idx]
        b_start, b_end = starts[best_idx], starts[best_idx] + col_blocks[best_idx]
        
        width_e = e_end - e_start
        width_b = b_end - b_start
        min_width = min(width_e, width_b)
        
        if width_e != width_b:
            self.logger.warning(f"[{label}] Block widths differ ({width_e} vs {width_b}), swapping only {min_width} columns")
        
        # Store original state for validation
        original_weights = weights.copy()
        
        # Perform swap - SWAP WITHIN THE AFFECTED FILE ITSELF
        temp = weights[e_start:e_start+min_width, :].copy()
        weights[e_start:e_start+min_width, :] = weights[b_start:b_start+min_width, :]
        weights[b_start:b_start+min_width, :] = temp
        
        # Validate swap
        if self.config.enable_validation:
            if not self.validate_swap(original_weights, weights, error_idx, best_idx, col_blocks):
                self.logger.error(f"[{label}] Swap validation failed, reverting")
                weights = original_weights
                return weights
        
        # Record swap
        swap_record = {
            'error_block': error_idx + 1,
            'best_block': best_idx + 1,
            'label': label,
            'timestamp': np.datetime64('now'),
            'swap_type': 'within_affected_file'  # NEW: Track swap type
        }
        self.swap_history.append(swap_record)
        
        self.logger.info(f"[{label}] Successfully swapped error block #{error_idx+1} with best block #{best_idx+1} WITHIN AFFECTED FILE")
        return weights
    
    def validate_swap(self, original: np.ndarray, swapped: np.ndarray, 
                     error_idx: int, best_idx: int, col_blocks: List[int]) -> bool:
        """Validate that the swap operation was performed correctly"""
        starts = np.cumsum([0] + col_blocks[:-1])
        e_start, e_end = starts[error_idx], starts[error_idx] + col_blocks[error_idx]
        b_start, b_end = starts[best_idx], starts[best_idx] + col_blocks[best_idx]
        
        min_width = min(e_end - e_start, b_end - b_start)
        
        # Check if blocks were actually swapped
        error_block_swapped = np.array_equal(
            swapped[e_start:e_start+min_width, :],
            original[b_start:b_start+min_width, :]
        )
        
        best_block_swapped = np.array_equal(
            swapped[b_start:b_start+min_width, :],
            original[e_start:e_start+min_width, :]
        )
        
        return error_block_swapped and best_block_swapped
    
    def detect_and_swap(self, weights: np.ndarray, reference: np.ndarray,
                       col_blocks: List[int], label: str = "") -> np.ndarray:
        """Main detection and swap function with enhanced features"""
        num_cols = sum(col_blocks)
        num_rows = weights.size // num_cols
        
        if weights.size != num_cols * num_rows:
            raise ValueError(f"Weight array size {weights.size} doesn't match expected size {num_cols * num_rows}")
        
        weights = weights.reshape((num_cols, num_rows))
        reference = reference.reshape((num_cols, num_rows))
        
        # Calculate comprehensive error metrics
        metrics = self.calculate_error_metrics(weights, reference, col_blocks)
        
        # Log detailed error analysis
        self.logger.info(f"\n[{label}] Error Analysis:")
        self.logger.info(f"Total MAE: {metrics['overall_stats']['total_mae']:.6f}")
        self.logger.info(f"Error Range: {metrics['overall_stats']['error_range']:.6f}")
        self.logger.info(f"Worst Block: {metrics['overall_stats']['worst_block'] + 1}")
        self.logger.info(f"Best Block: {metrics['overall_stats']['best_block'] + 1}")
        
        # Check if swapping is needed
        if not self.should_swap(metrics):
            return weights
        
        # Select swap strategy
        error_idx, best_idx = self.select_swap_strategy(metrics, col_blocks)
        
        # Perform swap WITHIN THE AFFECTED FILE
        swapped_weights = self.perform_swap(weights.copy(), error_idx, best_idx, col_blocks, label)
        
        # Save intermediate results if requested
        if self.config.save_intermediate:
            self.save_intermediate_results(swapped_weights, metrics, label)
        
        return swapped_weights
    
    def save_intermediate_results(self, weights: np.ndarray, metrics: Dict[str, Any], label: str):
        """Save intermediate results for analysis"""
        timestamp = np.datetime64('now').astype(str).replace(':', '-')
        filename = f"intermediate_{label}_{timestamp}.npz"
        
        np.savez(filename, 
                weights=weights,
                block_errors=np.array(metrics['block_errors']),
                overall_stats=metrics['overall_stats'])
        
        self.logger.info(f"Saved intermediate results to {filename}")
    
    def save_swap_history(self, filename: str = "swap_history.json"):
        """Save swap history to JSON file"""
        history_data = []
        for record in self.swap_history:
            history_data.append({
                'error_block': record['error_block'],
                'best_block': record['best_block'],
                'label': record['label'],
                'timestamp': record['timestamp'].astype(str),
                'swap_type': record.get('swap_type', 'within_affected_file')
            })
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self.logger.info(f"Saved swap history to {filename}")

def detect_and_swap_error_subarray_with_reference(
    posweight_file='data/posweight3.txt',
    negweight_file='data/negweight3.txt',
    posweight_ref='data/posweight3_reference.txt',
    negweight_ref='data/negweight3_reference.txt',
    posweight_out='data/posweight3_remapped.txt',
    negweight_out='data/negweight3_remapped.txt',
    num_rows=10, 
    col_blocks=[32, 32, 20],
    config: SwapConfig = None,
    verbose=True
):
    """
    Enhanced error detection and subarray swapping with reference weights.
    
    KEY FEATURE: Swaps subarrays WITHIN the affected file itself, not with reference values.
    This preserves actual weight values while reorganizing them for better performance.
    
    Args:
        posweight_file: Path to positive weight file (affected file)
        negweight_file: Path to negative weight file (affected file)
        posweight_ref: Path to positive weight reference file (for error detection only)
        negweight_ref: Path to negative weight reference file (for error detection only)
        posweight_out: Output path for swapped positive weights
        negweight_out: Output path for swapped negative weights
        num_rows: Number of rows in the weight matrix
        col_blocks: List of column block sizes
        config: Configuration object for error detection
        verbose: Enable verbose logging
    """
    
    # Initialize error detector
    detector = ErrorDetector(config)
    
    if not verbose:
        detector.logger.setLevel(logging.WARNING)
    
    # Validate input files
    required_files = [posweight_file, negweight_file, posweight_ref, negweight_ref]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    try:
        # Process positive weights
        detector.logger.info("Processing positive weights...")
        detector.logger.info("NOTE: Swapping subarrays WITHIN affected file, not with reference values")
        
        posweights = np.loadtxt(posweight_file)  # Affected file
        posweights_ref = np.loadtxt(posweight_ref)  # Reference file (for detection only)
        swapped_pos = detector.detect_and_swap(posweights, posweights_ref, col_blocks, label="POS")
        
        # Process negative weights
        detector.logger.info("Processing negative weights...")
        negweights = np.loadtxt(negweight_file)  # Affected file
        negweights_ref = np.loadtxt(negweight_ref)  # Reference file (for detection only)
        swapped_neg = detector.detect_and_swap(negweights, negweights_ref, col_blocks, label="NEG")
        
        # Save results
        np.savetxt(posweight_out, swapped_pos.flatten(), fmt='%.6f')
        np.savetxt(negweight_out, swapped_neg.flatten(), fmt='%.6f')
        
        # Save swap history
        detector.save_swap_history()
        
        detector.logger.info(f"Successfully saved swapped weights to {posweight_out} and {negweight_out}")
        detector.logger.info(f"Total swaps performed: {len(detector.swap_history)}")
        detector.logger.info("IMPORTANT: All swaps were performed WITHIN the affected files, preserving actual weight values")
        
        return {
            'pos_weights': swapped_pos,
            'neg_weights': swapped_neg,
            'swap_history': detector.swap_history
        }
        
    except Exception as e:
        detector.logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage with custom configuration
    config = SwapConfig(
        error_threshold=0.05,
        max_swaps_per_run=2,
        swap_strategy="best_worst",
        enable_adaptive_threshold=True,
        save_intermediate=True,
        swap_within_affected_file=True  # Ensure swapping within affected file
    )
    
    detect_and_swap_error_subarray_with_reference(config=config)
