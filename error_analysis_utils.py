import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import logging
from datetime import datetime

class ErrorAnalysisUtils:
    """Utility class for analyzing error detection and correction results"""
    
    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level):
        """Setup logging for analysis utilities"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_error_distribution(self, weights: np.ndarray, reference: np.ndarray, 
                                 col_blocks: List[int], save_plots: bool = True) -> Dict[str, Any]:
        """Analyze the distribution of errors across blocks"""
        
        num_cols, num_rows = weights.shape
        analysis_results = {
            'block_analysis': [],
            'overall_stats': {},
            'error_histogram': None,
            'block_error_plot': None
        }
        
        # Analyze each block
        start = 0
        block_errors = []
        block_sizes = []
        
        for i, cb in enumerate(col_blocks):
            block = weights[start:start+cb, :]
            ref_block = reference[start:start+cb, :]
            
            errors = np.abs(block - ref_block)
            
            block_stats = {
                'block_id': i + 1,
                'size': cb,
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(errors),
                'min_error': np.min(errors),
                'median_error': np.median(errors),
                'error_percentiles': {
                    '25': np.percentile(errors, 25),
                    '50': np.percentile(errors, 50),
                    '75': np.percentile(errors, 75),
                    '95': np.percentile(errors, 95),
                    '99': np.percentile(errors, 99)
                },
                'start_col': start,
                'end_col': start + cb
            }
            
            analysis_results['block_analysis'].append(block_stats)
            block_errors.append(np.mean(errors))
            block_sizes.append(cb)
            start += cb
        
        # Overall statistics
        all_errors = np.abs(weights - reference).flatten()
        analysis_results['overall_stats'] = {
            'total_mean_error': np.mean(all_errors),
            'total_std_error': np.std(all_errors),
            'total_max_error': np.max(all_errors),
            'total_min_error': np.min(all_errors),
            'error_skewness': self._calculate_skewness(all_errors),
            'error_kurtosis': self._calculate_kurtosis(all_errors),
            'worst_block': np.argmax(block_errors),
            'best_block': np.argmin(block_errors),
            'error_range': np.max(block_errors) - np.min(block_errors)
        }
        
        # Create visualizations
        if save_plots:
            self._create_error_visualizations(all_errors, block_errors, col_blocks, analysis_results)
        
        return analysis_results
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of error distribution"""
        mean = np.mean(data)
        std = np.std(data)
        n = len(data)
        skewness = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of error distribution"""
        mean = np.mean(data)
        std = np.std(data)
        n = len(data)
        kurtosis = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n-1)**2 / ((n-2) * (n-3)))
        return kurtosis
    
    def _create_error_visualizations(self, all_errors: np.ndarray, block_errors: List[float], 
                                   col_blocks: List[int], analysis_results: Dict[str, Any]):
        """Create comprehensive error visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Analysis Visualization (Within-File Swapping)', fontsize=16, fontweight='bold')
        
        # 1. Error histogram
        axes[0, 0].hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Error Distribution Histogram')
        axes[0, 0].set_xlabel('Absolute Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics to plot
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        axes[0, 0].axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.4f}')
        axes[0, 0].axvline(mean_error + std_error, color='orange', linestyle='--', label=f'Mean+Std: {mean_error + std_error:.4f}')
        axes[0, 0].legend()
        
        # 2. Block error comparison
        block_ids = [f'Block {i+1}' for i in range(len(col_blocks))]
        bars = axes[0, 1].bar(block_ids, block_errors, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Mean Error by Block')
        axes[0, 1].set_xlabel('Block ID')
        axes[0, 1].set_ylabel('Mean Absolute Error')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Highlight worst and best blocks
        worst_idx = np.argmax(block_errors)
        best_idx = np.argmin(block_errors)
        bars[worst_idx].set_color('red')
        bars[best_idx].set_color('green')
        
        # 3. Error vs Block Size
        block_sizes = col_blocks
        axes[1, 0].scatter(block_sizes, block_errors, s=100, alpha=0.7, c='purple')
        axes[1, 0].set_title('Error vs Block Size')
        axes[1, 0].set_xlabel('Block Size (columns)')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(block_sizes, block_errors, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(block_sizes, p(block_sizes), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        axes[1, 0].legend()
        
        # 4. Box plot of errors by block
        error_by_block = []
        block_labels = []
        start = 0
        for i, cb in enumerate(col_blocks):
            block = weights[start:start+cb, :]
            ref_block = reference[start:start+cb, :]
            errors = np.abs(block - ref_block).flatten()
            error_by_block.append(errors)
            block_labels.append(f'Block {i+1}')
            start += cb
        
        axes[1, 1].boxplot(error_by_block, labels=block_labels)
        axes[1, 1].set_title('Error Distribution by Block')
        axes[1, 1].set_xlabel('Block ID')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Error analysis plots saved to {filename}")
    
    def generate_error_report(self, analysis_results: Dict[str, Any], 
                            swap_history: List[Dict], output_file: str = None) -> str:
        """Generate a comprehensive error analysis report"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"error_report_{timestamp}.txt"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ERROR DETECTION AND CORRECTION ANALYSIS REPORT")
        report_lines.append("SWAPPING STRATEGY: WITHIN AFFECTED FILE")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall statistics
        overall_stats = analysis_results['overall_stats']
        report_lines.append("OVERALL ERROR STATISTICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Mean Error: {overall_stats['total_mean_error']:.6f}")
        report_lines.append(f"Total Std Error: {overall_stats['total_std_error']:.6f}")
        report_lines.append(f"Total Max Error: {overall_stats['total_max_error']:.6f}")
        report_lines.append(f"Total Min Error: {overall_stats['total_min_error']:.6f}")
        report_lines.append(f"Error Skewness: {overall_stats['error_skewness']:.6f}")
        report_lines.append(f"Error Kurtosis: {overall_stats['error_kurtosis']:.6f}")
        report_lines.append(f"Error Range: {overall_stats['error_range']:.6f}")
        report_lines.append(f"Worst Block: {overall_stats['worst_block'] + 1}")
        report_lines.append(f"Best Block: {overall_stats['best_block'] + 1}")
        report_lines.append("")
        
        # Block-by-block analysis
        report_lines.append("BLOCK-BY-BLOCK ANALYSIS:")
        report_lines.append("-" * 40)
        for block_analysis in analysis_results['block_analysis']:
            report_lines.append(f"Block {block_analysis['block_id']}:")
            report_lines.append(f"  Size: {block_analysis['size']} columns")
            report_lines.append(f"  Mean Error: {block_analysis['mean_error']:.6f}")
            report_lines.append(f"  Std Error: {block_analysis['std_error']:.6f}")
            report_lines.append(f"  Max Error: {block_analysis['max_error']:.6f}")
            report_lines.append(f"  Median Error: {block_analysis['median_error']:.6f}")
            report_lines.append(f"  95th Percentile: {block_analysis['error_percentiles']['95']:.6f}")
            report_lines.append("")
        
        # Swap history
        if swap_history:
            report_lines.append("SWAP OPERATIONS HISTORY:")
            report_lines.append("-" * 40)
            for i, swap in enumerate(swap_history):
                report_lines.append(f"Swap {i+1}:")
                report_lines.append(f"  Error Block: {swap['error_block']}")
                report_lines.append(f"  Best Block: {swap['best_block']}")
                report_lines.append(f"  Label: {swap['label']}")
                report_lines.append(f"  Timestamp: {swap['timestamp']}")
                report_lines.append(f"  Swap Type: {swap.get('swap_type', 'within_affected_file')}")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        if overall_stats['total_mean_error'] > 0.1:
            report_lines.append("⚠️  HIGH ERROR LEVEL DETECTED")
            report_lines.append("   - Consider lowering error threshold")
            report_lines.append("   - Investigate worst performing blocks")
            report_lines.append("   - Check for systematic errors")
        else:
            report_lines.append("✅ ERROR LEVELS ACCEPTABLE")
            report_lines.append("   - Current error levels are within acceptable range")
        
        if overall_stats['error_skewness'] > 1.0:
            report_lines.append("⚠️  SKEWED ERROR DISTRIBUTION")
            report_lines.append("   - Error distribution is not normal")
            report_lines.append("   - Consider different error metrics")
        
        if len(swap_history) > 5:
            report_lines.append("⚠️  FREQUENT SWAPPING DETECTED")
            report_lines.append("   - Many swap operations performed")
            report_lines.append("   - Consider improving initial weight mapping")
        
        # New section for within-file swapping benefits
        report_lines.append("")
        report_lines.append("WITHIN-FILE SWAPPING BENEFITS:")
        report_lines.append("-" * 40)
        report_lines.append("✅ Preserves actual weight values")
        report_lines.append("✅ Maintains circuit characteristics")
        report_lines.append("✅ Improves accuracy through reorganization")
        report_lines.append("✅ No reference value contamination")
        report_lines.append("✅ Better fault tolerance")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Error analysis report saved to {output_file}")
        return output_file
    
    def compare_before_after(self, original_weights: np.ndarray, corrected_weights: np.ndarray,
                           reference: np.ndarray, col_blocks: List[int]) -> Dict[str, Any]:
        """Compare error levels before and after correction"""
        
        # Calculate errors before correction
        original_errors = np.abs(original_weights - reference)
        original_mae = np.mean(original_errors)
        
        # Calculate errors after correction
        corrected_errors = np.abs(corrected_weights - reference)
        corrected_mae = np.mean(corrected_errors)
        
        # Calculate improvement
        improvement = original_mae - corrected_mae
        improvement_percent = (improvement / original_mae) * 100 if original_mae > 0 else 0
        
        # Block-wise comparison
        block_comparison = []
        start = 0
        for i, cb in enumerate(col_blocks):
            orig_block_errors = original_errors[start:start+cb, :]
            corr_block_errors = corrected_errors[start:start+cb, :]
            
            orig_block_mae = np.mean(orig_block_errors)
            corr_block_mae = np.mean(corr_block_errors)
            block_improvement = orig_block_mae - corr_block_mae
            
            block_comparison.append({
                'block_id': i + 1,
                'original_mae': orig_block_mae,
                'corrected_mae': corr_block_mae,
                'improvement': block_improvement,
                'improvement_percent': (block_improvement / orig_block_mae) * 100 if orig_block_mae > 0 else 0
            })
            start += cb
        
        comparison_results = {
            'overall_improvement': improvement,
            'overall_improvement_percent': improvement_percent,
            'original_mae': original_mae,
            'corrected_mae': corrected_mae,
            'block_comparison': block_comparison
        }
        
        # Create comparison visualization
        self._create_comparison_plot(block_comparison, comparison_results)
        
        return comparison_results
    
    def _create_comparison_plot(self, block_comparison: List[Dict], comparison_results: Dict[str, Any]):
        """Create before/after comparison visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Before vs After Error Correction (Within-File Swapping)', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        block_ids = [comp['block_id'] for comp in block_comparison]
        original_maes = [comp['original_mae'] for comp in block_comparison]
        corrected_maes = [comp['corrected_mae'] for comp in block_comparison]
        improvements = [comp['improvement'] for comp in block_comparison]
        
        # Before/After comparison
        x = np.arange(len(block_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, original_maes, width, label='Before Correction', color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, corrected_maes, width, label='After Correction', color='green', alpha=0.7)
        
        ax1.set_xlabel('Block ID')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Error Comparison by Block')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Block {bid}' for bid in block_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement visualization
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.bar(block_ids, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Block ID')
        ax2.set_ylabel('Error Improvement')
        ax2.set_title('Error Improvement by Block')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add overall improvement text
        overall_imp = comparison_results['overall_improvement_percent']
        ax2.text(0.02, 0.98, f'Overall Improvement: {overall_imp:.2f}%', 
                transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"correction_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison plot saved to {filename}")

def load_swap_history(filename: str) -> List[Dict]:
    """Load swap history from JSON file"""
    with open(filename, 'r') as f:
        history_data = json.load(f)
    return history_data

def export_analysis_to_excel(analysis_results: Dict[str, Any], 
                           comparison_results: Dict[str, Any],
                           swap_history: List[Dict],
                           output_file: str = None) -> str:
    """Export analysis results to Excel file"""
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"error_analysis_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Block analysis
        block_df = pd.DataFrame(analysis_results['block_analysis'])
        block_df.to_excel(writer, sheet_name='Block Analysis', index=False)
        
        # Overall stats
        overall_df = pd.DataFrame([analysis_results['overall_stats']])
        overall_df.to_excel(writer, sheet_name='Overall Statistics', index=False)
        
        # Block comparison
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results['block_comparison'])
            comparison_df.to_excel(writer, sheet_name='Before After Comparison', index=False)
        
        # Swap history
        if swap_history:
            swap_df = pd.DataFrame(swap_history)
            swap_df.to_excel(writer, sheet_name='Swap History', index=False)
    
    return output_file

if __name__ == "__main__":
    # Example usage
    utils = ErrorAnalysisUtils()
    
    # This would be used after running the error detection
    print("Error Analysis Utilities loaded successfully!")
    print("Use this module to analyze error detection results and generate reports.")
    print("NOTE: This system now supports within-file swapping for better accuracy improvement.") 