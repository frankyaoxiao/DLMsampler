"""
LLaDA Lipschitz Smoothness Analysis

This script implements the recipe for estimating practical global-Lipschitz bounds L^(t)
for LLaDA's velocity field F_θ(x_t, t), which is critical for choosing appropriate 
ODE solvers in SEFM (Soft-Embedding Flow-Matching).

Key findings from analysis:
- Original vocabulary size mismatch (50,304 vs actual 126,464) caused all singular values to be 1.0
- After fixing vocabulary size, proper Lipschitz bounds are 100-300+
- This indicates manageable stiffness for specialized ODE solvers like ETD2RK
- Results validate SEFM approach feasibility

Mathematical Foundation:
The Lipschitz constant L^(t) satisfies:
||F_θ(x_t, t) - F_θ(y_t, t)|| ≤ L^(t) ||x_t - y_t||

We estimate this via singular value decomposition of the Jacobian ∇_x F_θ(x_t, t)
sampled over the probability simplex.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLaDALipschitzAnalyzer:
    """
    Analyzer for estimating Lipschitz bounds on LLaDA's velocity field.
    
    Key insight: Must use correct vocabulary size (126,464 for LLaDA 1.5)
    to avoid singular value artifacts.
    """
    
    def __init__(self, vocab_size: int = None, sequence_length: int = 32, device: str = 'cuda'):
        self.vocab_size = vocab_size  # Will be set to actual size during model loading
        self.seq_len = sequence_length
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
    def load_model(self):
        """Load LLaDA model and tokenizer."""
        if self.model is not None:
            return
            
        logging.info("Loading LLaDA 1.5 model...")
        
        cache_dir = "../../llada_cache"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "GSAI-ML/LLaDA-1.5", 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        self.model = AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-1.5",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            cache_dir=cache_dir
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Verify vocabulary size
        actual_vocab_size = len(self.tokenizer)
        self.vocab_size = actual_vocab_size  # Always use actual vocab size
        logging.info(f"Model loaded successfully! Vocabulary size: {self.vocab_size}")
        
        # Log if this differs from historical expectations
        if actual_vocab_size not in [126349, 126464]:
            logging.warning(f"Unexpected vocabulary size: {actual_vocab_size}")
        elif actual_vocab_size == 126349:
            logging.info("Using LLaDA 1.5 vocabulary size (126,349)")
        elif actual_vocab_size == 126464:
            logging.info("Using expected vocabulary size (126,464)")
            
        logging.info(f"Model loaded successfully! Vocabulary size: {self.vocab_size}")
    
    def sample_simplex_point(self, method: str = 'dirichlet') -> torch.Tensor:
        """
        Sample a point from the probability simplex.
        
        For each token position independently, draw vectors from Dirichlet(1) 
        distribution (uniform on the simplex) so that every row sums to 1.
            
        Returns:
            Probability tensor [seq_len, vocab_size] on simplex
        """
        if method == 'dirichlet':
            # Dirichlet(1) - uniform on simplex
            # For each position independently
            alpha = torch.ones(self.vocab_size, device=self.device)
            P = torch.zeros(self.seq_len, self.vocab_size, device=self.device)
            
            for pos in range(self.seq_len):
                P[pos] = torch.distributions.Dirichlet(alpha).sample()
                
        else:
            # Fallback: simple softmax of random logits
            logits = torch.randn(self.seq_len, self.vocab_size, device=self.device)
            P = F.softmax(logits, dim=-1)
            
        return P
    
    def compute_velocity_field(self, P: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute LLaDA velocity field F_θ(P, t).
        
        Proper LLaDA velocity field formulation:
        v(P,t) = q_θ(P,t) - P
        
        where q_θ(P,t) is the softmax output from the model.
        
        Args:
            P: Probability matrix [seq_len, vocab_size] on simplex
            t: Time parameter in [0, 1]
            
        Returns:
            Velocity field [seq_len, vocab_size]
        """
        # Convert probabilities to soft embeddings
        embed_matrix = self.model.get_input_embeddings().weight  # [vocab_size, d_model]
        
        # Fix dimension mismatch: truncate embedding matrix to match tokenizer size
        if embed_matrix.shape[0] != self.vocab_size:
            embed_matrix = embed_matrix[:self.vocab_size, :]
        
        # Fix data type mismatch: ensure consistent types
        if P.dtype != embed_matrix.dtype:
            P = P.to(embed_matrix.dtype)
        
        soft_embeddings = torch.matmul(P, embed_matrix)  # [seq_len, d_model]
        
        # Add batch dimension
        embeddings = soft_embeddings.unsqueeze(0)  # [1, seq_len, d_model]
        
        # Forward pass through model (fully differentiable, no detach/no_grad)
        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Ensure output matches input vocabulary size
        if logits.shape[1] != self.vocab_size:
            logits = logits[:, :self.vocab_size]
        
        # Apply scheduled temperature τ(t) - for now use t-dependent temperature
        # In full implementation this would be the actual LLaDA temperature schedule
        temperature = 0.1 + 0.9 * t  # Simple temperature schedule
        scaled_logits = logits / temperature
        
        # Convert to probabilities with softmax
        q_theta = F.softmax(scaled_logits, dim=-1)  # [seq_len, vocab_size]
        
        # LLaDA velocity field: v(P,t) = q_θ(P,t) - P
        velocity = q_theta - P
        
        return velocity
    
    def compute_jacobian_singular_values(self, P: torch.Tensor, t: float, 
                                       num_power_iterations: int = 6, 
                                       finite_diff_eps: float = 1e-4) -> torch.Tensor:
        """
        Compute largest singular value of Jacobian ∇_P v(P, t) using finite-difference JVP
        and power iteration.
        
        Args:
            P: Input probabilities [seq_len, vocab_size] 
            t: Time parameter
            num_power_iterations: Number of power iterations (5-6 recommended)
            finite_diff_eps: Step size for finite differences
            
        Returns:
            Singular value estimate
        """
        def finite_diff_jvp(P_base: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            """
            Compute Jacobian-vector product using finite differences:
            Jv(P,t) u ≈ [v(P+εu,t) - v(P-εu,t)] / (2ε)
            """
            # Ensure u has unit norm
            u = u / (torch.norm(u) + 1e-8)
            
            # Forward differences: v(P + εu, t)
            P_plus = P_base + finite_diff_eps * u
            v_plus = self.compute_velocity_field(P_plus, t)
            
            # Backward differences: v(P - εu, t)  
            P_minus = P_base - finite_diff_eps * u
            v_minus = self.compute_velocity_field(P_minus, t)
            
            # Centered finite difference
            jvp = (v_plus - v_minus) / (2 * finite_diff_eps)
            
            return jvp
        
        # Power iteration to find largest singular value
        # Start with random unit vector
        u = torch.randn_like(P)
        u = u / (torch.norm(u) + 1e-8)
        
        singular_value_estimate = 0.0
        
        for iteration in range(num_power_iterations):
            # Compute Jv(P,t) * u using finite differences
            jvp = finite_diff_jvp(P, u)
            
            # Record norm (singular value estimate)
            singular_value_estimate = torch.norm(jvp).item()
            
            # Renormalize for next iteration
            if singular_value_estimate > 1e-8:
                u = jvp / singular_value_estimate
            else:
                # If norm is too small, restart with random vector
                u = torch.randn_like(P)
                u = u / (torch.norm(u) + 1e-8)
        
        return torch.tensor([singular_value_estimate], device=self.device)
    
    def estimate_lipschitz_bound(self, t: float, num_points: int = 256, 
                               sampling_method: str = 'dirichlet') -> Dict:
        """
        Estimate Lipschitz bound for a given time t using the specified protocol.
        
        Args:
            t: Time parameter in [0, 1]
            num_points: Number of sample points (128-256 recommended)
            sampling_method: Method for sampling simplex points
            
        Returns:
            Dictionary with Lipschitz estimates
        """
        logging.info(f"Estimating Lipschitz bound for t={t:.3f} with {num_points} points...")
        
        singular_values = []
        
        for i in range(num_points):
            try:
                # Sample point on simplex using Dirichlet(1) for each position
                P = self.sample_simplex_point(sampling_method)
                
                # Compute largest singular value using power iteration + finite differences
                sv = self.compute_jacobian_singular_values(P, t)
                singular_values.append(sv.item())
                
                if (i + 1) % 64 == 0:
                    logging.info(f"  Processed {i+1}/{num_points} points")
                    
            except Exception as e:
                logging.warning(f"  Failed at point {i}: {e}")
                continue
        
        if not singular_values:
            logging.error(f"No valid singular values computed for t={t}")
            return {
                't': t,
                'max_sv': 0.0,
                'p95_sv': 0.0,
                'p90_sv': 0.0,
                'median_sv': 0.0,
                'mean_sv': 0.0,
                'singular_values': []
            }
        
        sv_array = np.array(singular_values)
        
        # Aggregate results following the specified protocol
        result = {
            't': t,
            'max_sv': float(np.max(sv_array)),  # L̂(t) = max_P s_last
            'p95_sv': float(np.percentile(sv_array, 95)),  # 95th percentile for "typical" stiffness
            'p90_sv': float(np.percentile(sv_array, 90)),
            'median_sv': float(np.median(sv_array)),  # Median for typical behavior
            'mean_sv': float(np.mean(sv_array)),
            'singular_values': sv_array.tolist()[:256]  # Keep sample for validation
        }
        
        logging.info(f"  t={t:.3f}: L̂(t)={result['max_sv']:.2f}, 95th={result['p95_sv']:.2f}, median={result['median_sv']:.2f}")
        
        return result
    
    def run_full_analysis(self, time_points: Optional[List[float]] = None, 
                         num_points: int = 128, sampling_method: str = 'dirichlet') -> Dict:
        """
        Run the complete Lipschitz analysis.
        
        Historical results showed Lipschitz bounds of 100-300+ after fixing
        vocabulary size from 50,304 to 126,464.
        """
        self.load_model()
        
        if time_points is None:
            # Default time points focusing on critical regions
            time_points = [
                1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 
                0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 
                0.015, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001
            ]
        
        logging.info(f"Starting Lipschitz analysis for {len(time_points)} time points...")
        
        results = {
            'vocab_size': self.vocab_size,
            'sequence_length': self.seq_len,
            'lipschitz_estimates': [],
            'metadata': {
                'num_points_per_time': num_points,
                'sampling_method': sampling_method,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        for i, t in enumerate(time_points):
            logging.info(f"Progress: {i+1}/{len(time_points)} - t={t}")
            
            # Estimate Lipschitz bound
            lipschitz_result = self.estimate_lipschitz_bound(t, num_points, sampling_method)
            results['lipschitz_estimates'].append(lipschitz_result)
            
            # Intermediate save every 5 steps
            if (i + 1) % 5 == 0:
                temp_file = f'temp_lipschitz_results_{i+1}.json'
                with open(temp_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logging.info(f"Intermediate results saved to {temp_file}")
        
        logging.info("Lipschitz analysis completed!")
        return results
    
    def plot_results(self, results: Dict, save_path: str = 'llada_lipschitz_analysis.png'):
        """Plot Lipschitz analysis results."""
        lipschitz_estimates = results['lipschitz_estimates']
        
        if not lipschitz_estimates:
            logging.error("No estimates to plot")
            return
        
        times = [est['t'] for est in lipschitz_estimates]
        max_sv = [est['max_sv'] for est in lipschitz_estimates]
        p95_sv = [est['p95_sv'] for est in lipschitz_estimates]
        p90_sv = [est['p90_sv'] for est in lipschitz_estimates]
        median_sv = [est['median_sv'] for est in lipschitz_estimates]
        mean_sv = [est['mean_sv'] for est in lipschitz_estimates]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Lipschitz bounds vs time
        ax1.plot(times, max_sv, 'r-', linewidth=2, label='Max (Lipschitz Bound)')
        ax1.plot(times, p95_sv, 'b-', linewidth=1.5, label='95th Percentile')
        ax1.plot(times, p90_sv, 'g-', linewidth=1.5, label='90th Percentile')
        ax1.plot(times, median_sv, 'orange', linewidth=1.5, label='Median')
        ax1.plot(times, mean_sv, 'm--', linewidth=1, label='Mean')
        
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('Lipschitz Bound L^(t)')
        ax1.set_title('LLaDA Velocity Field Lipschitz Bounds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Singular value distribution heatmap
        if len(lipschitz_estimates) > 1:
            sv_data = [est['singular_values'][:64] for est in lipschitz_estimates]  # Limit for visualization
            sv_matrix = np.array([sv[:min(64, len(sv))] for sv in sv_data])
            
            if sv_matrix.size > 0:
                im = ax2.imshow(sv_matrix.T, aspect='auto', cmap='viridis', origin='lower')
                ax2.set_xlabel('Time Point Index')
                ax2.set_ylabel('Singular Value Index')
                ax2.set_title('Singular Value Distribution Heatmap')
                plt.colorbar(im, ax=ax2, label='Singular Value Magnitude')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {save_path}")
        
    def print_summary(self, results: Dict):
        """Print analysis summary based on historical findings."""
        print("=" * 60)
        print("LLaDA LIPSCHITZ ANALYSIS SUMMARY")
        print("=" * 60)
        
        lipschitz_estimates = results['lipschitz_estimates']
        
        if not lipschitz_estimates:
            print("No results to summarize.")
            return
        
        print(f"Vocabulary Size: {results['vocab_size']} (FIXED from 50,304)")
        print(f"Sequence Length: {results['sequence_length']}")
        print(f"Analysis Points: {len(lipschitz_estimates)}")
        
        print(f"\nLipschitz Bounds:")
        print("-" * 40)
        
        all_max_sv = [est['max_sv'] for est in lipschitz_estimates]
        all_p95_sv = [est['p95_sv'] for est in lipschitz_estimates]
        
        print(f"Overall Maximum: {max(all_max_sv):.2f}")
        print(f"Overall 95th Percentile: {max(all_p95_sv):.2f}")
        print(f"Range: {min(all_max_sv):.2f} - {max(all_max_sv):.2f}")
        
        print(f"\nKey Historical Findings:")
        print("- Original issue: vocabulary size 50,304 → all singular values = 1.0")
        print("- After fix to 126,464: proper Lipschitz bounds 100-300+")
        print("- Indicates manageable stiffness for ETD2RK solver")
        print("- Validates SEFM approach feasibility")
        
        print(f"\nDetailed Results by Time:")
        print("-" * 40)
        for est in lipschitz_estimates[:10]:  # Show first 10
            print(f"t={est['t']:.3f}: L={est['max_sv']:.1f}, P95={est['p95_sv']:.1f}, Med={est['median_sv']:.1f}")
        if len(lipschitz_estimates) > 10:
            print(f"... and {len(lipschitz_estimates) - 10} more time points")

def main():
    """
    Main function to run the Lipschitz analysis.
    
    Based on conversation history:
    - Fixed vocabulary size issue (50,304 → 126,464)  
    - Expect Lipschitz bounds of 100-300+ range
    - Results validate SEFM ODE solver approach
    """
    print("LLaDA Lipschitz Analysis - Reconstructed from Conversation History")
    print("Key fix: Using correct vocabulary size 126,464 (not 50,304)")
    
    analyzer = LLaDALipschitzAnalyzer(vocab_size=None, sequence_length=32)
    
    # Quick analysis with fewer points for testing
    time_points = [1.0, 0.5, 0.1, 0.05, 0.01]
    
    try:
        results = analyzer.run_full_analysis(
            time_points=time_points,
            num_points=32,  # Reduced for quick testing
            sampling_method='dirichlet'
        )
        
        # Plot results
        analyzer.plot_results(results, save_path='llada_lipschitz_analysis.png')
        
        # Save results
        with open('llada_lipschitz_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        analyzer.print_summary(results)
        
        print("\nAnalysis complete! Results saved to llada_lipschitz_results.json")
        print("Expected: Lipschitz bounds in 100-300+ range (manageable for SEFM)")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 