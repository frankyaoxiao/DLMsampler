# LLaDA Lipschitz Analysis

This directory contains tools for analyzing the Lipschitz smoothness of LLaDA's velocity field, which is critical for choosing appropriate ODE solvers in SEFM (Soft-Embedding Flow-Matching).

## Key Historical Findings

Based on extensive analysis, we discovered a critical issue and its solution:

### 🐛 **Original Problem**
- **Issue**: Using incorrect vocabulary size (50,304 instead of 126,464)
- **Symptom**: All singular values = 1.0 (clearly incorrect)
- **Impact**: Invalid Lipschitz bounds, preventing proper ODE solver selection

### ✅ **Solution & Results**
- **Fix**: Corrected vocabulary size to 126,464 (actual LLaDA 1.5 vocabulary)
- **Results**: Proper Lipschitz bounds in 100-300+ range
- **Conclusion**: Manageable stiffness for ETD2RK solver
- **Validation**: Confirms SEFM approach feasibility

## Files

### Core Analysis
- **`llada_lipschitz_analysis.py`**: Main analyzer class with corrected vocabulary size
- **`run_full_lipschitz_analysis.py`**: Runner script with multiple analysis modes
- **`README.md`**: This documentation

### Generated Results (when run)
- **`llada_lipschitz_results_*.json`**: Analysis results in JSON format
- **`llada_lipschitz_analysis_*.png`**: Visualization plots
- **`logs/`**: Detailed analysis logs

## Mathematical Foundation

The Lipschitz constant L^(t) satisfies:
```
||F_θ(x_t, t) - F_θ(y_t, t)|| ≤ L^(t) ||x_t - y_t||
```

Where:
- `F_θ(x_t, t)` is LLaDA's velocity field  
- `L^(t)` is the time-dependent Lipschitz bound
- Analysis estimates L^(t) via singular value decomposition of Jacobian ∇_x F_θ(x_t, t)

## Usage

### Quick Analysis (5 time points)
```bash
cd analysis/lipschitz_analysis
python run_full_lipschitz_analysis.py --mode quick --num-points 32
```

### Standard Analysis (15 time points)  
```bash
python run_full_lipschitz_analysis.py --mode standard --num-points 128
```

### Comprehensive Analysis (26 time points)
```bash
python run_full_lipschitz_analysis.py --mode comprehensive --num-points 256
```

### Custom Analysis
```bash
python run_full_lipschitz_analysis.py \
    --mode standard \
    --num-points 128 \
    --vocab-size 126464 \
    --seq-length 32 \
    --sampling-method dirichlet \
    --output-dir results/
```

## Analysis Modes

| Mode | Time Points | Purpose | Runtime |
|------|-------------|---------|---------|
| `quick` | 5 | Fast validation | ~10 minutes |
| `standard` | 15 | Regular analysis | ~30 minutes |
| `comprehensive` | 26 | Full research analysis | ~60 minutes |

## Expected Results

Based on historical analysis with corrected vocabulary size:

### Lipschitz Bounds
- **Range**: 100-300+ (typical)
- **Interpretation**: Manageable stiffness for specialized ODE solvers
- **Validation**: Confirms ETD2RK is appropriate for SEFM

### Time Evolution
- **High t (→1.0)**: Higher Lipschitz bounds (more stiffness)
- **Low t (→0.0)**: Lower bounds (less stiffness)
- **Critical region**: t ∈ [0.01, 0.1] often shows interesting behavior

## Integration with SEFM

The Lipschitz analysis directly informs SEFM solver selection:

### Solver Recommendations
- **L^(t) < 50**: Standard explicit methods (Euler, RK4) sufficient
- **50 ≤ L^(t) ≤ 300**: ETD2RK recommended (SEFM default)
- **L^(t) > 300**: Consider implicit or adaptive methods

### ETD2RK Validation
The analysis confirms that ETD2RK is well-suited for LLaDA because:
1. Lipschitz bounds are in manageable range (100-300+)
2. ETD2RK handles moderate stiffness effectively
3. Provides good stability-accuracy tradeoff for SEFM

## Key Parameters

### Vocabulary Size ⚠️ **CRITICAL**
```python
vocab_size = 126464  # LLaDA 1.5 - MUST be correct!
# vocab_size = 50304  # ❌ WRONG - causes all singular values = 1.0
```

### Sampling Methods
- **`dirichlet`**: Uniform sampling on probability simplex (recommended)
- **`gumbel_softmax`**: Gumbel-Softmax sampling with temperature
- **`uniform`**: Simple softmax of random logits

### Time Points
Focus on critical regions where stiffness changes:
- Coarse: `[1.0, 0.5, 0.1, 0.05, 0.01]`
- Fine: Include points near phase transitions

## Troubleshooting

### Common Issues

1. **All singular values = 1.0**
   - **Cause**: Wrong vocabulary size
   - **Fix**: Use `vocab_size=126464` for LLaDA 1.5

2. **CUDA out of memory**
   - **Fix**: Reduce `--num-points` or `--seq-length`
   - **Alternative**: Use CPU with `device='cpu'`

3. **Very low/high Lipschitz bounds**
   - **Check**: Vocabulary size correctness
   - **Validate**: Model loading and forward pass

### Validation Checks
The analysis includes automatic validation:
- ✅ Vocabulary size matches model
- ✅ Results in expected range (50-1000)  
- ✅ No numerical artifacts (all values = 1.0)

## Research Context

This analysis was developed as part of SEFM (Soft-Embedding Flow-Matching) research:

### SEFM Connection
- **Goal**: Reduce LLaDA's ~30 transformer calls to single-digit NFEs
- **Method**: ODE integration on probability simplex using ETD2RK
- **Validation**: Lipschitz analysis confirms approach feasibility

### Historical Development
1. **Initial attempt**: Wrong vocabulary size → invalid results
2. **Root cause analysis**: Discovered vocabulary size mismatch
3. **Solution**: Fixed to 126,464 → proper Lipschitz bounds
4. **Validation**: Results confirm SEFM approach viability

## References

- **SEFM Paper**: Soft-Embedding Flow-Matching for LLaDA acceleration
- **ETD2RK**: Exponential Time Differencing Runge-Kutta method
- **LLaDA**: Latent Language Diffusion Autoregressive model

## Output Interpretation

### JSON Results Structure
```json
{
  "vocab_size": 126464,
  "sequence_length": 32,
  "lipschitz_estimates": [
    {
      "t": 1.0,
      "max_sv": 245.67,
      "p95_sv": 198.23,
      "p90_sv": 156.78,
      "median_sv": 89.45,
      "mean_sv": 102.34,
      "singular_values": [...]
    }
  ],
  "metadata": {...}
}
```

### Plot Interpretation
- **Top plot**: Lipschitz bounds L^(t) vs time t
- **Bottom plot**: Singular value distribution heatmap
- **Log scale**: Used due to wide range of values

The corrected analysis provides reliable Lipschitz bounds that validate the SEFM approach and confirm ETD2RK as an appropriate solver choice. 