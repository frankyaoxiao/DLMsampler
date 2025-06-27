#!/usr/bin/env python3
"""
Test script for SEFM (Soft-Embedding Flow-Matching) Sampler

This script demonstrates how to use the SEFM sampler for text generation.
SEFM replaces the entire masked diffusion process with continuous ODE integration,
targeting single-digit NFEs (Number of Function Evaluations).
"""

import time
import torch
from sefm_sampler import sefm_generate_text

def test_sefm_basic():
    """Basic SEFM test with simple prompts."""
    print("🚀 Testing SEFM Sampler")
    print("=" * 50)
    
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where robots and humans coexist,",
        "The most important discovery in science was",
        "Write a short story about space exploration:"
    ]
    
    # SEFM parameters to test
    test_configs = [
        {"steps": 6, "gen_length": 64, "name": "Fast (6 steps)"},
        {"steps": 12, "gen_length": 64, "name": "Balanced (12 steps)"},
        {"steps": 4, "gen_length": 32, "name": "Ultra-fast (4 steps)"},
    ]
    
    for config in test_configs:
        print(f"\n📊 Configuration: {config['name']}")
        print(f"   Steps: {config['steps']}, Length: {config['gen_length']}")
        print("-" * 40)
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n🔤 Prompt {i+1}: {prompt}")
            
            try:
                start_time = time.time()
                
                # Generate text using SEFM
                generated_text = sefm_generate_text(
                    prompt=prompt,
                    gen_length=config['gen_length'],
                    steps=config['steps'],
                    k=8,           # Top-k for soft embeddings
                    H_freeze=0.05,  # Entropy threshold for freezing
                    tol=1e-4,      # Error tolerance
                    tau_min=0.1,   # Minimum temperature
                    tau_0=1.0,     # Initial temperature
                    gamma=1.0      # Temperature schedule exponent
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                print(f"⚡ Generated ({generation_time:.2f}s): {generated_text}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 50)

def test_sefm_comparison():
    """Test SEFM with different step counts to compare efficiency vs quality."""
    print("\n🔬 SEFM Step Count Comparison")
    print("=" * 50)
    
    prompt = "The key to understanding quantum mechanics is"
    step_counts = [4, 6, 8, 12, 16]
    
    print(f"Prompt: {prompt}")
    print("-" * 40)
    
    for steps in step_counts:
        try:
            start_time = time.time()
            
            generated_text = sefm_generate_text(
                prompt=prompt,
                gen_length=50,
                steps=steps
            )
            
            end_time = time.time()
            
            print(f"Steps: {steps:2d} | Time: {end_time - start_time:.2f}s | Text: {generated_text}")
            
        except Exception as e:
            print(f"Steps: {steps:2d} | Error: {e}")

def test_sefm_parameters():
    """Test different SEFM parameters to understand their effects."""
    print("\n⚙️  SEFM Parameter Testing")
    print("=" * 50)
    
    prompt = "Artificial intelligence will transform society by"
    
    # Test different parameter combinations
    param_tests = [
        {"name": "Default", "params": {"steps": 6}},
        {"name": "High Precision", "params": {"tol": 1e-5, "steps": 8}},
        {"name": "Fast Freezing", "params": {"H_freeze": 0.1, "steps": 6}},
        {"name": "Low Temperature", "params": {"tau_0": 0.5, "tau_min": 0.05, "steps": 6}},
        {"name": "Conservative", "params": {"k": 4, "H_freeze": 0.02, "steps": 10}},
    ]
    
    for test in param_tests:
        print(f"\n🧪 Test: {test['name']}")
        print(f"   Parameters: {test['params']}")
        
        try:
            start_time = time.time()
            
            generated_text = sefm_generate_text(
                prompt=prompt,
                gen_length=60,
                **test['params']
            )
            
            end_time = time.time()
            
            print(f"   Time: {end_time - start_time:.2f}s")
            print(f"   Result: {generated_text}")
            
        except Exception as e:
            print(f"   Error: {e}")

def main():
    """Main test function."""
    print("🎯 SEFM (Soft-Embedding Flow-Matching) Sampler Test")
    print("   Replacing ~30 diffusion steps with single-digit NFEs")
    print("   Using ETD2RK integration on probability simplex")
    print()
    
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available, using CPU")
        
        # Run tests
        test_sefm_basic()
        test_sefm_comparison()
        test_sefm_parameters()
        
        print("\n🎉 SEFM testing completed!")
        print("   Check the results above to evaluate:")
        print("   - Text quality vs standard diffusion")
        print("   - Speed improvement (target: significant speedup)")
        print("   - Convergence with fewer steps")
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 