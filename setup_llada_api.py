#!/usr/bin/env python3
"""
Setup script for LLaDA Model API with Inspect AI

This script installs the package in development mode and registers
the LLaDA model API with Inspect AI.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed!")
        print(f"   Error: {e.stderr.strip()}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("inspect_ai", "Inspect AI")
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is not installed")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Please install them first:")
        for package in missing:
            if package == "inspect_ai":
                print(f"  pip install inspect-ai")
            else:
                print(f"  pip install {package}")
        return False
    
    print("✓ All dependencies are installed")
    return True

def main():
    """Main setup process."""
    print("🚀 Setting up LLaDA Model API for Inspect AI")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ pyproject.toml not found. Please run this script from the project root.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        return 1
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        from llada_inspect.providers import llada, llada_batch
        print("✓ LLaDA providers can be imported")
        
        # Try to register with inspect AI
        from inspect_ai.model import get_model
        try:
            model = get_model("llada/llada-1.5")
            print("✓ LLaDA model API is registered with Inspect AI")
        except Exception as e:
            print(f"⚠ Could not instantiate model (this might be OK if LLaDA model files aren't downloaded): {e}")
            
    except ImportError as e:
        print(f"✗ Cannot import LLaDA providers: {e}")
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Test the installation: python test_llada_api.py")
    print("2. Run GPQA evaluation: inspect eval llada_inspect/gpqa.py --model llada/llada-1.5")
    print("3. For batch processing: inspect eval llada_inspect/gpqa.py --model llada-batch/llada-1.5")
    
    print("\nModel configuration options:")
    print("  --model-config gen_length=256,steps=128,temperature=0.0")
    print("  --model-config batch_size=8,block_length=32")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 