#!/usr/bin/env python3
"""Quick start script for the explainable GNN project."""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(description="Quick start for explainable GNN project")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--demo-only", action="store_true", help="Only run the demo")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 Explainable GNN Models - Quick Start")
    print("=" * 80)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not args.skip_install:
        if not run_command("pip install -r requirements.txt", "Installing dependencies"):
            print("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Run tests
    if not args.skip_tests:
        if not run_command("python -m pytest tests/ -v", "Running tests"):
            print("⚠️  Some tests failed, but continuing...")
    
    # Run example
    if not args.demo_only:
        if not run_command("python example.py", "Running example script"):
            print("⚠️  Example script failed, but continuing...")
    
    # Launch demo
    print("\n" + "=" * 80)
    print("🎯 Launching Interactive Demo")
    print("=" * 80)
    print("The Streamlit demo will open in your browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the demo when you're done.")
    
    try:
        subprocess.run(["streamlit", "run", "demo/app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for trying the explainable GNN toolkit!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch demo: {e}")
        print("\nYou can try running it manually:")
        print("  streamlit run demo/app.py")
    
    print("\n" + "=" * 80)
    print("🎉 Quick start completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. 📚 Read the README.md for detailed documentation")
    print("2. 🔧 Explore the configs/ directory for different model configurations")
    print("3. 🧪 Try different datasets: python scripts/train.py --dataset citeseer")
    print("4. 🧠 Experiment with different models: python scripts/train.py --model gat")
    print("5. 🔍 Generate explanations: python -m src.cli explain")
    print("6. 📊 Check out the notebooks/ directory for tutorials")
    print("\nHappy exploring! 🚀")


if __name__ == "__main__":
    main()
