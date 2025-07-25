#!/usr/bin/env python3
"""
Setup script for local LLM using Ollama.

This script helps you set up Ollama for use with the SFCExtraction project.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description, check=True):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def check_ollama_installed():
    """Check if Ollama is installed."""
    return run_command("ollama --version", "Checking if Ollama is installed", check=False)

def check_ollama_running():
    """Check if Ollama service is running."""
    return run_command("ollama list", "Checking if Ollama service is running", check=False)

def install_ollama_instructions():
    """Provide installation instructions for different operating systems."""
    print("\n🔧 Ollama Installation Instructions:")
    print("\n📋 Option 1 - Direct Download (Recommended):")
    print("   Visit: https://ollama.ai/download")
    print("   Download and install for your operating system")
    
    print("\n📋 Option 2 - Command Line:")
    print("   Linux/macOS: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   Windows: winget install Ollama.Ollama")
    
    print("\n📋 Option 3 - Package Managers:")
    print("   macOS: brew install ollama")
    print("   Arch Linux: pacman -S ollama")

def download_recommended_models():
    """Download recommended Ollama models for the project."""
    models = [
        ("llama3.2:3b", "Best balance of speed and quality (3B parameters)"),
        ("llama3.2:1b", "Fastest option for limited resources (1B parameters)"),
    ]
    
    print("\n📦 Downloading recommended models...")
    
    for model, description in models:
        print(f"\n🔄 Downloading {model} - {description}")
        success = run_command(f"ollama pull {model}", f"Pulling {model}", check=False)
        
        if success:
            print(f"✅ {model} downloaded successfully")
        else:
            print(f"❌ Failed to download {model}")
            print("   This might be due to:")
            print("   - Ollama service not running")
            print("   - Network connectivity issues")
            print("   - Insufficient disk space")

def test_ollama_models():
    """Test available Ollama models."""
    print("\n🧪 Testing available models...")
    
    # Get list of installed models
    result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Could not list Ollama models")
        return False
    
    print("📋 Installed models:")
    print(result.stdout)
    
    # Test a simple query with the recommended model
    test_models = ["llama3.2:3b", "llama3.2:1b", "llama3.1:8b"]
    
    for model in test_models:
        if model in result.stdout:
            print(f"\n🧪 Testing {model}...")
            test_success = run_command(
                f'ollama run {model} "Hello, respond with just: Working correctly"',
                f"Testing {model}",
                check=False
            )
            if test_success:
                print(f"✅ {model} is working correctly")
                return True
            else:
                print(f"❌ {model} test failed")
    
    return False

def setup_project_config():
    """Set up project configuration for local LLM."""
    config_path = Path("config.py")
    
    if config_path.exists():
        print("\n📝 config.py already exists")
        
        # Read current config
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Add LLM configuration if not present
        if "# LLM Configuration" not in content:
            llm_config = """
# LLM Configuration
# Leave OPENAI_API_KEY empty to use local Ollama
# OPENAI_API_KEY = 'your_openai_api_key_here'

# Ollama Configuration (used when OPENAI_API_KEY is not set)
OLLAMA_MODEL = 'llama3.2:3b'  # Can be: llama3.2:3b, llama3.2:1b, llama3.1:8b
OLLAMA_BASE_URL = 'http://localhost:11434'  # Default Ollama server
"""
            
            with open(config_path, 'a') as f:
                f.write(llm_config)
            
            print("✅ Added LLM configuration to config.py")
        else:
            print("✅ LLM configuration already present in config.py")
    else:
        print("⚠️ config.py not found. Run: cp config.example.py config.py")

def main():
    """Main setup function."""
    print("🚀 SFCExtraction Local LLM Setup")
    print("=" * 50)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("❌ Ollama is not installed")
        install_ollama_instructions()
        print("\n⏹️ Please install Ollama first, then run this script again")
        return False
    
    print("✅ Ollama is installed")
    
    # Check if Ollama service is running
    if not check_ollama_running():
        print("❌ Ollama service is not running")
        print("\n🔄 Starting Ollama service...")
        print("   Run in a separate terminal: ollama serve")
        print("   Or on macOS/Linux: brew services start ollama")
        print("   Then run this script again")
        return False
    
    print("✅ Ollama service is running")
    
    # Download models
    download_recommended_models()
    
    # Test models
    if test_ollama_models():
        print("\n✅ Local LLM setup completed successfully!")
    else:
        print("\n⚠️ Setup completed but model testing failed")
        print("   You may need to download models manually:")
        print("   ollama pull llama3.2:3b")
    
    # Setup project config
    setup_project_config()
    
    print("\n🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Test the setup: python -c \"from utils.segment_analysis import initialize_llm; initialize_llm()\"")
    print("2. Run analysis: python example_workflow.py --highlights 2518537772 3")
    print("\n💡 Tips:")
    print("- First run will be slower as models load into memory")
    print("- llama3.2:3b offers best quality, llama3.2:1b is fastest")
    print("- You can switch models by editing config.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1) 