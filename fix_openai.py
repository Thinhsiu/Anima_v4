#!/usr/bin/env python
"""
Quick fix for OpenAI API version compatibility issues.
This script downgrades the OpenAI package to version 0.28 which works with the current code.
"""

import sys
import subprocess
import os

def main():
    print("Fixing OpenAI version compatibility issue...")
    
    try:
        # Check current version
        import openai
        current_version = openai.__version__
        print(f"Current OpenAI version: {current_version}")
        
        if current_version.startswith("0.28"):
            print("OpenAI version is already compatible. No changes needed.")
            return
        
        print("Installing compatible OpenAI version (0.28)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai==0.28"])
        print("OpenAI version successfully downgraded to 0.28.")
        print("Please restart Anima to apply the changes.")
        
    except Exception as e:
        print(f"Error fixing OpenAI version: {e}")
        print("You may need to manually run: pip install openai==0.28")

if __name__ == "__main__":
    main()
