#!/usr/bin/env python
"""
Launcher script for Anima File Sharing UI

This script launches the graphical file sharing interface that allows
users to add, manage, and share files with Anima.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Launch the file sharing UI"""
    try:
        # Import required modules
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing required dependencies...")
        
        # Try to install missing packages
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
            print("Dependencies installed successfully.")
        except Exception as install_error:
            print(f"Error installing dependencies: {install_error}")
            print("Please manually install PIL/Pillow using: pip install pillow")
            input("Press Enter to exit...")
            return 1
    
    print("Starting Anima File Sharing UI...")
    
    try:
        from ui.file_sharing_ui import launch_file_sharing_ui
        launch_file_sharing_ui()
        return 0
    except Exception as e:
        print(f"Error launching file sharing UI: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    sys.exit(main())
