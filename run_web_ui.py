#!/usr/bin/env python
"""
Launcher for the Anima Voice Emotion Web UI
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def main():
    """Run the web UI application"""
    print("Starting Anima Voice Emotion Web UI...")
    
    # Get the project root directory
    project_dir = Path(__file__).parent
    web_dir = project_dir / "web"
    
    # Make sure we're in the virtual environment
    if not os.environ.get("VIRTUAL_ENV"):
        print("Activating virtual environment...")
        if sys.platform == "win32":
            venv_path = project_dir / "venv" / "Scripts" / "python.exe"
            if venv_path.exists():
                subprocess.call([str(venv_path), __file__])
                return
            else:
                print("Virtual environment not found. Please activate it manually.")
                return
    
    # Check requirements
    try:
        import flask
        import flask_socketio
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                              str(web_dir / "requirements.txt")])
    
    # Launch browser after a short delay
    def open_browser():
        webbrowser.open('http://localhost:5000')
    
    # Start the web application
    print(f"Starting web server on http://localhost:5000...")
    print("The web UI will open automatically in your browser.")
    import threading
    threading.Timer(1.5, open_browser).start()
    
    # Run the Flask app
    os.chdir(web_dir)
    from web.app import app, socketio
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
