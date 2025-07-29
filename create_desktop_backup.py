"""
Create a complete backup of Anima V4 to the Desktop
"""

import os
import sys
import shutil
import datetime
import zipfile

def create_backup(project_dir, backup_path):
    """Create a complete backup of the project directory"""
    print(f"Creating backup of {project_dir}")
    print(f"Saving to: {backup_path}")
    
    # Track total files and size
    total_files = 0
    total_size = 0
    
    # Directories to exclude
    exclude_dirs = [
        ".git",
        "__pycache__",
        "venv",
        "env",
        ".vscode",
        ".idea"
    ]
    
    # Create zip file
    with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through project directory
        for root, dirs, files in os.walk(project_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # Get relative path
            rel_path = os.path.relpath(root, project_dir)
            if rel_path == ".":
                rel_path = ""
            
            # Add files to zip
            for file in files:
                # Skip the backup file itself
                if os.path.abspath(os.path.join(root, file)) == os.path.abspath(backup_path):
                    continue
                
                # Add file to zip
                file_path = os.path.join(root, file)
                zip_path = os.path.join(rel_path, file)
                
                try:
                    # Get file size for tracking
                    file_size = os.path.getsize(file_path)
                    
                    # Add file to zip (no size limit)
                    print(f"Adding: {zip_path} ({file_size / 1024 / 1024:.2f} MB)")
                    zipf.write(file_path, zip_path)
                    total_files += 1
                    total_size += file_size
                except Exception as e:
                    print(f"Error adding {file_path}: {e}")
    
    # Convert total size to MB
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\nBackup complete!")
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size_mb:.2f} MB")

if __name__ == "__main__":
    # Get project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get desktop path
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    # Create backup path
    backup_filename = "Anima V4 backup.zip"
    backup_path = os.path.join(desktop_path, backup_filename)
    
    # Create backup
    create_backup(project_dir, backup_path)
    
    print(f"\nBackup saved to desktop: {backup_path}")
