"""
Backup Utility for Anima Project

Creates a complete backup of the project before encryption or other major changes.
"""

import os
import sys
import shutil
import datetime
import zipfile

def create_backup(backup_dir=None, include_backups=False, exclude_dirs=None):
    """
    Create a complete backup of the project
    
    Args:
        backup_dir: Directory to save backup in (default: project_dir/backups)
        include_backups: Whether to include previous backups
        exclude_dirs: List of directories to exclude from backup
        
    Returns:
        Path to the created backup file
    """
    # Get base project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default backup directory if not specified
    if backup_dir is None:
        backup_dir = os.path.join(project_dir, "backups")
    
    # Create backup directory if it doesn't exist
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create timestamp for backup filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"anima_backup_{timestamp}.zip"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Default exclude directories
    if exclude_dirs is None:
        exclude_dirs = [
            ".git",
            "__pycache__",
            "backups",
            "venv",
            "env",
            ".vscode",
            ".idea"
        ]
    
    # Add the backups directory to exclude list if not including backups
    if not include_backups and "backups" not in exclude_dirs:
        exclude_dirs.append("backups")
    
    print(f"Creating backup of Anima project at: {backup_path}")
    print("This may take a few moments depending on project size...")
    
    # Track total files and size
    total_files = 0
    total_size = 0
    
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
                    # Skip if file is too large (> 100MB)
                    file_size = os.path.getsize(file_path)
                    if file_size > 100 * 1024 * 1024:
                        print(f"Skipping large file: {zip_path} ({file_size / 1024 / 1024:.2f} MB)")
                        continue
                    
                    zipf.write(file_path, zip_path)
                    total_files += 1
                    total_size += file_size
                except Exception as e:
                    print(f"Error adding {file_path}: {e}")
    
    # Convert total size to MB
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\nBackup complete!")
    print(f"Backup file: {backup_path}")
    print(f"Total files: {total_files}")
    print(f"Total size: {total_size_mb:.2f} MB")
    
    return backup_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a backup of the Anima project")
    parser.add_argument("--backup-dir", help="Directory to save backup in")
    parser.add_argument("--include-backups", action="store_true", help="Include previous backups")
    
    args = parser.parse_args()
    
    backup_path = create_backup(args.backup_dir, args.include_backups)
    print(f"Backup saved to: {backup_path}")
