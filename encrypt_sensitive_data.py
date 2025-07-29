"""
Encrypt Sensitive Anima Data Files

This script encrypts persona, knowledge, and memory files while leaving
code files accessible. Run with a password of your choice.
"""

import os
import sys
from pathlib import Path
import argparse

# Make sure we can import from utils
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from utils.file_encryption import encrypt_directory
except ImportError:
    print("Error: Could not import encryption module. Make sure utils/file_encryption.py exists.")
    sys.exit(1)

def encrypt_sensitive_files(password, create_backup=True):
    """Encrypt sensitive data files while leaving code accessible"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define directories with sensitive data
    sensitive_dirs = [
        os.path.join(base_dir, "persona"),
        os.path.join(base_dir, "memories"),
        os.path.join(base_dir, "knowledge")
    ]
    
    # Define patterns for sensitive files
    sensitive_patterns = ["*.json", "*.txt", "*.csv", "*.md", "*.yaml", "*.yml"]
    
    # Define patterns to exclude (files we still need readable)
    exclude_patterns = [
        "README*", 
        "LICENSE*", 
        "requirements.txt",
        "*.py",
        "*.bak",
        "*.backup",
        "*.encrypted",
        "*.pyc",
        "*.git*"
    ]
    
    total_encrypted = 0
    
    print("Starting encryption of sensitive files...")
    print("This will encrypt data files but leave code files accessible.")
    
    # Encrypt each directory
    for directory in sensitive_dirs:
        if os.path.exists(directory):
            print(f"\nEncrypting files in {directory}...")
            count = encrypt_directory(
                directory, 
                password, 
                sensitive_patterns,
                exclude_patterns,
                create_backup
            )
            total_encrypted += count
            print(f"Encrypted {count} files in {directory}")
        else:
            print(f"Directory not found, skipping: {directory}")
    
    print(f"\nEncryption complete. {total_encrypted} files were encrypted.")
    print("Backups were created with .backup extension.")
    print("\nIMPORTANT: Store your password in a safe place. Without it, you won't be able to decrypt the files.")
    
    return total_encrypted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encrypt Anima's sensitive data files")
    parser.add_argument("--password", required=True, help="Password for encryption")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files")
    
    args = parser.parse_args()
    
    encrypt_sensitive_files(args.password, not args.no_backup)
