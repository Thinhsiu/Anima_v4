"""
File Encryption Utility for Anima

This module provides functions to encrypt and decrypt sensitive data files
while leaving code files accessible for future modifications.
"""

import os
import sys
import json
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def derive_key(password, salt=None):
    """Derive a key from a password"""
    if salt is None:
        salt = os.urandom(16)
        
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key, salt


def encrypt_file(file_path, password, backup=True):
    """
    Encrypt a file using a password
    
    Args:
        file_path: Path to the file to encrypt
        password: Password to use for encryption
        backup: Whether to create a backup before encrypting
        
    Returns:
        True if encryption was successful, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
            
        # Generate a key from the password
        key, salt = derive_key(password)
        cipher = Fernet(key)
        
        # Read the file content
        with open(file_path, 'rb') as f:
            data = f.read()
            
        # Create a backup if requested
        if backup:
            backup_path = file_path + '.backup'
            with open(backup_path, 'wb') as f:
                f.write(data)
                
        # Encrypt the data
        encrypted_data = cipher.encrypt(data)
        
        # Add the salt to the beginning of the file
        final_data = salt + encrypted_data
        
        # Write the encrypted data back to the file
        with open(file_path + '.encrypted', 'wb') as f:
            f.write(final_data)
            
        # Replace the original file with a pointer to the encrypted file
        with open(file_path, 'w') as f:
            f.write(f"ENCRYPTED_FILE: {os.path.basename(file_path)}.encrypted")
            
        print(f"Encrypted: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error encrypting {file_path}: {str(e)}")
        return False


def decrypt_file(file_path, password):
    """
    Decrypt a file using a password
    
    Args:
        file_path: Path to the file to decrypt
        password: Password used for encryption
        
    Returns:
        True if decryption was successful, False otherwise
    """
    try:
        # Check if this is a pointer file
        with open(file_path, 'r') as f:
            content = f.read().strip()
            
        if not content.startswith("ENCRYPTED_FILE:"):
            print(f"Not an encrypted pointer file: {file_path}")
            return False
            
        # Get the encrypted file name
        encrypted_file = content.split(":", 1)[1].strip()
        encrypted_path = os.path.join(os.path.dirname(file_path), encrypted_file)
        
        if not os.path.exists(encrypted_path):
            print(f"Encrypted file not found: {encrypted_path}")
            return False
            
        # Read the encrypted data
        with open(encrypted_path, 'rb') as f:
            data = f.read()
            
        # Extract the salt and encrypted data
        salt = data[:16]
        encrypted_data = data[16:]
        
        # Derive the key from the password and salt
        key, _ = derive_key(password, salt)
        cipher = Fernet(key)
        
        # Decrypt the data
        decrypted_data = cipher.decrypt(encrypted_data)
        
        # Write the decrypted data back to the original file
        with open(file_path + '.decrypted', 'wb') as f:
            f.write(decrypted_data)
            
        print(f"Decrypted to: {file_path}.decrypted")
        return True
        
    except Exception as e:
        print(f"Error decrypting {file_path}: {str(e)}")
        return False


def encrypt_directory(directory, password, patterns=None, exclude_patterns=None, backup=True):
    """
    Encrypt files in a directory matching specific patterns
    
    Args:
        directory: Directory to encrypt files in
        password: Password to use for encryption
        patterns: List of glob patterns to include (e.g. ['*.json', '*.txt'])
        exclude_patterns: List of glob patterns to exclude
        backup: Whether to create backups before encrypting
        
    Returns:
        Number of files encrypted successfully
    """
    if patterns is None:
        patterns = ['*.json', '*.txt', '*.csv', '*.md']
        
    if exclude_patterns is None:
        exclude_patterns = []
        
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check if file matches any pattern
            if not any(Path(file_path).match(pattern) for pattern in patterns):
                continue
                
            # Check if file matches any exclude pattern
            if any(Path(file_path).match(pattern) for pattern in exclude_patterns):
                continue
                
            # Skip already encrypted files
            if file.endswith('.encrypted') or file.endswith('.backup') or file.endswith('.decrypted'):
                continue
                
            # Try to read the first line to check if it's already an encrypted pointer
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                if first_line.startswith("ENCRYPTED_FILE:"):
                    continue
            except:
                pass
                
            # Encrypt the file
            if encrypt_file(file_path, password, backup):
                count += 1
                
    return count


def main():
    """Command line interface for file encryption"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Encrypt or decrypt files for Anima')
    parser.add_argument('action', choices=['encrypt', 'decrypt'], help='Action to perform')
    parser.add_argument('--password', required=True, help='Password for encryption/decryption')
    parser.add_argument('--file', help='Single file to encrypt/decrypt')
    parser.add_argument('--dir', help='Directory to encrypt/decrypt')
    parser.add_argument('--patterns', nargs='+', default=['*.json', '*.txt'], help='File patterns to include')
    parser.add_argument('--exclude', nargs='+', default=[], help='File patterns to exclude')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backups')
    
    args = parser.parse_args()
    
    if args.action == 'encrypt':
        if args.file:
            result = encrypt_file(args.file, args.password, not args.no_backup)
            print(f"Encryption {'successful' if result else 'failed'}")
        elif args.dir:
            count = encrypt_directory(args.dir, args.password, args.patterns, args.exclude, not args.no_backup)
            print(f"Encrypted {count} files successfully")
        else:
            parser.error("Either --file or --dir must be specified")
    else:  # decrypt
        if args.file:
            result = decrypt_file(args.file, args.password)
            print(f"Decryption {'successful' if result else 'failed'}")
        else:
            parser.error("Only single file decryption is supported")


if __name__ == "__main__":
    main()
