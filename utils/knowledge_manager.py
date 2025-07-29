import os
import json
import time
import shutil
import datetime
from pathlib import Path

# Constants
# Use local directory instead of hardcoded drive
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KNOWLEDGE_DIR = str(BASE_DIR / "knowledge")
DRIVE_PATH = str(BASE_DIR.parent)  # Use parent directory of Anima_v4 for drive stats
MAX_DRIVE_USAGE_PERCENT = 40  # Maximum percentage of drive to use

# Create knowledge directory if it doesn't exist
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

def get_drive_stats():
    """Get total and free space on the E: drive"""
    total, used, free = shutil.disk_usage(DRIVE_PATH)
    return {
        "total_gb": total / (1024**3),
        "used_gb": used / (1024**3),
        "free_gb": free / (1024**3),
        "used_percent": (used / total) * 100
    }

def get_knowledge_size():
    """Calculate the total size of the knowledge directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(KNOWLEDGE_DIR):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # Return size in GB

def prune_old_knowledge():
    """Remove oldest knowledge files when storage exceeds threshold"""
    # Check if we're exceeding the maximum allowed usage
    drive_stats = get_drive_stats()
    if drive_stats["used_percent"] > MAX_DRIVE_USAGE_PERCENT:
        # Get all knowledge files with their creation times
        knowledge_files = []
        for dirpath, dirnames, filenames in os.walk(KNOWLEDGE_DIR):
            for f in filenames:
                if f.endswith('.json'):
                    fp = os.path.join(dirpath, f)
                    creation_time = os.path.getctime(fp)
                    size = os.path.getsize(fp)
                    knowledge_files.append((fp, creation_time, size))
        
        # Sort by creation time (oldest first)
        knowledge_files.sort(key=lambda x: x[1])
        
        # Delete oldest files until we're under the threshold
        for fp, ctime, size in knowledge_files:
            os.remove(fp)
            print(f"Pruned old knowledge file: {fp}")
            
            # Check if we're now under the threshold
            if get_drive_stats()["used_percent"] <= MAX_DRIVE_USAGE_PERCENT:
                break

def save_knowledge(topic, content):
    """Save knowledge to a file"""
    # Ensure we have space by pruning if necessary
    prune_old_knowledge()
    
    # Create a filename based on topic and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{sanitize_filename(topic)}_{timestamp}.json"
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    
    # Store the knowledge with metadata
    knowledge = {
        "topic": topic,
        "content": content,
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "gpt4_download"
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(knowledge, f, indent=2)
    
    return filepath

def sanitize_filename(name):
    """Convert a string to a valid filename"""
    # Replace spaces and invalid chars with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.replace(' ', '_')[:50]  # Limit length to 50 chars

def get_knowledge(topic=None):
    """Retrieve knowledge from files, optionally filtered by topic"""
    knowledge_list = []
    
    for dirpath, dirnames, filenames in os.walk(KNOWLEDGE_DIR):
        for f in filenames:
            if f.endswith('.json'):
                try:
                    with open(os.path.join(dirpath, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if topic is None or topic.lower() in data["topic"].lower():
                            knowledge_list.append(data)
                except Exception as e:
                    print(f"Error reading knowledge file {f}: {e}")
    
    # Sort by timestamp (newest first)
    knowledge_list.sort(key=lambda x: x["timestamp"], reverse=True)
    return knowledge_list

def download_deep_knowledge(query, use_gpt4=True):
    """Download deep knowledge about a topic using GPT-4"""
    from llm.openai_llm import query_openai
    
    # Create a prompt that asks for deep, comprehensive knowledge
    deep_prompt = f"""I want to expand my knowledge base with deep, comprehensive information about: {query}
    
    Please provide detailed knowledge that includes:
    1. Core concepts and definitions
    2. Historical context and development
    3. Key principles and frameworks
    4. Practical applications
    5. Current state of the art
    6. Future implications
    
    Format your response as comprehensive knowledge suited for an AI assistant's knowledge base.
    """
    
    # Use GPT-4 for this deep knowledge request
    response = query_openai(deep_prompt, use_gpt4=True)
    
    # Save the downloaded knowledge
    filepath = save_knowledge(query, response)
    
    return {
        "topic": query,
        "filepath": filepath,
        "content_preview": response[:150] + "..." if len(response) > 150 else response
    }

def get_knowledge_stats():
    """Get statistics about the knowledge base"""
    files = get_knowledge()
    return {
        "total_files": len(files),
        "total_size_gb": get_knowledge_size(),
        "oldest_file": min(files, key=lambda x: x["timestamp"])["topic"] if files else None,
        "newest_file": max(files, key=lambda x: x["timestamp"])["topic"] if files else None,
        "drive_stats": get_drive_stats()
    }
