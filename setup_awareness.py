"""
Setup script for Anima's awareness system

This script creates necessary directories and initializes empty files
for the enhanced awareness system to work properly.
"""

import os
import sys
import json
from pathlib import Path

def create_directory(path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)
    else:
        print(f"Directory already exists: {path}")
        
def create_empty_json(path, default_content=None):
    """Create an empty JSON file if it doesn't exist"""
    if not os.path.exists(path):
        print(f"Creating file: {path}")
        with open(path, 'w', encoding='utf-8') as f:
            if default_content is None:
                default_content = {}
            json.dump(default_content, f, indent=2)
    else:
        print(f"File already exists: {path}")

def setup_directories():
    """Set up all necessary directories for the awareness system"""
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create main directories
    create_directory(os.path.join(base_dir, "memories"))
    create_directory(os.path.join(base_dir, "memories", "emotions"))
    create_directory(os.path.join(base_dir, "memories", "concepts"))
    create_directory(os.path.join(base_dir, "memories", "interests"))
    create_directory(os.path.join(base_dir, "memories", "summaries"))
    create_directory(os.path.join(base_dir, "memories", "persona"))
    
    # Create base files for each system
    create_empty_json(os.path.join(base_dir, "memories", "emotions", "emotion_history.json"))
    create_empty_json(os.path.join(base_dir, "memories", "emotions", "emotional_baseline.json"))
    
    create_empty_json(os.path.join(base_dir, "memories", "concepts", "learned_concepts.json"))
    create_empty_json(os.path.join(base_dir, "memories", "concepts", "concept_index.json"))
    
    create_empty_json(os.path.join(base_dir, "memories", "interests", "interest_model.json"))
    create_empty_json(os.path.join(base_dir, "memories", "interests", "topic_history.json"))
    
    create_empty_json(os.path.join(base_dir, "memories", "summaries", "conversation_summaries.json"))
    
    create_empty_json(os.path.join(base_dir, "memories", "persona", "persona_model.json"))
    create_empty_json(os.path.join(base_dir, "memories", "persona", "style_preferences.json"))
    
    # Core awareness files
    create_empty_json(os.path.join(base_dir, "memories", "context_awareness.json"))
    create_empty_json(os.path.join(base_dir, "memories", "user_preferences.json"))
    
    print("\nAll directories and base files created successfully!")
    print("The awareness system is now ready to be used.")

def check_module_imports():
    """Try to import each module to check for issues"""
    print("\nChecking module imports...")
    
    modules_to_check = [
        "core.awareness", 
        "core.memory_integration",
        "core.emotion_recognition",
        "core.contextual_learning",
        "core.interest_tracking",
        "core.conversation_summarization",
        "core.adaptive_persona",
        "core.enhanced_awareness_integration"
    ]
    
    results = {}
    
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            results[module_name] = "IMPORTED SUCCESSFULLY"
        except ImportError as e:
            results[module_name] = f"IMPORT ERROR: {str(e)}"
    
    # Print results
    print("\n=== MODULE IMPORT STATUS ===")
    for module, status in results.items():
        print(f"{module}: {status}")
    print("===========================")

if __name__ == "__main__":
    print("Setting up Anima's awareness system...")
    setup_directories()
    check_module_imports()
