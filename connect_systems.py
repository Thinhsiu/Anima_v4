"""
System Connection Script for Anima AI

This script helps connect the various components of Anima AI:
- Connects voice system to emotion analysis
- Connects emotion system to memory system
- Verifies all connections are working properly
"""

import os
import sys
import logging
import importlib
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connect_systems")

def connect_emotion_voice():
    """Connect emotion analysis system to voice interaction system.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    print("\nAttempting to connect emotion and voice systems...")
    
    # Check for voice_bridge module
    try:
        from emotion.voice_bridge import connect_to_voice_system
        connected = connect_to_voice_system()
        if connected:
            print("✓ Voice-emotion bridge connected successfully")
            return True
        else:
            print("✗ Voice-emotion bridge connection failed")
            return False
    except ImportError:
        print("✗ Voice-emotion bridge module not found")
        return False
    except Exception as e:
        print(f"✗ Error connecting voice-emotion bridge: {e}")
        return False

def connect_emotion_memory():
    """Connect emotion system to memory/awareness system.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    print("\nAttempting to connect emotion and memory systems...")
    
    # First check if memory_bridge exists
    try:
        from core.memory_bridge import get_memory_bridge, MEMORY_BRIDGE_AVAILABLE
        
        if not MEMORY_BRIDGE_AVAILABLE:
            print("✗ Memory bridge is not available")
            return False
            
        # Now check if the memory integration is accessible
        memory_bridge = get_memory_bridge()
        if memory_bridge and memory_bridge.available:
            print("✓ Memory bridge connected to memory integration")
            
            # Test if the memory bridge can store data
            test_data = {
                "status": "success",
                "emotions": {
                    "dominant_emotion": "neutral",
                    "dominant_intensity": "low"
                },
                "profile_id": "test",
                "timestamp": 0
            }
            
            if memory_bridge.add_emotional_data(test_data):
                print("✓ Memory bridge successfully stored test data")
                return True
            else:
                print("✗ Memory bridge failed to store test data")
                return False
        else:
            print("✗ Memory bridge available but couldn't access memory integration")
            return False
            
    except ImportError:
        print("✗ Memory bridge module not found")
        return False
    except Exception as e:
        print(f"✗ Error connecting memory bridge: {e}")
        return False

def verify_emotion_system():
    """Verify the emotion system is functioning properly.
    
    Returns:
        bool: True if system is functional, False otherwise
    """
    print("\nVerifying emotion system functionality...")
    
    try:
        # Try to import and use the emotion system
        from emotion.integration import get_instance, analyze_user_input
        
        emotion_system = get_instance()
        if not emotion_system:
            print("✗ Emotion system instance not available")
            return False
            
        # Test analyze functionality
        test_input = "I'm feeling happy today"
        analysis = analyze_user_input(test_input)
        
        if analysis and analysis.get("status") == "success":
            emotions = analysis.get("emotions", {})
            print(f"✓ Emotion system analyzed text successfully")
            print(f"  Detected emotion: {emotions.get('dominant_emotion', 'unknown')}")
            return True
        else:
            print("✗ Emotion analysis failed")
            return False
            
    except ImportError:
        print("✗ Emotion system modules not found")
        return False
    except Exception as e:
        print(f"✗ Error verifying emotion system: {e}")
        return False

def connect_all_systems():
    """Attempt to connect all systems.
    
    Returns:
        Dict[str, bool]: Status of each connection
    """
    results = {
        "emotion_system": False,
        "voice_connection": False,
        "memory_connection": False
    }
    
    # Verify emotion system first
    results["emotion_system"] = verify_emotion_system()
    
    # Only attempt connections if emotion system is working
    if results["emotion_system"]:
        results["voice_connection"] = connect_emotion_voice()
        results["memory_connection"] = connect_emotion_memory()
    
    # Print summary
    print("\n=== CONNECTION SUMMARY ===")
    for system, status in results.items():
        status_text = "CONNECTED" if status else "FAILED"
        print(f"- {system.replace('_', ' ').title()}: {status_text}")
    print("========================\n")
    
    return results

if __name__ == "__main__":
    print("Anima AI System Connector")
    print("=========================")
    
    connect_all_systems()
