"""
Enhanced Awareness Integration for Anima

This module integrates all enhanced awareness components:
1. Emotion Recognition
2. Contextual Learning
3. Interest Tracking
4. Conversation Summarization
5. Adaptive Persona

It provides a unified interface for Anima's enhanced awareness capabilities.
"""

import os
import sys
from pathlib import Path
import importlib.util
import traceback
import logging
# Set up logger
logger = logging.getLogger(__name__)


# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Core awareness system - create basic implementation if not available
try:
    from core.awareness import awareness, add_conversation, enhance_prompt_with_awareness
    from core.memory_integration import memory_integration, enhance_prompt_with_memory
    MEMORY_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.debug("Creating fallback awareness system...")
    # Create minimal awareness system
    class BasicAwarenessSystem:
        """Basic fallback awareness system"""
        def __init__(self):
            self.context = {}
            self.history = []
            
        def register_subsystem(self, name, instance=None):
            description = str(instance) if instance else "Unknown instance"
            self.context[name] = {"description": description, "active": True}
            # Update global module availability
            if name == "memory_integration":
                global MEMORY_INTEGRATION_AVAILABLE
                MEMORY_INTEGRATION_AVAILABLE = True
            logger.debug(f"Registered subsystem: {name}")
            
        def get_context_for_input(self, user_input):
            return {"source": "basic_awareness", "context": "Limited functionality mode"}
    
    # Create global instance with basic functionality
    awareness = BasicAwarenessSystem()
    
    # Define basic functions
    def add_conversation(user_input, assistant_response):
        if hasattr(awareness, 'history'):
            awareness.history.append({"user": user_input, "assistant": assistant_response})
            
    def enhance_prompt_with_awareness(prompt, user_input=None):
        return prompt
        
    # Create minimal memory integration system
    class BasicMemoryIntegration:
        """Basic fallback memory integration system"""
        def __init__(self):
            self.memories = {}
            
        def get_relevant_memories(self, user_input, limit=3):
            return []
            
    # Create global instance
    memory_integration = BasicMemoryIntegration()
    MEMORY_INTEGRATION_AVAILABLE = False
    
    # Define basic function
    def enhance_prompt_with_memory(prompt, user_input):
        return prompt

# Import enhanced modules if available
def import_module(name):
    """Import a module and return None if not available"""
    try:
        return importlib.import_module(f"core.{name}")
    except ImportError:
        print(f"Warning: {name} module not available.")
        return None

# Import all enhanced modules
emotion_recognition = import_module("emotion_recognition")
contextual_learning = import_module("contextual_learning")
interest_tracking = import_module("interest_tracking")
conversation_summarization = import_module("conversation_summarization")
adaptive_persona = import_module("adaptive_persona")

# Track which modules are available
available_modules = {
    "emotion_recognition": emotion_recognition is not None,
    "contextual_learning": contextual_learning is not None,
    "interest_tracking": interest_tracking is not None,
    "conversation_summarization": conversation_summarization is not None,
    "adaptive_persona": adaptive_persona is not None,
    "core_awareness": awareness is not None,
    "memory_integration": MEMORY_INTEGRATION_AVAILABLE
}

def register_with_core_awareness(*args, **kwargs):
    """
    Register a subsystem with the core awareness system
    
    Args:
        Can be called with positional or keyword arguments:
        - First positional arg or subsystem_name: Name of the subsystem to register
        - Second positional arg or instance: Instance of the subsystem to register (optional)
        
    Returns:
        True if registration successful, False otherwise
    """
    global available_modules, MEMORY_INTEGRATION_AVAILABLE
    
    # Handle both positional and keyword arguments for backward compatibility
    subsystem_name = None
    instance = None
    
    # First, check for positional arguments
    if len(args) >= 1:
        subsystem_name = args[0]
    if len(args) >= 2:
        instance = args[1]
    
    # Next, check for keyword arguments (override positional if provided)
    if 'subsystem_name' in kwargs:
        subsystem_name = kwargs['subsystem_name']
    if 'instance' in kwargs:
        instance = kwargs['instance']
    
    # Exit early if no valid subsystem name
    if not subsystem_name or not isinstance(subsystem_name, str):
        return False
        
    try:
        # Register with core awareness if available
        if awareness is not None:
            if hasattr(awareness, 'register_subsystem'):
                if instance is not None:
                    awareness.register_subsystem(subsystem_name, instance)
                else:
                    awareness.register_subsystem(subsystem_name, subsystem_name)
            
        # Update the available modules list
        if subsystem_name in available_modules:
            available_modules[subsystem_name] = True
            
        # Update memory integration specifically
        if subsystem_name == "memory_integration":
            MEMORY_INTEGRATION_AVAILABLE = True
            
        return True
    except Exception as e:
        logger.error(f"Error registering {subsystem_name} with core awareness: {e}")
        traceback.print_exc()
        return False

# Export this for other modules to use
__all__ = [
    "enhance_prompt_with_all", 
    "process_exchange", 
    "get_system_status",
    "available_modules",
    "register_with_core_awareness",
    "check_modules"
]


def check_modules():
    """Print the status of all modules to help with debugging"""
    logger.debug("=== ENHANCED AWARENESS MODULE STATUS ===")
    for name, is_available in available_modules.items():
        status = "AVAILABLE" if is_available else "NOT AVAILABLE"
        logger.debug(f"- {name}: {status}")
    print("======================================\n")


def enhance_prompt_with_all(prompt, user_input=None, conversation_id=None):
    """
    Enhance a prompt with all available awareness enhancements
    
    Args:
        prompt: The original prompt to enhance
        user_input: The user's latest input (optional)
        conversation_id: ID of the current conversation (optional)
    
    Returns:
        Enhanced prompt with all available awareness contexts
    """
    enhanced_prompt = prompt
    
    # Only enhance if we have both prompt and user input
    if not prompt or not user_input:
        return prompt
        
    try:
        # 1. Core awareness
        if available_modules["core_awareness"]:
            enhanced_prompt = enhance_prompt_with_awareness(enhanced_prompt)
        
        # 2. Emotion recognition
        if available_modules["emotion_recognition"] and user_input:
            enhanced_prompt = emotion_recognition.enhance_prompt_with_emotion(enhanced_prompt, user_input)
        
        # 3. Contextual learning
        if available_modules["contextual_learning"] and user_input:
            enhanced_prompt = contextual_learning.enhance_prompt_with_concepts(enhanced_prompt, user_input)
        
        # 4. Interest tracking
        if available_modules["interest_tracking"] and user_input:
            enhanced_prompt = interest_tracking.enhance_prompt_with_interests(enhanced_prompt, user_input)
        
        # 5. Conversation summarization
        if available_modules["conversation_summarization"]:
            enhanced_prompt = conversation_summarization.enhance_prompt_with_summaries(enhanced_prompt, conversation_id)
        
        # 6. Adaptive persona
        if available_modules["adaptive_persona"]:
            enhanced_prompt = adaptive_persona.enhance_prompt_with_style(enhanced_prompt)
            
    except Exception as e:
        print(f"Error enhancing prompt with awareness: {e}")
        traceback.print_exc()
    
    return enhanced_prompt


def process_exchange(user_input, assistant_response, conversation_id=None):
    """
    Process a conversation exchange through all awareness systems
    
    Args:
        user_input: The user's message
        assistant_response: Anima's response
        conversation_id: ID of the conversation (optional)
        
    Returns:
        Modified assistant response (if any adjustments were made)
    """
    modified_response = assistant_response
    
    if not user_input or not assistant_response:
        return assistant_response
        
    try:
        # Extract topics for shared context
        topics = []
        if available_modules["core_awareness"] and hasattr(awareness, "context_memory"):
            recent = awareness.context_memory.get_recent_exchanges(1)
            if recent and "topics" in recent[0]:
                topics = recent[0]["topics"]
                
        # 1. Core awareness
        if available_modules["core_awareness"]:
            add_conversation(user_input, assistant_response)
        
        # 2. Emotion recognition
        if available_modules["emotion_recognition"]:
            emotions = emotion_recognition.detect_emotions(user_input)
            modified_response = emotion_recognition.get_emotional_response(modified_response, emotions)
        
        # 3. Contextual learning
        if available_modules["contextual_learning"]:
            contextual_learning.learn_from_text(user_input, conversation_id)
            contextual_learning.process_response_for_learning(user_input, assistant_response, conversation_id)
        
        # 4. Interest tracking
        if available_modules["interest_tracking"]:
            interest_tracking.update_interest_model(user_input, topics)
        
        # 5. Conversation summarization
        if available_modules["conversation_summarization"]:
            # Create metadata with topics and any other context
            metadata = {"topics": topics} if topics else {}
            conversation_summarization.add_exchange(user_input, assistant_response, conversation_id, metadata)
        
        # 6. Adaptive persona
        if available_modules["adaptive_persona"]:
            adaptive_persona.update_persona(user_input, assistant_response)
            modified_response = adaptive_persona.adjust_response(modified_response)
            
    except Exception as e:
        print(f"Error processing exchange through awareness systems: {e}")
        traceback.print_exc()
        
    return modified_response


def get_system_status():
    """Get the status of all enhanced awareness systems"""
    status = {
        "available_modules": available_modules,
        "details": {}
    }
    
    # Get interest summary if available
    if available_modules["interest_tracking"]:
        try:
            status["details"]["interests"] = interest_tracking.get_interest_summary()
        except Exception:
            pass
    
    # Get style guidance if available
    if available_modules["adaptive_persona"]:
        try:
            status["details"]["persona_style"] = adaptive_persona.get_style_guidance()
        except Exception:
            pass
            
    return status


def register_with_core_awareness():
    """Register enhanced modules with core awareness system"""
    if not available_modules["core_awareness"] or not hasattr(awareness, "register_subsystem"):
        return False
        
    try:
        # Register each module
        for name in available_modules:
            if name != "core_awareness" and available_modules[name]:
                awareness.register_subsystem(name, f"Enhanced awareness: {name}")
        return True
    except Exception as e:
        print(f"Error registering with core awareness: {e}")
        return False


# Auto-register with core awareness if available
if available_modules["core_awareness"]:
    register_with_core_awareness()
