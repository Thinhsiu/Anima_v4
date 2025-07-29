# Core package initialization
# This package contains central intelligence components for Anima

# Import key functions for easy access
try:
    from .awareness import awareness, add_conversation, enhance_prompt
    from .memory_integration import memory_integration, get_relevant_memories, enhance_prompt_with_memories
    
    # Define alias functions for consistent naming across the codebase
    enhance_prompt_with_awareness = enhance_prompt
    enhance_prompt_with_memory = enhance_prompt_with_memories
    get_awareness_context = lambda x: awareness.get_context_for_input(x) if hasattr(awareness, 'get_context_for_input') else {}
except ImportError:
    print("Warning: Could not import all core components")
