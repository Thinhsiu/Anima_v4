"""
Memory integration for Anima's awareness system

Connects memory subsystems and provides a unified interface for memory access.
"""

import os
import sys
import json
import datetime
from pathlib import Path
import threading

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from core.awareness import awareness

# Try to import knowledge manager
try:
    from utils.knowledge_manager import retrieve_knowledge, save_knowledge
    KNOWLEDGE_MANAGER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_MANAGER_AVAILABLE = False

# Try to import vision smart memory manager
try:
    from vision.smart_memory_manager import smart_memory_manager
    VISION_MEMORY_AVAILABLE = True
except ImportError:
    VISION_MEMORY_AVAILABLE = False


class MemoryIntegration:
    """
    Integrates different memory subsystems for unified access
    
    This class connects:
    1. Short-term conversation memory
    2. Long-term knowledge base
    3. Vision/media memories
    4. User preferences and personalization
    """
    
    def __init__(self):
        """Initialize the memory integration system"""
        self.memory_sources = {}
        
        # Register with awareness system
        awareness.register_subsystem("memory", self)
        
        # Register available memory sources
        if KNOWLEDGE_MANAGER_AVAILABLE:
            self.memory_sources["knowledge"] = {
                "retrieve": retrieve_knowledge,
                "save": save_knowledge
            }
            
        if VISION_MEMORY_AVAILABLE:
            self.memory_sources["vision"] = {
                "manager": smart_memory_manager
            }
    
    def get_relevant_memories(self, user_input, limit=5):
        """
        Retrieve memories relevant to the current user input
        
        Args:
            user_input: The user's input text
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memory objects
        """
        memories = []
        
        # Check for vision memory references
        if VISION_MEMORY_AVAILABLE:
            is_reference, memory_info = smart_memory_manager.detect_memory_reference(user_input)
            if is_reference and memory_info:
                memories.append({
                    "type": "vision",
                    "source": "vision_memory",
                    "content": f"Referenced image at {memory_info.get('path')}",
                    "timestamp": memory_info.get("timestamp", ""),
                    "path": memory_info.get("path", "")
                })
        
        # Check knowledge base if available
        if KNOWLEDGE_MANAGER_AVAILABLE:
            # Extract potential search terms from user input
            search_terms = self._extract_search_terms(user_input)
            
            # Search knowledge base for relevant items
            for term in search_terms:
                knowledge_items = retrieve_knowledge(term, limit=2)
                if knowledge_items:
                    for item in knowledge_items:
                        memories.append({
                            "type": "knowledge",
                            "source": "knowledge_base",
                            "content": item.get("content", ""),
                            "timestamp": item.get("timestamp", ""),
                            "topic": item.get("topic", "")
                        })
        
        # Add awareness context memories
        context = awareness.get_relevant_context(user_input)
        if "related_topics" in context and context["related_topics"]:
            for topic in context["related_topics"]:
                memories.append({
                    "type": "conversation",
                    "source": "conversation_history",
                    "content": f"User: {topic.get('user', '')} | Anima: {topic.get('ai', '')}",
                    "timestamp": topic.get("timestamp", "")
                })
        
        # Sort by relevance (simplified - just using recency)
        # In a full implementation, this would use semantic relevance
        memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return memories[:limit]
    
    def _extract_search_terms(self, text):
        """
        Extract potential search terms from text
        
        This is a simplified approach - in a full implementation,
        this would use NLP to extract key entities and concepts
        """
        # Simple approach - just get longer words
        words = text.lower().split()
        terms = [w for w in words if len(w) > 5]  # Longer words only
        
        # Deduplicate
        terms = list(set(terms))
        
        # Limit to top 3
        return terms[:3]
    
    def add_memory(self, memory_type, content, metadata=None):
        """
        Add a new memory to the appropriate storage system
        
        Args:
            memory_type: Type of memory (knowledge, vision, preference)
            content: The content to store
            metadata: Additional metadata for the memory
            
        Returns:
            Success indicator and memory ID
        """
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
        # Handle different memory types
        if memory_type == "knowledge" and KNOWLEDGE_MANAGER_AVAILABLE:
            # Save to knowledge base
            topic = metadata.get("topic", "general")
            save_knowledge(content, topic)
            return True, f"knowledge:{topic}"
            
        elif memory_type == "vision" and VISION_MEMORY_AVAILABLE:
            # Check if this is a file path
            if "path" in metadata:
                memory_mode = metadata.get("memory_mode", "long_term")
                tags = metadata.get("tags", [])
                
                memory_id = smart_memory_manager.save_to_memory(metadata["path"], memory_mode, tags)
                return bool(memory_id), memory_id
        
        # Default - store in awareness system
        if "important_topics" not in awareness.context_memory:
            awareness.context_memory["important_topics"] = []
            
        awareness.context_memory["important_topics"].append({
            "content": content,
            "metadata": metadata,
            "timestamp": metadata.get("timestamp")
        })
        
        return True, "awareness:context_memory"
    
    def format_memories_for_prompt(self, memories):
        """
        Format memories for inclusion in the prompt
        
        Args:
            memories: List of memory objects
            
        Returns:
            Formatted string for insertion into prompt
        """
        if not memories:
            return ""
            
        formatted = []
        
        for memory in memories:
            memory_type = memory.get("type", "unknown")
            content = memory.get("content", "")
            source = memory.get("source", "")
            
            # Format differently based on type
            if memory_type == "vision":
                formatted.append(f"Image memory: {content}")
            elif memory_type == "knowledge":
                formatted.append(f"Knowledge: {content}")
            elif memory_type == "conversation":
                formatted.append(f"Previous conversation: {content}")
            else:
                formatted.append(f"{memory_type}: {content}")
        
        return " | ".join(formatted)


# Create global instance
memory_integration = MemoryIntegration()

# Utility functions
def get_relevant_memories(user_input, limit=5):
    """Get memories relevant to the user input"""
    return memory_integration.get_relevant_memories(user_input, limit)

def add_memory(memory_type, content, metadata=None):
    """Add a new memory"""
    return memory_integration.add_memory(memory_type, content, metadata)

def enhance_prompt_with_memories(prompt, user_input):
    """Enhance a prompt with relevant memories"""
    memories = get_relevant_memories(user_input)
    if not memories:
        return prompt
        
    memory_str = memory_integration.format_memories_for_prompt(memories)
    
    # Add memories to prompt
    if memory_str:
        # Find a good place to add context (before user message)
        lines = prompt.split("\n")
        user_idx = -1
        
        # Find the last "User:" line
        for i, line in enumerate(lines):
            if line.startswith("User:"):
                user_idx = i
        
        if user_idx >= 0:
            # Insert memories before the user message
            lines.insert(user_idx, f"Relevant memories: {memory_str}")
            return "\n".join(lines)
        else:
            # Fallback - just append to the end
            return f"{prompt}\nRelevant memories: {memory_str}"
    
    return prompt
