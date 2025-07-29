"""
File Memory Integration for Anima

This module handles integration between the file sharing UI and Anima's memory system,
allowing shared files to be stored, recalled, and discussed as part of Anima's memory.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anima.memory.file")

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent))

# File storage directory (relative to project root)
FILE_STORAGE_DIR = os.path.join(Path(__file__).parent.parent, "shared_files")

# Ensure file storage directory exists
os.makedirs(FILE_STORAGE_DIR, exist_ok=True)


class FileMemoryManager:
    """Manager for file-related memories in Anima"""
    
    def __init__(self):
        self.memory_file = os.path.join(FILE_STORAGE_DIR, "file_memories.json")
        self.memories = self._load_memories()
        
    def _load_memories(self) -> List[Dict[str, Any]]:
        """Load file memories from storage"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading file memories: {e}")
        return []
        
    def _save_memories(self):
        """Save memories to storage"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving file memories: {e}")
            
    def add_file_memory(self, 
                        file_path: str, 
                        file_name: str, 
                        file_type: str,
                        description: str = "", 
                        tags: List[str] = None) -> Dict[str, Any]:
        """
        Add a file to Anima's memory
        
        Args:
            file_path: Path to the stored file
            file_name: Original filename
            file_type: Type of file (image, document, other)
            description: User-provided description of the file
            tags: Optional list of tags for categorization
            
        Returns:
            Dictionary with memory metadata
        """
        # Create memory entry
        memory = {
            "id": f"file_{len(self.memories) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "type": "file",
            "file_path": file_path,
            "file_name": file_name,
            "file_type": file_type,
            "description": description,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        
        # Add to memories
        self.memories.append(memory)
        self._save_memories()
        
        logger.info(f"Added file to memory: {file_name}")
        return memory
        
    def get_file_memories(self, 
                         file_type: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve file memories with optional filtering
        
        Args:
            file_type: Optional filter by file type
            tags: Optional filter by tags
            limit: Maximum number of memories to return
            
        Returns:
            List of matching memories
        """
        results = self.memories
        
        # Filter by file type if specified
        if file_type:
            results = [m for m in results if m["file_type"] == file_type]
            
        # Filter by tags if specified
        if tags:
            results = [m for m in results if any(tag in m["tags"] for tag in tags)]
            
        # Sort by last accessed (most recent first)
        results = sorted(results, key=lambda m: m["last_accessed"], reverse=True)
        
        return results[:limit]
        
    def update_file_access(self, memory_id: str):
        """
        Update the last access time for a file memory
        
        Args:
            memory_id: ID of the memory to update
        """
        for memory in self.memories:
            if memory["id"] == memory_id:
                memory["last_accessed"] = datetime.now().isoformat()
                self._save_memories()
                break
                
    def add_tags_to_file(self, memory_id: str, new_tags: List[str]):
        """
        Add tags to a file memory
        
        Args:
            memory_id: ID of the memory to update
            new_tags: List of tags to add
        """
        for memory in self.memories:
            if memory["id"] == memory_id:
                # Add new tags (avoid duplicates)
                memory["tags"] = list(set(memory["tags"] + new_tags))
                self._save_memories()
                break
                
    def remove_file(self, memory_id: str) -> bool:
        """
        Remove a file memory from storage
        
        Args:
            memory_id: ID of the memory to remove
            
        Returns:
            True if successful, False otherwise
        """
        initial_count = len(self.memories)
        
        # Find and remove the memory with the given ID
        for i, memory in enumerate(self.memories):
            if memory["id"] == memory_id:
                try:
                    # Get file path
                    file_path = memory["file_path"]
                    
                    # Remove the actual file if it exists
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted file: {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete file {file_path}: {e}")
                    
                    # Remove from memories list
                    self.memories.pop(i)
                    self._save_memories()
                    
                    logger.info(f"Removed file memory with ID: {memory_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error removing file memory: {e}")
                    return False
        
        # Memory not found
        if len(self.memories) == initial_count:
            logger.warning(f"File memory not found: {memory_id}")
            
        return False


# Global instance
file_memory_manager = FileMemoryManager()

# Utility functions for integration with main Anima memory system

def register_file_with_memory(file_path: str, 
                            file_name: str, 
                            file_type: str,
                            description: str = "",
                            tags: List[str] = None) -> Dict[str, Any]:
    """
    Register a file with Anima's memory system
    
    Args:
        file_path: Path to the stored file
        file_name: Original filename
        file_type: Type of file (image, document, other)
        description: User-provided description of the file
        tags: Optional list of tags for categorization
        
    Returns:
        Memory metadata
    """
    # Add to file memory manager
    memory = file_memory_manager.add_file_memory(
        file_path=file_path,
        file_name=file_name,
        file_type=file_type,
        description=description,
        tags=tags
    )
    
    # Try to integrate with main memory system if available
    try:
        from memory import memory_manager
        
        # Create a memory entry suitable for the main memory system
        memory_entry = {
            "type": "shared_file",
            "file_path": file_path,
            "file_name": file_name,
            "file_type": file_type,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add tags if provided
        if tags:
            memory_entry["tags"] = tags
            
        # Add to main memory
        memory_manager.add_memory(memory_entry)
        
        logger.info(f"File '{file_name}' integrated with main memory system")
    except ImportError:
        logger.info("Main memory system not available, using file-only memory")
        
    return memory

def get_recent_file_memories(limit: int = 5) -> List[Dict[str, Any]]:
    """Get most recently accessed file memories"""
    return file_memory_manager.get_file_memories(limit=limit)

def get_file_memories_by_type(file_type: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get file memories of a specific type"""
    return file_memory_manager.get_file_memories(file_type=file_type, limit=limit)

def get_file_memories_by_tags(tags: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    """Get file memories with specified tags"""
    return file_memory_manager.get_file_memories(tags=tags, limit=limit)
