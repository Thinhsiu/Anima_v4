"""
Integration module for connecting the file sharing UI with the main Anima application.
"""

import os
import sys
import threading
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anima.file_sharing_integration")

# Track if UI is already running
_file_ui_running = False

def handle_file_command(command_text):
    """
    Parse and handle file-related commands
    
    Returns:
        True if command was handled, False otherwise
    """
    command = command_text.lower().strip()
    
    # Handle file sharing commands
    if command in ("show files", "open files", "file sharing", "share files", "file ui"):
        launch_file_ui()
        return True
        
    # Handle file memory commands
    if command.startswith("remember file") or command.startswith("recall file"):
        # Get file query if any
        query = command.split(" ", 2)[2] if len(command.split(" ")) > 2 else ""
        recall_files(query)
        return True
        
    return False

def launch_file_ui():
    """Launch the file sharing UI in a separate thread"""
    global _file_ui_running
    
    if _file_ui_running:
        logger.info("File sharing UI is already running")
        return "File sharing UI is already open"
    
    def run_ui():
        global _file_ui_running
        _file_ui_running = True
        
        try:
            # Import here to avoid circular imports
            from ui.file_sharing_ui import launch_file_sharing_ui
            launch_file_sharing_ui()
        except Exception as e:
            logger.error(f"Error launching file UI: {e}")
        finally:
            _file_ui_running = False
    
    # Start UI in a separate thread
    logger.info("Launching file sharing UI")
    ui_thread = threading.Thread(target=run_ui, daemon=False)
    ui_thread.start()
    
    return "Opening file sharing interface. You can add, view, and share files with me through this window."

def recall_files(query=None):
    """
    Recall files from memory based on optional query
    
    Args:
        query: Optional search terms
    """
    try:
        # Import file memory system
        from memory.file_memory import get_recent_file_memories, get_file_memories_by_tags
        
        if not query:
            # Get recent files
            memories = get_recent_file_memories(5)
            return format_file_memories(memories, "Recent files")
        else:
            # Treat query terms as tags
            tags = [tag.strip() for tag in query.split(",")]
            memories = get_file_memories_by_tags(tags, 5)
            return format_file_memories(memories, f"Files related to {query}")
    except ImportError:
        logger.error("File memory system not available")
        return "File memory system is not available"
    except Exception as e:
        logger.error(f"Error recalling files: {e}")
        return "Sorry, I had trouble retrieving file memories"

def format_file_memories(memories, title):
    """Format file memories for display in conversation"""
    if not memories:
        return f"{title}: No files found"
        
    result = f"{title}:\n"
    for i, mem in enumerate(memories, 1):
        file_type_icon = "📄" if mem["file_type"] == "document" else "🖼️" if mem["file_type"] == "image" else "📁"
        result += f"{i}. {file_type_icon} {mem['file_name']}"
        if mem.get("description"):
            result += f" - {mem['description']}"
        result += "\n"
        
    return result

def register_with_anima(command_parser):
    """
    Register file commands with Anima's command parser
    
    Args:
        command_parser: Anima's command parsing function
    """
    try:
        # Register our handler with Anima
        if hasattr(command_parser, "register_handler"):
            command_parser.register_handler("file", handle_file_command)
            logger.info("File sharing commands registered with Anima")
    except Exception as e:
        logger.error(f"Failed to register with command parser: {e}")

# Auto-registration attempt
try:
    from commands import command_parser
    register_with_anima(command_parser)
except ImportError:
    logger.warning("Couldn't auto-register file commands, manual integration required")
