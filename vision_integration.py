import os
import sys
import re
import json
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from vision.vision_processor import analyze_image, save_image_memory, get_memory
from vision.vision_interface import (
    process_image_file, 
    handle_clipboard_image,
    drag_drop_handler,
    list_recent_memories
)

# Import the smart memory manager
try:
    from vision.smart_memory_manager import smart_memory_manager
    SMART_MEMORY_ENABLED = True
except ImportError:
    print("Smart memory management not available.")
    SMART_MEMORY_ENABLED = False

# Constants
DEFAULT_WATCH_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "Anima")
os.makedirs(DEFAULT_WATCH_DIR, exist_ok=True)

class VisionIntegration:
    """Integration of vision capabilities with Anima's conversation flow"""
    
    def __init__(self):
        self._last_processed_image = None
        self._last_analysis = None
        self._smart_memory_enabled = SMART_MEMORY_ENABLED
    
    def process_message(self, user_input):
        """
        Process user message for vision-related commands and actions
        
        Args:
            user_input (str): User's message
            
        Returns:
            tuple: (modified_input, vision_context, response_override)
                - modified_input: Possibly modified user input
                - vision_context: Additional context from vision analysis
                - response_override: Complete response override if vision handles it
        """
        # Default return values
        modified_input = user_input
        vision_context = None
        response_override = None
        
        # Check if smart memory manager detects a memory reference
        if self._smart_memory_enabled:
            is_reference, memory_info = smart_memory_manager.detect_memory_reference(user_input)
            if is_reference and memory_info:
                response_override = f"I found that file from your memory. It was from {memory_info.get('timestamp', '')[:10] if 'timestamp' in memory_info else 'recently'}."                
                return modified_input, {"memory_reference": memory_info}, response_override
                
            # Check if this is a save request
            is_save_request, memory_type = smart_memory_manager.detect_save_request(user_input)
            if is_save_request and smart_memory_manager.recent_files:
                most_recent = smart_memory_manager.recent_files[-1]
                
                save_id = smart_memory_manager.save_to_memory(most_recent["path"], memory_type)
                
                if save_id:
                    if memory_type == "long_term":
                        response_override = "I've saved that in my long-term memory. I'll remember it for you."
                    else:
                        response_override = f"I've temporarily saved that for you."
                        
                    return modified_input, {"saved_memory": save_id}, response_override
        
        # Check for image path in the message (surrounded by quotes or specific syntax)
        image_path_match = re.search(r'"([^"]+\.(jpg|jpeg|png|gif|bmp|webp))"', user_input, re.IGNORECASE)
        if not image_path_match:
            image_path_match = re.search(r'image:\s*([^\s]+\.(jpg|jpeg|png|gif|bmp|webp))', user_input, re.IGNORECASE)
        
        if image_path_match:
            image_path = image_path_match.group(1)
            
            # Try different path variations if the direct path doesn't exist
            if not os.path.exists(image_path):
                # Try with full path
                if not os.path.isabs(image_path):
                    full_path = os.path.abspath(image_path)
                    if os.path.exists(full_path):
                        image_path = full_path
                
                # Try in default watch directory
                watch_path = os.path.join(DEFAULT_WATCH_DIR, os.path.basename(image_path))
                if os.path.exists(watch_path):
                    image_path = watch_path
            
            if os.path.exists(image_path):
                # Extract any query about the image (text after the image path)
                remaining_text = user_input.split(image_path_match.group(0), 1)[-1].strip()
                query = remaining_text if remaining_text else "Describe this image in detail."
                
                # Process the image with smart memory manager if available
                if self._smart_memory_enabled:
                    smart_result = smart_memory_manager.process_file(image_path, user_input)
                    analysis = smart_result["analysis"]
                    
                    # Store the smart result for additional context
                    vision_context_data = {
                        "analysis": analysis["analysis"],
                        "should_ask": smart_result["should_ask"],
                        "suggested_memory_type": smart_result["suggested_memory_type"],
                        "importance": smart_result["importance"],
                        "category": smart_result["category"],
                        "is_personal": smart_result["is_personal"]
                    }
                else:
                    # Process the image normally
                    analysis = process_image_file(image_path, query)
                    vision_context_data = {"analysis": analysis["analysis"]}
                    
                self._last_processed_image = image_path
                self._last_analysis = analysis
                
                # Create vision context
                vision_context = vision_context_data
                
                # Remove the image path from the input and keep the query
                modified_input = query
        
        # Check for clipboard command
        elif any(cmd in user_input.lower() for cmd in ["check clipboard", "analyze clipboard", "clipboard image"]):
            analysis = handle_clipboard_image()
            
            if "error" in analysis:
                response_override = f"I couldn't find an image in your clipboard: {analysis['error']}"
            else:
                self._last_processed_image = analysis.get("file_path")
                self._last_analysis = analysis
                
                # Process with smart memory if available
                if self._smart_memory_enabled:
                    smart_result = smart_memory_manager.process_file(analysis.get("file_path"), user_input)
                    
                    # Store the smart result for additional context
                    vision_context = {
                        "analysis": analysis["analysis"],
                        "should_ask": smart_result["should_ask"],
                        "suggested_memory_type": smart_result["suggested_memory_type"],
                        "importance": smart_result["importance"],
                        "category": smart_result["category"],
                        "is_personal": smart_result["is_personal"]
                    }
                else:
                    # Create vision context
                    vision_context = {"analysis": analysis["analysis"]}
                
                # Modify input to focus on the image
                modified_input = "What can you tell me about this image from my clipboard?"
        
        # Check for memory listing command
        elif any(cmd in user_input.lower() for cmd in ["show memories", "list memories", "image memories"]):
            memories = list_recent_memories(limit=5)
            
            if not memories:
                response_override = "You don't have any saved image memories yet."
            else:
                memory_list = []
                for i, memory in enumerate(memories, 1):
                    desc = memory.get("analysis", {}).get("analysis", "")
                    if isinstance(desc, dict):
                        desc = desc.get("analysis", "")
                    desc = desc[:100] + "..." if len(desc) > 100 else desc
                    
                    memory_list.append(f"{i}. {os.path.basename(memory.get('original_path', 'Unknown'))}: {desc}")
                
                response_override = "Here are your recent image memories:\n\n" + "\n\n".join(memory_list)
        
        # Continue conversation about the last image
        elif self._last_processed_image and any(phrase in user_input.lower() for phrase in [
            "tell me more about", "what else", "more details", "analyze further", "what's in the image"
        ]):
            if self._last_analysis:
                vision_context = f"[Image Analysis: {self._last_analysis['analysis']}]"
        
        return modified_input, vision_context, response_override
    
    def add_vision_context_to_prompt(self, prompt, vision_context):
        """Add vision context to the prompt for the LLM"""
        if not vision_context:
            return prompt
        
        # Format context based on whether it's a dict or string
        if isinstance(vision_context, dict):
            # Smart memory provides rich context
            if "memory_reference" in vision_context:
                context_str = f"[User is referring to a previously shared file: {os.path.basename(vision_context['memory_reference'].get('path', 'unknown file'))}]"
            elif "saved_memory" in vision_context:
                context_str = f"[User has asked to save the recently shared file with ID: {vision_context['saved_memory']}]"
            elif "analysis" in vision_context:
                # Include analysis and smart memory info if available
                analysis = vision_context["analysis"]
                
                context_str = f"[Image Analysis: {analysis}]"
                
                # Add suggestions for smart memory if available
                if "should_ask" in vision_context and vision_context["should_ask"]:
                    if vision_context.get("is_personal", False):
                        context_str += "\n[This appears to be a personal image. Consider asking if the user wants you to remember it.]"
                    elif vision_context.get("importance", 0) >= 7:
                        context_str += "\n[This image seems important. Consider asking if the user wants you to save it.]"
            else:
                # Generic context
                context_str = f"[Vision Context: {json.dumps(vision_context)}]"
        else:
            # String context (legacy format)
            context_str = vision_context
        
        # Add vision context before the user message
        lines = prompt.split("\n")
        user_idx = -1
        
        # Find the last "User:" line
        for i, line in enumerate(lines):
            if line.startswith("User:"):
                user_idx = i
        
        if user_idx >= 0:
            # Insert vision context before the user message
            lines.insert(user_idx, f"Context: {context_str}")
            return "\n".join(lines)
        else:
            # Fallback - just append to the end
            return f"{prompt}\nContext: {context_str}"
    
    def get_vision_commands_help(self):
        """Get help text for available vision commands"""
        return """
Available vision commands:
1. Share an image: Include the path to an image file in quotes, e.g., "C:/path/to/image.jpg"
2. Ask about clipboard image: "Check clipboard" or "Analyze clipboard image"
3. View saved memories: "Show memories" or "List image memories"
4. Ask follow-up questions about the last analyzed image

To share images easily, save them to your Anima pictures folder:
{}
""".format(DEFAULT_WATCH_DIR)


# Initialize the vision integration
vision_integration = VisionIntegration()

def integrate_vision_with_anima(user_input, full_prompt, ai_response=None):
    """
    Function to integrate with Anima's conversation flow
    
    Args:
        user_input (str): Original user input
        full_prompt (str): Full prompt being sent to the LLM
        ai_response (str, optional): Anima's response (for context tracking)
        
    Returns:
        tuple: (modified_input, modified_prompt, response_override)
    """
    # Process the message for vision-related content
    modified_input, vision_context, response_override = vision_integration.process_message(user_input)
    
    # Update conversation context in smart memory manager
    if SMART_MEMORY_ENABLED and ai_response:
        smart_memory_manager.add_conversation_context(user_input, ai_response)
    
    # Modify the prompt if we have vision context
    modified_prompt = full_prompt
    if vision_context:
        modified_prompt = vision_integration.add_vision_context_to_prompt(full_prompt, vision_context)
    
    return modified_input, modified_prompt, response_override

def get_help():
    """Get help text for vision commands"""
    return vision_integration.get_vision_commands_help()
