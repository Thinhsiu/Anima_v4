import os
import sys
import json
import time
import datetime
import re
from pathlib import Path
import shutil
import threading

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from vision.vision_processor import analyze_image, save_image_memory, get_memory
from vision.vision_interface import process_image_file, handle_clipboard_image

# Constants
MEMORIES_DIR = os.path.join(parent_dir, "vision", "memories")
TEMP_DIR = os.path.join(parent_dir, "vision", "temp")
UPLOADS_DIR = os.path.join(parent_dir, "vision", "uploads")
TEMP_RETENTION_DAYS = 7  # Number of days to keep temporary files
CLEANUP_INTERVAL = 24 * 60 * 60  # Run cleanup once per day (in seconds)

# Create directories if they don't exist
os.makedirs(MEMORIES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Memory categories
CATEGORIES = {
    "personal": ["family", "friends", "pets", "selfie", "personal"],
    "work": ["document", "spreadsheet", "presentation", "code", "project", "work"],
    "reference": ["receipt", "id", "passport", "license", "certificate", "reference"],
    "creative": ["art", "drawing", "design", "creative"],
    "other": ["other"]
}

class SmartMemoryManager:
    """
    Context-aware file manager for Anima that handles files intelligently
    based on conversation context and content analysis.
    """
    
    def __init__(self):
        self.recent_files = []  # List of recently processed files
        self.last_accessed_memory = None
        self.conversation_context = []  # Store recent conversation for context
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start a background thread to clean up old temporary files"""
        thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        thread.start()
    
    def _cleanup_thread(self):
        """Background thread that periodically cleans up old temporary files"""
        while True:
            try:
                self.cleanup_temp_files()
                # Also clean up uploads directory
                self.cleanup_old_files(UPLOADS_DIR, TEMP_RETENTION_DAYS)
            except Exception as e:
                print(f"Error in cleanup thread: {e}")
            
            # Sleep for the cleanup interval
            time.sleep(CLEANUP_INTERVAL)
    
    def cleanup_temp_files(self):
        """Clean up temporary files older than TEMP_RETENTION_DAYS"""
        return self.cleanup_old_files(TEMP_DIR, TEMP_RETENTION_DAYS)
    
    def cleanup_old_files(self, directory, days_old):
        """Delete files in directory older than specified days"""
        deleted_count = 0
        now = datetime.datetime.now()
        
        for file_path in Path(directory).glob('*'):
            if file_path.is_file():
                file_age = now - datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.days > days_old:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} old files from {directory}")
        
        return deleted_count
    
    def add_conversation_context(self, user_input, ai_response):
        """Add conversation exchange to context"""
        self.conversation_context.append({
            "user": user_input,
            "ai": ai_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Keep only recent context (last 10 exchanges)
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)
    
    def get_file_importance(self, file_path, user_input):
        """
        Determine how important a file seems based on:
        1. User's language around it
        2. Content of the file
        3. Recent conversation context
        
        Returns a tuple of (importance_score, category, is_personal)
        - importance_score: 0-10 (10 being most important)
        - category: Best matching category
        - is_personal: Boolean indicating if it contains personal content
        """
        importance_score = 5  # Default middle importance
        category = "other"
        is_personal = False
        
        # Check user language indicators
        important_phrases = [
            "important", "save this", "keep this", "remember this", 
            "for later", "don't forget", "crucial", "vital", "essential"
        ]
        
        personal_indicators = [
            "my family", "my kids", "my child", "my wife", "my husband", "my partner",
            "my parents", "my mother", "my father", "my daughter", "my son",
            "my wedding", "my graduation", "my birthday", "my pet", "my dog", "my cat"
        ]
        
        # Check user input for importance signals
        for phrase in important_phrases:
            if phrase in user_input.lower():
                importance_score += 2
                break
                
        # Check for personal content
        for indicator in personal_indicators:
            if indicator in user_input.lower():
                is_personal = True
                importance_score += 2
                break
        
        # Limit score to 0-10 range
        importance_score = min(10, max(0, importance_score))
        
        # Determine category
        for cat, keywords in CATEGORIES.items():
            for keyword in keywords:
                if keyword in user_input.lower():
                    category = cat
                    break
            if category != "other":
                break
                
        # If it's a personal photo, categorize as personal
        if is_personal:
            category = "personal"
        
        return (importance_score, category, is_personal)
    
    def should_ask_to_save(self, importance_score, is_personal):
        """Determine if Anima should ask to save this file"""
        # Always ask about personal content
        if is_personal:
            return True
            
        # Ask about important content
        if importance_score >= 7:
            return True
            
        # Don't bother asking about low importance files
        return False
    
    def process_file(self, file_path, user_input, ai_response=None):
        """
        Process a file intelligently based on context
        
        Returns:
            dict containing:
                - analysis: Analysis of the content
                - should_ask: Boolean - should Anima ask to save it?
                - suggested_memory_type: "long_term" or "temporary"
                - importance: Importance score (0-10)
                - category: Suggested category
        """
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Different handling for different file types
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            # It's an image
            analysis = analyze_image(file_path)
            
            # Get importance metrics
            importance, category, is_personal = self.get_file_importance(file_path, user_input)
            
            # Determine if we should ask to save
            should_ask = self.should_ask_to_save(importance, is_personal)
            
            # Suggest memory type based on importance
            suggested_memory_type = "long_term" if importance >= 6 else "temporary"
            
            # Add to recent files
            self.recent_files.append({
                "path": file_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "importance": importance,
                "category": category,
                "is_personal": is_personal
            })
            
            # Keep only recent files in memory
            if len(self.recent_files) > 20:
                self.recent_files.pop(0)
                
            return {
                "analysis": analysis,
                "should_ask": should_ask,
                "suggested_memory_type": suggested_memory_type,
                "importance": importance,
                "category": category,
                "is_personal": is_personal
            }
        else:
            # Other file types - currently not analyzed deeply
            return {
                "analysis": {"analysis": f"This appears to be a {ext[1:]} file."},
                "should_ask": False,
                "suggested_memory_type": "temporary",
                "importance": 3,
                "category": "other",
                "is_personal": False
            }
    
    def save_to_memory(self, file_path, memory_type="temporary", tags=None):
        """
        Save a file to memory (either temporary or long-term)
        
        Args:
            file_path: Path to the file
            memory_type: "temporary" or "long_term"
            tags: List of tags to associate with the memory
            
        Returns:
            memory_id if successful, None otherwise
        """
        if memory_type == "temporary":
            # For temporary storage, just copy to temp directory
            try:
                filename = os.path.basename(file_path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                temp_filename = f"{timestamp}_{filename}"
                temp_path = os.path.join(TEMP_DIR, temp_filename)
                
                shutil.copy2(file_path, temp_path)
                
                # Create a simple JSON metadata file
                metadata = {
                    "original_path": file_path,
                    "temp_path": temp_path,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "expires": (datetime.datetime.now() + datetime.timedelta(days=TEMP_RETENTION_DAYS)).isoformat(),
                    "tags": tags or []
                }
                
                metadata_path = f"{temp_path}.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                return os.path.basename(temp_path)
                
            except Exception as e:
                print(f"Error saving temporary file: {e}")
                return None
        else:
            # For long-term storage, use the save_image_memory function
            try:
                # Add standard tags if not provided
                if tags is None:
                    tags = []
                
                if "long-term" not in tags:
                    tags.append("long-term")
                    
                memory_id = save_image_memory(file_path, tags=tags)
                return memory_id
                
            except Exception as e:
                print(f"Error saving long-term memory: {e}")
                return None
    
    def get_file_by_description(self, user_query):
        """
        Try to find a file based on user's description
        
        Args:
            user_query: User's description of the file they're looking for
            
        Returns:
            dict with file info if found, None otherwise
        """
        # First, check recent files (most likely what user is referring to)
        if self.recent_files:
            # Start with most recent file as default guess
            best_match = self.recent_files[-1]
            
            # Look for specific references
            keywords = ["last", "recent", "previous", "that", "the"]
            if any(kw in user_query.lower() for kw in keywords):
                # They're probably referring to the most recent file
                return best_match
        
        # Then check long term memories
        memories = get_memory(limit=20)
        
        if not memories:
            return None
            
        # TODO: Implement more sophisticated memory search based on user query
        # This would require embedding the query and comparing to memory embeddings
        
        # For now, just return the most recent memory as a fallback
        if memories:
            self.last_accessed_memory = memories[0]["id"]
            return {
                "path": memories[0]["saved_path"],
                "timestamp": memories[0]["timestamp"],
                "category": "unknown"
            }
            
        return None
    
    def detect_memory_reference(self, user_input):
        """
        Detect if user is referring to a previously shared file
        
        Returns:
            tuple of (is_reference, memory_info)
        """
        memory_references = [
            r"show me (that|the) (image|photo|picture|file)",
            r"(find|get|retrieve) (that|the) (image|photo|picture|file)",
            r"(show|display|pull up) (that|the|my) (image|photo|picture|file)",
            r"remember (that|the) (image|photo|picture|file)",
            r"can you show me (that|the) (image|photo|picture)",
        ]
        
        for pattern in memory_references:
            if re.search(pattern, user_input, re.IGNORECASE):
                memory_info = self.get_file_by_description(user_input)
                return (True, memory_info)
                
        return (False, None)
    
    def detect_save_request(self, user_input):
        """
        Detect if user is asking to save a file as a memory
        
        Returns:
            tuple of (is_save_request, memory_type)
        """
        save_patterns = [
            r"(save|keep|store|remember) this (image|photo|picture|file)",
            r"(save|keep|store) this (for|in) (your|the) (memory|long[- ]term memory)",
            r"don't forget this (image|photo|picture|file)",
            r"put this in your (memory|long[- ]term memory)",
            r"remember this (image|photo|picture|file)",
        ]
        
        temp_patterns = [
            r"(save|keep|store) this temporarily",
            r"just (keep|save) this for now",
        ]
        
        for pattern in save_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return (True, "long_term")
                
        for pattern in temp_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return (True, "temporary")
        
        return (False, None)
    
    def get_memory_suggestions(self, user_input, vision_context=None):
        """
        Generate suggestions for how Anima should respond to memory references
        
        Args:
            user_input: User's message
            vision_context: Optional vision context if image was just processed
            
        Returns:
            dict with suggestions for Anima's response
        """
        # First check if this is a reference to a previously shared memory
        is_reference, memory_info = self.detect_memory_reference(user_input)
        
        if is_reference and memory_info:
            return {
                "action": "retrieve",
                "memory_path": memory_info["path"],
                "suggestion": f"I found that file. It was from {memory_info['timestamp'][:10]}."
            }
            
        # Check if this is a save request
        is_save_request, memory_type = self.detect_save_request(user_input)
        
        if is_save_request and self.recent_files:
            most_recent = self.recent_files[-1]
            
            save_id = self.save_to_memory(most_recent["path"], memory_type)
            
            if save_id:
                if memory_type == "long_term":
                    return {
                        "action": "saved_long_term",
                        "memory_id": save_id,
                        "suggestion": "I've saved that in my long-term memory. I'll remember it for you."
                    }
                else:
                    return {
                        "action": "saved_temporary",
                        "temp_id": save_id,
                        "suggestion": f"I've temporarily saved that for you. I'll keep it for {TEMP_RETENTION_DAYS} days."
                    }
            
        # If we just processed an image and it's important, suggest saving it
        if vision_context and "should_ask" in vision_context and vision_context["should_ask"]:
            return {
                "action": "suggest_saving",
                "importance": vision_context["importance"],
                "is_personal": vision_context["is_personal"],
                "suggestion": "Would you like me to save this in my memory?" if vision_context["importance"] < 8 else 
                             "This seems important. Should I save it in my long-term memory for you?"
            }
            
        return {"action": "none"}


# Initialize a global instance
smart_memory_manager = SmartMemoryManager()

if __name__ == "__main__":
    # Simple test
    print("Testing SmartMemoryManager...")
    
    # Test cleanup
    smart_memory_manager.cleanup_temp_files()
    
    print("SmartMemoryManager ready.")
