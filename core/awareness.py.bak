"""
Core awareness system for Anima - enables human-like understanding and memory

This module provides Anima with a context-aware intelligence layer that connects
all subsystems and enables more natural interaction patterns.
"""

import os
import sys
import json
import datetime
import re
import time
from pathlib import Path
import threading
import importlib
from collections import deque

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Core constants
MEMORY_DIR = os.path.join(parent_dir, "memories")
CONTEXT_FILE = os.path.join(MEMORY_DIR, "context_awareness.json")
USER_PREFS_FILE = os.path.join(MEMORY_DIR, "user_preferences.json")
MAX_CONTEXT_ITEMS = 50
CONTEXT_SAVE_INTERVAL = 300  # seconds

# Create directories if they don't exist
os.makedirs(MEMORY_DIR, exist_ok=True)


class AwarenessSystem:
    """
    Core awareness system for Anima
    
    This system maintains context across conversations, learns user preferences,
    and provides a more human-like memory and decision-making process.
    """
    
    def __init__(self):
        """Initialize the awareness system"""
        self.conversation_history = deque(maxlen=MAX_CONTEXT_ITEMS)
        self.user_preferences = {}
        self.context_memory = {}
        self.session_highlights = []
        self.subsystems = {}
        self.last_save_time = 0
        
        # Load existing data
        self._load_context()
        self._load_user_preferences()
        
        # Start background save thread
        self._start_save_thread()
    
    def _start_save_thread(self):
        """Start background thread to periodically save context"""
        thread = threading.Thread(target=self._background_save, daemon=True)
        thread.start()
    
    def _background_save(self):
        """Background thread that periodically saves context data"""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_save_time > CONTEXT_SAVE_INTERVAL:
                    self._save_context()
                    self.last_save_time = current_time
            except Exception as e:
                print(f"Error in awareness save thread: {e}")
            
            # Sleep for a while
            time.sleep(60)  # Check every minute
    
    def _load_context(self):
        """Load context memory from file"""
        try:
            if os.path.exists(CONTEXT_FILE):
                with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Load conversation history
                    if "conversation_history" in data:
                        self.conversation_history = deque(data["conversation_history"], maxlen=MAX_CONTEXT_ITEMS)
                    
                    # Load other context memory
                    if "context_memory" in data:
                        self.context_memory = data["context_memory"]
        except Exception as e:
            print(f"Error loading context: {e}")
    
    def _save_context(self):
        """Save context memory to file"""
        try:
            data = {
                "conversation_history": list(self.conversation_history),
                "context_memory": self.context_memory,
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            with open(CONTEXT_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving context: {e}")
    
    def _load_user_preferences(self):
        """Load user preferences from file"""
        try:
            if os.path.exists(USER_PREFS_FILE):
                with open(USER_PREFS_FILE, 'r', encoding='utf-8') as f:
                    self.user_preferences = json.load(f)
        except Exception as e:
            print(f"Error loading user preferences: {e}")
    
    def _save_user_preferences(self):
        """Save user preferences to file"""
        try:
            with open(USER_PREFS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving user preferences: {e}")
    
    def add_conversation_exchange(self, user_input, ai_response):
        """Add a conversation exchange to history and analyze for context"""
        # Add to conversation history
        exchange = {
            "user": user_input,
            "ai": ai_response,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.conversation_history.append(exchange)
        
        # Analyze for important context
        self._analyze_for_context(user_input, ai_response)
        
        # Check if we should save context
        current_time = time.time()
        if current_time - self.last_save_time > CONTEXT_SAVE_INTERVAL:
            self._save_context()
            self.last_save_time = current_time
    
    def _analyze_for_context(self, user_input, ai_response):
        """
        Analyze conversation for important context and user preferences
        This is where Anima learns about the user over time
        """
        # Extract user preferences
        self._extract_preferences(user_input, ai_response)
        
        # Check for important topics
        self._identify_important_topics(user_input)
        
        # Save important exchanges
        self._save_notable_exchanges(user_input, ai_response)
    
    def _extract_preferences(self, user_input, ai_response):
        """Extract user preferences from conversation"""
        # Look for preference indicators
        preference_patterns = [
            # Format preference
            (r"i (prefer|like|want) (more|less) (detailed|brief|concise)", "verbosity"),
            # Topic interest
            (r"i('m| am) (interested|curious) about ([a-zA-Z\s]+)", "interests"),
            # Dislike patterns
            (r"i (don't|do not|dislike|hate) ([a-zA-Z\s]+)", "dislikes"),
            # Communication style
            (r"(please|can you) (be more|speak more|talk more) ([a-zA-Z\s]+)", "communication_style")
        ]
        
        for pattern, pref_type in preference_patterns:
            matches = re.search(pattern, user_input.lower())
            if matches:
                # Initialize preference category if needed
                if pref_type not in self.user_preferences:
                    self.user_preferences[pref_type] = []
                
                # Add the preference
                if len(matches.groups()) >= 2:
                    preference = matches.group(2)
                    if preference not in self.user_preferences[pref_type]:
                        self.user_preferences[pref_type].append(preference)
                        self._save_user_preferences()
    
    def _identify_important_topics(self, user_input):
        """Identify important topics from user input"""
        # List of important topic indicators
        important_indicators = [
            "remember", "important", "don't forget", "key", "critical", 
            "essential", "significant", "remember this", "take note"
        ]
        
        # Check if user message contains important indicators
        if any(indicator in user_input.lower() for indicator in important_indicators):
            # Extract what might be important
            # This is a simplified approach - in reality would need more NLP
            important_content = user_input
            
            # Store in context memory
            timestamp = datetime.datetime.now().isoformat()
            if "important_topics" not in self.context_memory:
                self.context_memory["important_topics"] = []
            
            self.context_memory["important_topics"].append({
                "content": important_content,
                "timestamp": timestamp
            })
    
    def _save_notable_exchanges(self, user_input, ai_response):
        """Save exchanges that seem particularly notable"""
        # Emotional indicators (positive or negative)
        emotion_indicators = [
            "love", "hate", "amazing", "terrible", "wonderful", "awful",
            "great", "bad", "happy", "sad", "angry", "excited", "thank you"
        ]
        
        # Check if this exchange seems important emotionally
        if any(indicator in user_input.lower() for indicator in emotion_indicators):
            # Save as a highlight
            self.session_highlights.append({
                "user": user_input,
                "ai": ai_response,
                "timestamp": datetime.datetime.now().isoformat()
            })
    
    def get_relevant_context(self, user_input):
        """
        Get relevant context for the current conversation
        
        Args:
            user_input: Current user input
            
        Returns:
            Context information to enhance Anima's awareness
        """
        context = {}
        
        # Add basic context
        context["user_preferences"] = self._get_relevant_preferences()
        context["conversation_tone"] = self._analyze_conversation_tone()
        
        # Look for topic references
        related_topics = self._find_related_topics(user_input)
        if related_topics:
            context["related_topics"] = related_topics
        
        # Check for references to past conversations
        past_references = self._find_past_references(user_input)
        if past_references:
            context["past_references"] = past_references
        
        return context
    
    def _get_relevant_preferences(self):
        """Get relevant user preferences"""
        # Return a summarized version of preferences
        # In a full implementation, this would be more selective based on current context
        return self.user_preferences
    
    def _analyze_conversation_tone(self):
        """Analyze the tone of recent conversation"""
        # In a full implementation, this would use sentiment analysis
        # For now, just provide a simple analysis based on recent exchanges
        if not self.conversation_history:
            return "neutral"
        
        # Look at last few exchanges
        recent = list(self.conversation_history)[-3:]
        
        # Simple keyword-based tone detection
        positive_indicators = ["thanks", "good", "great", "appreciate", "happy", "love", "like"]
        negative_indicators = ["bad", "not", "don't", "isn't", "problem", "issue", "wrong"]
        
        positive_count = 0
        negative_count = 0
        
        for exchange in recent:
            user_text = exchange.get("user", "").lower()
            for indicator in positive_indicators:
                if indicator in user_text:
                    positive_count += 1
            for indicator in negative_indicators:
                if indicator in user_text:
                    negative_count += 1
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _find_related_topics(self, user_input):
        """Find related topics from past conversations"""
        related_topics = []
        
        # Extract key terms from user input (simplified)
        words = user_input.lower().split()
        key_terms = [w for w in words if len(w) > 4]  # Simple approach - longer words
        
        # Look through history for related exchanges
        for exchange in self.conversation_history:
            past_user = exchange.get("user", "").lower()
            for term in key_terms:
                if term in past_user and exchange not in related_topics:
                    related_topics.append(exchange)
                    break
        
        return related_topics[-3:] if related_topics else []  # Return most recent 3
    
    def _find_past_references(self, user_input):
        """Find references to past conversations"""
        past_reference_patterns = [
            r"(earlier|before|previously) (you|we) (said|mentioned|talked about)",
            r"(remember|recall) when (i|you) (said|mentioned|asked about)",
            r"(as|like) (i|you) (said|mentioned) (earlier|before)"
        ]
        
        for pattern in past_reference_patterns:
            if re.search(pattern, user_input.lower()):
                # Find likely related past exchanges
                return self._find_related_topics(user_input)
        
        return []
    
    def register_subsystem(self, name, subsystem):
        """Register a subsystem with the awareness system"""
        self.subsystems[name] = subsystem
    
    def enhance_prompt(self, prompt, user_input):
        """
        Enhance a prompt with awareness context
        
        Args:
            prompt: Original prompt
            user_input: Current user input
            
        Returns:
            Enhanced prompt with awareness context
        """
        # Get relevant context
        context = self.get_relevant_context(user_input)
        
        # Format context for insertion
        context_str = self._format_context_for_prompt(context)
        
        # Add context to prompt
        if context_str:
            # Find a good place to add context (before user message)
            lines = prompt.split("\n")
            user_idx = -1
            
            # Find the last "User:" line
            for i, line in enumerate(lines):
                if line.startswith("User:"):
                    user_idx = i
            
            if user_idx >= 0:
                # Insert context before the user message
                lines.insert(user_idx, f"Context: {context_str}")
                return "\n".join(lines)
            else:
                # Fallback - just append to the end
                return f"{prompt}\nContext: {context_str}"
        
        return prompt
    
    def _format_context_for_prompt(self, context):
        """Format context data for insertion into prompt"""
        context_parts = []
        
        # Add user preferences if available
        if "user_preferences" in context and context["user_preferences"]:
            prefs = context["user_preferences"]
            pref_summary = []
            
            for pref_type, values in prefs.items():
                if values:
                    pref_summary.append(f"{pref_type}: {', '.join(values[:3])}")
            
            if pref_summary:
                context_parts.append(f"User preferences: {'; '.join(pref_summary)}")
        
        # Add conversation tone
        if "conversation_tone" in context:
            context_parts.append(f"Conversation tone: {context['conversation_tone']}")
        
        # Add related topics summary if available
        if "related_topics" in context and context["related_topics"]:
            topics = context["related_topics"]
            topic_summary = []
            
            for topic in topics:
                # Format timestamp to be more readable
                ts = topic.get("timestamp", "")
                if ts:
                    try:
                        dt = datetime.datetime.fromisoformat(ts)
                        time_str = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        time_str = ts
                else:
                    time_str = "unknown time"
                
                topic_summary.append(f"Previous exchange ({time_str}): User: \"{topic.get('user', '')}\" Anima: \"{topic.get('ai', '')}\"")
            
            if topic_summary:
                context_parts.append("Related topics from conversation history: " + " | ".join(topic_summary))
        
        # Add past references if available
        if "past_references" in context and context["past_references"]:
            context_parts.append("User appears to be referencing a previous conversation")
        
        return " | ".join(context_parts)


# Create global instance
awareness = AwarenessSystem()

# Utility functions
def enhance_prompt(prompt, user_input):
    """Enhance a prompt with awareness context"""
    return awareness.enhance_prompt(prompt, user_input)

def add_conversation(user_input, ai_response):
    """Add a conversation exchange to the awareness system"""
    awareness.add_conversation_exchange(user_input, ai_response)

def get_awareness_context(user_input):
    """Get awareness context for the current user input"""
    return awareness.get_relevant_context(user_input)
