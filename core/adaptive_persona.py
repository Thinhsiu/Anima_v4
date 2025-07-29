"""
Adaptive Persona System for Anima

This module allows Anima to subtly adjust her communication style based on
user preferences without being explicitly told. It analyzes user interactions
and feedback to continuously refine Anima's persona.
"""

import os
import sys
import json
import time
import threading
import datetime
from pathlib import Path
import re
from collections import deque

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import the main awareness system
try:
    from core.awareness import awareness
    from core.emotion_recognition import detect_emotions
except ImportError:
    print("Warning: Core awareness module not found. Some features may be limited.")


class AdaptivePersona:
    """
    Adaptive persona system that subtly adjusts communication style
    based on user interactions and feedback.
    """
    
    def __init__(self):
        """Initialize the adaptive persona system"""
        self.memory_path = Path(parent_dir) / "memories" / "adaptive_persona"
        self.memory_path.mkdir(exist_ok=True, parents=True)
        
        self.persona_model = self._load_persona_model()
        self.interaction_history = deque(maxlen=100)  # Store recent interactions
        
        # Communication style dimensions
        self.dimensions = {
            "formal_casual": 0.5,  # 0 = very formal, 1 = very casual
            "concise_detailed": 0.5,  # 0 = very concise, 1 = very detailed
            "serious_lighthearted": 0.5,  # 0 = serious, 1 = lighthearted
            "technical_simplified": 0.5,  # 0 = technical, 1 = simplified
            "reserved_expressive": 0.5,  # 0 = reserved, 1 = expressive
        }
        
        # Update dimensions from model
        if "dimensions" in self.persona_model:
            self.dimensions.update(self.persona_model["dimensions"])
        
        # Response patterns for different dimensions
        self.response_patterns = self._define_response_patterns()
        
        # Start background thread
        self.update_thread = threading.Thread(target=self._periodic_update, daemon=True)
        self.update_thread.start()
        
        # Time tracking
        self.last_save_time = time.time()
    
    def _load_persona_model(self):
        """Load the existing persona model"""
        model_path = self.memory_path / "persona_model.json"
        if model_path.exists():
            try:
                with open(model_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_default_model()
        else:
            return self._create_default_model()
    
    def _create_default_model(self):
        """Create a default persona model"""
        return {
            "dimensions": {
                "formal_casual": 0.5,
                "concise_detailed": 0.5,
                "serious_lighthearted": 0.5,
                "technical_simplified": 0.5,
                "reserved_expressive": 0.5
            },
            "learned_preferences": {},  # Topics/contexts -> preferred style
            "avoided_phrases": [],  # Phrases that got negative reactions
            "preferred_phrases": [],  # Phrases that got positive reactions
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def _save_persona_model(self):
        """Save the persona model to disk"""
        # Update dimensions in model
        self.persona_model["dimensions"] = self.dimensions
        self.persona_model["last_updated"] = datetime.datetime.now().isoformat()
        
        model_path = self.memory_path / "persona_model.json"
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(self.persona_model, f, indent=2)
    
    def _periodic_update(self):
        """Periodically update the persona model"""
        while True:
            time.sleep(300)  # Every 5 minutes
            try:
                if time.time() - self.last_save_time > 300:  # Only if likely changed
                    self._analyze_recent_interactions()
                    self._save_persona_model()
                    self.last_save_time = time.time()
            except Exception as e:
                print(f"Error updating persona model: {e}")
    
    def _define_response_patterns(self):
        """Define response patterns for different communication styles"""
        return {
            "formal_casual": {
                "low": {  # Formal patterns
                    "greetings": ["Good morning", "Good afternoon", "Good evening", "Greetings"],
                    "closings": ["Best regards", "Sincerely", "Respectfully"],
                    "phrases": ["I would like to", "It would be appropriate to", "I would recommend"],
                    "avoid": ["yeah", "nah", "cool", "awesome", "hey there"]
                },
                "high": {  # Casual patterns
                    "greetings": ["Hi", "Hey", "Hello there", "Hey there"],
                    "closings": ["Cheers", "Talk soon", "Take care"],
                    "phrases": ["Let's", "I think we should", "How about we"],
                    "avoid": ["pursuant to", "henceforth", "in accordance with"]
                }
            },
            "concise_detailed": {
                "low": {  # Concise patterns
                    "sentences": ["short", "direct"],
                    "structure": ["bullet points", "numbered lists"],
                    "avoid": ["to elaborate further", "to provide more context", "in addition"]
                },
                "high": {  # Detailed patterns
                    "sentences": ["detailed", "thorough", "comprehensive"],
                    "structure": ["paragraphs", "sections"],
                    "phrases": ["Let me explain further", "For additional context", "To provide more detail"]
                }
            },
            "serious_lighthearted": {
                "low": {  # Serious patterns
                    "tone": ["serious", "formal", "professional"],
                    "avoid": ["joke", "fun", "exciting", "wow", "cool", "awesome"]
                },
                "high": {  # Lighthearted patterns
                    "tone": ["lighthearted", "friendly", "warm"],
                    "phrases": ["That's great!", "How fun!", "Isn't that cool?"],
                    "include": ["analogies", "metaphors", "casual examples"]
                }
            },
            "technical_simplified": {
                "low": {  # Technical patterns
                    "vocabulary": ["technical", "specialized", "precise"],
                    "structure": ["technical details first", "deep dive"],
                    "phrases": ["Technically speaking", "From a technical perspective"]
                },
                "high": {  # Simplified patterns
                    "vocabulary": ["simple", "everyday", "common"],
                    "structure": ["simple explanations first", "analogies"],
                    "phrases": ["In simple terms", "Think of it like", "It's similar to"]
                }
            },
            "reserved_expressive": {
                "low": {  # Reserved patterns
                    "punctuation": ["minimal", "restrained"],
                    "emotions": ["subtle", "implied"],
                    "avoid": ["!", "!!", "??", "CAPS", "very", "extremely"]
                },
                "high": {  # Expressive patterns
                    "punctuation": ["expressive", "varied"],
                    "emotions": ["explicit", "emphasized"],
                    "phrases": ["I'm excited about", "That's amazing!", "How wonderful!"]
                }
            }
        }
    
    def _detect_user_preferences(self, user_message, assistant_message, user_reaction=None):
        """
        Detect user style preferences from their message and reaction
        Returns dictionary of dimension adjustments
        """
        adjustments = {}
        message_lower = user_message.lower()
        
        # Detect formality preference
        if any(word in message_lower for word in ["casual", "informal", "relaxed", "chill"]):
            adjustments["formal_casual"] = 0.05  # Nudge toward casual
        elif any(word in message_lower for word in ["formal", "professional", "proper"]):
            adjustments["formal_casual"] = -0.05  # Nudge toward formal
            
        # Detect detail preference
        if any(phrase in message_lower for phrase in ["more detail", "elaborate", "tell me more", "explain more"]):
            adjustments["concise_detailed"] = 0.05  # Nudge toward detailed
        elif any(phrase in message_lower for phrase in ["be brief", "keep it short", "summarize", "too long"]):
            adjustments["concise_detailed"] = -0.05  # Nudge toward concise
            
        # Detect tone preference
        if any(word in message_lower for word in ["serious", "focus", "professional"]):
            adjustments["serious_lighthearted"] = -0.05  # Nudge toward serious
        elif any(word in message_lower for word in ["joke", "funny", "lighten up", "relax"]):
            adjustments["serious_lighthearted"] = 0.05  # Nudge toward lighthearted
            
        # Detect technical level preference
        if any(phrase in message_lower for phrase in ["technical", "advanced", "in depth", "expert"]):
            adjustments["technical_simplified"] = -0.05  # Nudge toward technical
        elif any(phrase in message_lower for phrase in ["simple", "explain simply", "basic", "beginner"]):
            adjustments["technical_simplified"] = 0.05  # Nudge toward simplified
            
        # Detect expressiveness preference
        if any(word in message_lower for word in ["enthusiastic", "expressive", "excited"]):
            adjustments["reserved_expressive"] = 0.05  # Nudge toward expressive
        elif any(word in message_lower for word in ["calm", "reserved", "subdued", "quiet"]):
            adjustments["reserved_expressive"] = -0.05  # Nudge toward reserved
            
        # If user reaction provided, use it for stronger adjustments
        if user_reaction:
            reaction_lower = user_reaction.lower()
            
            # Positive reaction - reinforce current style
            if any(word in reaction_lower for word in ["good", "great", "perfect", "thanks", "helpful", "like"]):
                # Strengthen all current dimensions (smaller adjustment)
                for dim in self.dimensions:
                    if dim not in adjustments:
                        # If dimension is above 0.5, nudge higher, if below, nudge lower
                        if self.dimensions[dim] > 0.5:
                            adjustments[dim] = 0.02  # Small reinforcement
                        elif self.dimensions[dim] < 0.5:
                            adjustments[dim] = -0.02  # Small reinforcement
                            
            # Negative reaction - adjust away from current style
            elif any(word in reaction_lower for word in ["bad", "not helpful", "don't like", "change", "differently"]):
                # Move dimensions toward middle (0.5) as a reset
                for dim in self.dimensions:
                    if dim not in adjustments:
                        # Move toward 0.5
                        if self.dimensions[dim] > 0.5:
                            adjustments[dim] = -0.05
                        elif self.dimensions[dim] < 0.5:
                            adjustments[dim] = 0.05
        
        return adjustments
    
    def _extract_context(self, message):
        """Extract context/topic from message for context-specific adaptation"""
        # Simple topic extraction - a real system would use NLP
        topics = []
        
        # Check for explicit topic indicators
        topic_matches = re.findall(r'(?:about|regarding|concerning|on the topic of)\s+([a-zA-Z0-9\s]+)', message.lower())
        if topic_matches:
            topics.extend([match.strip() for match in topic_matches])
            
        # Try to get topics from awareness if available
        try:
            if hasattr(awareness, "context_memory"):
                recent_exchange = awareness.context_memory.get_recent_exchanges(1)
                if recent_exchange and "topics" in recent_exchange[0]:
                    topics.extend(recent_exchange[0]["topics"])
        except Exception:
            pass
            
        return list(set(topics))  # Remove duplicates
    
    def _analyze_recent_interactions(self):
        """Analyze recent interactions to refine the persona model"""
        if not self.interaction_history:
            return
            
        # Count context-specific preferences
        context_preferences = {}
        
        for interaction in self.interaction_history:
            context = interaction.get("context", [])
            adjustments = interaction.get("adjustments", {})
            
            # Only process if we have both context and adjustments
            if context and adjustments:
                for topic in context:
                    if topic not in context_preferences:
                        context_preferences[topic] = {dim: [] for dim in self.dimensions}
                        
                    # Add adjustments to respective dimensions
                    for dim, value in adjustments.items():
                        context_preferences[topic][dim].append(value)
        
        # Calculate average preferences for each context
        learned_preferences = {}
        
        for topic, dimensions in context_preferences.items():
            learned_preferences[topic] = {}
            
            for dim, values in dimensions.items():
                if values:  # Only if we have data
                    # Calculate average adjustment
                    avg_adjustment = sum(values) / len(values)
                    learned_preferences[topic][dim] = avg_adjustment
        
        # Update the model
        self.persona_model["learned_preferences"] = learned_preferences
    
    def update_persona(self, user_message, assistant_message, user_reaction=None):
        """
        Update the persona model based on user interaction
        Returns the updated dimensions
        """
        if not user_message:
            return self.dimensions.copy()
            
        # Detect emotional content
        try:
            emotions = detect_emotions(user_message)
        except Exception:
            emotions = {}
            
        # Extract context/topic
        context = self._extract_context(user_message)
        
        # Get basic adjustments from message
        adjustments = self._detect_user_preferences(user_message, assistant_message, user_reaction)
        
        # Apply context-specific adjustments if available
        if context:
            for topic in context:
                if topic in self.persona_model["learned_preferences"]:
                    topic_prefs = self.persona_model["learned_preferences"][topic]
                    
                    for dim, value in topic_prefs.items():
                        # Context-specific adjustments have lower weight than direct feedback
                        if dim in adjustments:
                            adjustments[dim] = (adjustments[dim] * 0.8) + (value * 0.2)
                        else:
                            adjustments[dim] = value * 0.5  # Half weight for context-only
        
        # Apply adjustments to dimensions
        for dim, adjustment in adjustments.items():
            # Apply adjustment, keeping within 0-1 range
            self.dimensions[dim] = max(0.0, min(1.0, self.dimensions[dim] + adjustment))
        
        # Record interaction for future analysis
        self.interaction_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "context": context,
            "emotions": emotions,
            "adjustments": adjustments
        })
        
        self.last_save_time = time.time()
        return self.dimensions.copy()
    
    def get_style_guidance(self):
        """
        Get guidance on communication style based on current dimensions
        Returns dict with style recommendations
        """
        guidance = {}
        
        # Process each dimension
        for dim, value in self.dimensions.items():
            guidance[dim] = {}
            
            # Determine if we're in the low or high end
            if value < 0.4:
                end = "low"
                strength = 1.0 - (value / 0.4)  # 0 to 1 scale
            elif value > 0.6:
                end = "high"
                strength = (value - 0.6) / 0.4  # 0 to 1 scale
            else:
                # In the middle, provide balanced guidance
                guidance[dim]["balanced"] = True
                continue
                
            # Get patterns for this dimension and end
            patterns = self.response_patterns.get(dim, {}).get(end, {})
            
            # Add to guidance with strength
            guidance[dim]["end"] = end
            guidance[dim]["strength"] = strength
            guidance[dim]["patterns"] = patterns
            
        return guidance
    
    def format_style_guidance(self, guidance):
        """Format style guidance for inclusion in prompts"""
        if not guidance:
            return ""
            
        formatted = "\n\nCommunication style preferences:\n"
        
        # Format each dimension
        for dim, info in guidance.items():
            if "balanced" in info and info["balanced"]:
                continue  # Skip balanced dimensions
                
            end = info.get("end")
            strength = info.get("strength", 0)
            
            # Only include strong preferences
            if strength < 0.5:
                continue
                
            # Format dimension name for readability
            readable_dim = dim.replace("_", " to ")
            
            # Indicate which end is preferred
            if end == "low":
                formatted += f"- Prefer {readable_dim.split(' to ')[0]} style"
            else:
                formatted += f"- Prefer {readable_dim.split(' to ')[1]} style"
                
            # Add specific guidance for strong preferences
            if strength > 0.7:
                patterns = info.get("patterns", {})
                
                if "phrases" in patterns:
                    phrases = patterns["phrases"]
                    if phrases:
                        formatted += f"\n  Consider using phrases like: {', '.join(phrases[:2])}"
                        
                if "avoid" in patterns:
                    avoid = patterns["avoid"]
                    if avoid:
                        formatted += f"\n  Consider avoiding: {', '.join(avoid[:2])}"
                        
            formatted += "\n"
            
        return formatted
    
    def enhance_prompt_with_style(self, prompt):
        """Enhance a prompt with style guidance"""
        # Get current style guidance
        guidance = self.get_style_guidance()
        style_guidance = self.format_style_guidance(guidance)
        
        if not style_guidance:
            return prompt
            
        # Add to prompt in a location that won't interfere with system instructions
        split_prompt = prompt.split("\n\n")
        if len(split_prompt) > 1:
            # Insert after system instructions but before user input
            return "\n\n".join(split_prompt[:-1]) + style_guidance + "\n\n" + split_prompt[-1]
        else:
            return prompt + style_guidance
    
    def adjust_response(self, response):
        """
        Adjust a response based on the current persona dimensions
        Returns modified response
        """
        # Get current style guidance
        guidance = self.get_style_guidance()
        
        # For now, implement simple adjustments:
        
        # 1. Formal vs Casual adjustment
        formal_casual = self.dimensions.get("formal_casual", 0.5)
        if formal_casual < 0.3:  # Very formal
            # Replace casual greetings
            for casual in ["hey", "hi there", "what's up"]:
                response = re.sub(r'^' + casual, "Hello", response, flags=re.IGNORECASE)
        elif formal_casual > 0.7:  # Very casual
            # Replace formal greetings
            for formal in ["greetings", "good day"]:
                response = re.sub(r'^' + formal, "Hey there", response, flags=re.IGNORECASE)
        
        # 2. Concise vs Detailed adjustment
        concise_detailed = self.dimensions.get("concise_detailed", 0.5)
        if concise_detailed < 0.3:  # Very concise
            # This would require more complex processing to shorten responses
            # Simple approximation: trim very long paragraphs
            paragraphs = response.split("\n\n")
            if len(paragraphs) > 3:
                response = "\n\n".join(paragraphs[:3])
        
        # 3. Reserved vs Expressive adjustment
        reserved_expressive = self.dimensions.get("reserved_expressive", 0.5)
        if reserved_expressive < 0.3:  # Very reserved
            # Reduce exclamation marks
            response = response.replace("!!", ".")
            response = response.replace("!", ".")
        elif reserved_expressive > 0.7:  # Very expressive
            # Add more expressiveness - though this should be done carefully
            pass
            
        return response


# Create global instance
adaptive_persona = AdaptivePersona()

# Exposed functions
def update_persona(user_message, assistant_message, user_reaction=None):
    """Update the persona based on interaction"""
    return adaptive_persona.update_persona(user_message, assistant_message, user_reaction)

def get_style_guidance():
    """Get current style guidance"""
    return adaptive_persona.get_style_guidance()

def enhance_prompt_with_style(prompt):
    """Enhance prompt with style guidance"""
    return adaptive_persona.enhance_prompt_with_style(prompt)

def adjust_response(response):
    """Adjust response based on current persona"""
    return adaptive_persona.adjust_response(response)
