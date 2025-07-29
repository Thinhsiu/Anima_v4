"""
Emotion Response Generator - Creates emotion-aware responses
Adjusts tone and content based on emotional context
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import emotion components
try:
    from emotion.emotion_core import get_instance as get_emotion_detector
    from emotion.emotion_memory import get_instance as get_emotion_memory
    EMOTION_MODULES_AVAILABLE = True
except ImportError:
    logger.warning("Emotion modules not available for response generation")
    EMOTION_MODULES_AVAILABLE = False

class EmotionResponseGenerator:
    """
    Creates emotion-aware responses by adjusting tone and content
    based on detected emotions
    """
    
    def __init__(self):
        """Initialize the emotion response generator"""
        # Load response templates
        self.response_templates = self._load_response_templates()
        
        # Statistics
        self.stats = {
            "responses_generated": 0,
            "empathetic_responses": 0,
            "emotion_adjustments": 0
        }
        
        # Get emotion modules if available
        self.emotion_detector = get_emotion_detector() if EMOTION_MODULES_AVAILABLE else None
        self.emotion_memory = get_emotion_memory() if EMOTION_MODULES_AVAILABLE else None
        
        logger.info("Emotion response generator initialized")
    
    def enhance_response(self, 
                        response: str, 
                        user_input: str,
                        profile_id: str = "default") -> str:
        """
        Enhance a response with emotional awareness
        
        Args:
            response: Original response text
            user_input: User input that triggered the response
            profile_id: Emotional profile ID
            
        Returns:
            Enhanced response text
        """
        # Update statistics
        self.stats["responses_generated"] += 1
        
        # If emotion modules not available, return original
        if not EMOTION_MODULES_AVAILABLE or not self.emotion_detector or not self.emotion_memory:
            return response
            
        try:
            # Analyze user input emotions
            user_emotions = self.emotion_detector.detect_emotions(user_input)
            user_emotion = user_emotions.get("dominant_emotion", "neutral")
            user_intensity = user_emotions.get("dominant_intensity", "low")
            
            # Check for emotional shift
            shift = self.emotion_memory.detect_emotional_shift(
                user_input, 
                profile_id=profile_id
            )
            
            # Get emotional profile
            profile = self.emotion_memory.get_emotion_profile(profile_id)
            
            # Decide on response strategy
            strategy = self._select_response_strategy(
                user_emotion, 
                user_intensity, 
                shift.get("shift_detected", False),
                profile
            )
            
            # Apply response strategy
            enhanced = self._apply_response_strategy(
                response, 
                strategy, 
                user_emotion, 
                user_intensity
            )
            
            # Only count if we actually changed something
            if enhanced != response:
                self.stats["emotion_adjustments"] += 1
                
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing response with emotion: {e}")
            return response
    
    def generate_empathetic_response(self, 
                                   user_input: str,
                                   profile_id: str = "default") -> Optional[str]:
        """
        Generate a standalone empathetic response based on user input
        
        Args:
            user_input: User input text
            profile_id: Emotional profile ID
            
        Returns:
            Empathetic response or None if not applicable
        """
        # If emotion modules not available, return None
        if not EMOTION_MODULES_AVAILABLE or not self.emotion_detector:
            return None
            
        try:
            # Analyze user input emotions
            user_emotions = self.emotion_detector.detect_emotions(user_input)
            user_emotion = user_emotions.get("dominant_emotion", "neutral")
            user_intensity = user_emotions.get("dominant_intensity", "low")
            
            # Only generate for strong non-neutral emotions
            if user_emotion == "neutral" or user_intensity == "low":
                return None
                
            # Get emotional profile for context
            profile = self.emotion_memory.get_emotion_profile(profile_id) if self.emotion_memory else None
            
            # Select appropriate empathetic response template
            response = self._select_empathetic_response(user_emotion, user_intensity)
            
            if response:
                # Update statistics
                self.stats["empathetic_responses"] += 1
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating empathetic response: {e}")
            return None
    
    def enhance_prompt(self, 
                     prompt: str, 
                     user_input: str,
                     profile_id: str = "default") -> str:
        """
        Enhance an LLM prompt with emotional context
        
        Args:
            prompt: Original prompt
            user_input: Recent user input for context
            profile_id: Emotional profile ID
            
        Returns:
            Enhanced prompt
        """
        # If emotion modules not available, return original
        if not EMOTION_MODULES_AVAILABLE or not self.emotion_detector or not self.emotion_memory:
            return prompt
            
        try:
            # Analyze user input emotions
            user_emotions = self.emotion_detector.detect_emotions(user_input)
            
            # Get emotional profile
            profile = self.emotion_memory.get_emotion_profile(profile_id)
            
            # Get emotional history summary
            history = self.emotion_memory.summarize_emotional_history(profile_id)
            
            # Create emotion context section
            emotion_context = (
                "\n\nEMOTIONAL CONTEXT:\n"
                f"- Current emotional state: {user_emotions.get('dominant_emotion', 'neutral')} "
                f"(intensity: {user_emotions.get('dominant_intensity', 'low')})\n"
            )
            
            # Add emotional history if available
            if history.get("records", 0) > 0:
                emotion_context += (
                    f"- Dominant emotional pattern: {history.get('dominant_emotion', 'neutral')}\n"
                    f"- Emotional stability: {history.get('emotional_stability', 1.0):.1f}\n"
                    f"- Recent emotional trend: {history.get('trend', 'stable')}\n"
                )
                
                # Add emotional range if diverse
                if len(history.get("emotional_range", [])) > 1:
                    emotion_context += (
                        f"- Emotional range: {', '.join(history.get('emotional_range', []))}\n"
                    )
            
            # Add emotional response guidance
            emotion_context += "\nConsider this emotional context when generating your response."
            
            # Add to prompt
            enhanced_prompt = prompt + emotion_context
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt with emotion: {e}")
            return prompt
    
    def get_help(self) -> str:
        """Get help text for emotion response generator"""
        help_text = """
🔸 Emotional Intelligence 🔸

I can detect and respond to emotions in our conversations. My emotional intelligence features include:

- Emotion detection: I can identify joy, sadness, anger, fear, surprise, disgust, trust, and more
- Emotional memory: I track emotional patterns over time and build emotional profiles
- Empathetic responses: I adjust my tone and responses based on emotional context
- Emotional awareness: I can recognize significant emotional shifts

You can ask me about:
- "How do I seem today?" - Get insights into your emotional patterns
- "Adjust your emotional tone" - Make me more or less emotionally responsive
- "Emotion help" - Show this help information

I use this emotional awareness to provide more helpful and appropriate responses.
"""
        return help_text
    
    def _select_response_strategy(self,
                                user_emotion: str,
                                user_intensity: str,
                                shift_detected: bool,
                                profile: Dict[str, Any]) -> str:
        """
        Select appropriate response strategy based on emotional context
        
        Args:
            user_emotion: User's dominant emotion
            user_intensity: Intensity of user's emotion
            shift_detected: Whether an emotional shift was detected
            profile: User's emotional profile
            
        Returns:
            Response strategy name
        """
        # Default strategy
        strategy = "neutral"
        
        # Strong emotions need acknowledgment
        if user_intensity == "high":
            if user_emotion in ["joy", "surprise", "trust"]:
                strategy = "match_positive"
            elif user_emotion in ["sadness", "fear"]:
                strategy = "empathize"
            elif user_emotion in ["anger", "disgust"]:
                strategy = "de_escalate"
        # Medium intensity emotions
        elif user_intensity == "medium":
            if user_emotion in ["joy", "surprise", "trust"]:
                strategy = "acknowledge_positive"
            elif user_emotion in ["sadness", "fear", "anger", "disgust"]:
                strategy = "acknowledge_negative"
        
        # Override for emotional shifts
        if shift_detected:
            if user_emotion in ["joy", "surprise", "trust"]:
                strategy = "acknowledge_positive_shift"
            elif user_emotion in ["sadness", "fear", "anger", "disgust"]:
                strategy = "acknowledge_negative_shift"
        
        return strategy
    
    def _apply_response_strategy(self,
                               response: str,
                               strategy: str,
                               user_emotion: str,
                               user_intensity: str) -> str:
        """
        Apply selected response strategy to enhance response
        
        Args:
            response: Original response text
            strategy: Selected response strategy
            user_emotion: User's dominant emotion
            user_intensity: Intensity of user's emotion
            
        Returns:
            Enhanced response
        """
        # If neutral strategy, return original
        if strategy == "neutral":
            return response
            
        # Get templates for this strategy
        templates = self.response_templates.get(strategy, {})
        
        # Get specific template for this emotion if available
        specific_templates = templates.get(user_emotion, templates.get("default", []))
        
        # If no templates found, return original
        if not specific_templates:
            return response
            
        # Select random template
        template = random.choice(specific_templates)
        
        # Check if it's a prefix or suffix
        if template.get("position") == "prefix":
            return f"{template.get('text')} {response}"
        elif template.get("position") == "suffix":
            return f"{response} {template.get('text')}"
        elif template.get("position") == "replace":
            return template.get('text', response)
        else:
            # Default to prefix
            return f"{template.get('text')} {response}"
    
    def _select_empathetic_response(self, 
                                  user_emotion: str,
                                  user_intensity: str) -> Optional[str]:
        """
        Select appropriate empathetic response for emotion
        
        Args:
            user_emotion: User's dominant emotion
            user_intensity: Intensity of user's emotion
            
        Returns:
            Empathetic response or None if not applicable
        """
        # Get empathetic templates
        templates = self.response_templates.get("empathetic", {})
        
        # Get specific template for this emotion if available
        specific_templates = templates.get(user_emotion, templates.get("default", []))
        
        # If no templates found, return None
        if not specific_templates:
            return None
            
        # Filter by intensity if specified
        intensity_templates = [
            t for t in specific_templates 
            if t.get("intensity", "any") in ["any", user_intensity]
        ]
        
        if not intensity_templates:
            return None
            
        # Select random template
        template = random.choice(intensity_templates)
        
        return template.get("text", "")
    
    def _load_response_templates(self) -> Dict[str, Any]:
        """
        Load response templates from file or create default
        
        Returns:
            Dictionary of response templates
        """
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "resources", "response_templates.json")
        
        # If templates exist, load them
        if os.path.exists(templates_path):
            try:
                with open(templates_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading response templates: {e}")
        
        # Create basic default templates
        logger.info("Creating default response templates")
        
        # Default templates for different strategies
        default_templates = {
            "empathetic": {
                "joy": [
                    {"text": "I'm happy to see you're feeling good!", "intensity": "any"},
                    {"text": "That's wonderful news! I'm glad things are going well.", "intensity": "high"},
                    {"text": "It's great to hear you're in good spirits.", "intensity": "medium"}
                ],
                "sadness": [
                    {"text": "I'm sorry you're feeling down.", "intensity": "any"},
                    {"text": "That sounds really difficult. I'm here to listen if you want to talk more about it.", "intensity": "high"},
                    {"text": "I understand that can be disappointing. How can I help?", "intensity": "medium"}
                ],
                "anger": [
                    {"text": "I can see this is frustrating for you.", "intensity": "any"},
                    {"text": "I understand you're upset, and that's completely valid.", "intensity": "high"},
                    {"text": "That does sound annoying. Let's see if we can work through this.", "intensity": "medium"}
                ],
                "fear": [
                    {"text": "It's okay to be concerned about that.", "intensity": "any"},
                    {"text": "That does sound scary. Remember that I'm here to help however I can.", "intensity": "high"},
                    {"text": "I understand your concern. Let's think about this together.", "intensity": "medium"}
                ],
                "surprise": [
                    {"text": "That is quite unexpected!", "intensity": "any"},
                    {"text": "Wow! That's certainly surprising news.", "intensity": "high"},
                    {"text": "I can see why that would catch you off guard.", "intensity": "medium"}
                ],
                "default": [
                    {"text": "I'm here to help with whatever you need.", "intensity": "any"},
                    {"text": "Thank you for sharing that with me.", "intensity": "any"}
                ]
            },
            "match_positive": {
                "joy": [
                    {"text": "That's fantastic!", "position": "prefix"},
                    {"text": "I'm so glad to hear that!", "position": "prefix"},
                    {"text": "This is wonderful news.", "position": "prefix"}
                ],
                "surprise": [
                    {"text": "Wow! That's amazing!", "position": "prefix"},
                    {"text": "That's incredible!", "position": "prefix"},
                    {"text": "I'm pleasantly surprised by that too!", "position": "prefix"}
                ],
                "default": [
                    {"text": "That's great to hear!", "position": "prefix"}
                ]
            },
            "empathize": {
                "sadness": [
                    {"text": "I'm sorry to hear that.", "position": "prefix"},
                    {"text": "That must be difficult for you.", "position": "prefix"},
                    {"text": "I understand this is hard.", "position": "prefix"}
                ],
                "fear": [
                    {"text": "I understand your concern.", "position": "prefix"},
                    {"text": "It's natural to feel worried about this.", "position": "prefix"},
                    {"text": "I can see why that would be concerning.", "position": "prefix"}
                ],
                "default": [
                    {"text": "I understand how you feel.", "position": "prefix"}
                ]
            },
            "de_escalate": {
                "anger": [
                    {"text": "I understand your frustration.", "position": "prefix"},
                    {"text": "I can see this is important to you.", "position": "prefix"},
                    {"text": "Let's work through this together.", "position": "prefix"}
                ],
                "disgust": [
                    {"text": "I understand your reaction.", "position": "prefix"},
                    {"text": "I can see why you'd feel that way.", "position": "prefix"},
                    {"text": "Let's focus on finding a solution.", "position": "prefix"}
                ],
                "default": [
                    {"text": "I'd like to help address your concerns.", "position": "prefix"}
                ]
            },
            "acknowledge_positive": {
                "default": [
                    {"text": "I'm glad to hear that.", "position": "prefix"},
                    {"text": "That sounds positive.", "position": "prefix"}
                ]
            },
            "acknowledge_negative": {
                "default": [
                    {"text": "I understand.", "position": "prefix"},
                    {"text": "I see what you mean.", "position": "prefix"}
                ]
            },
            "acknowledge_positive_shift": {
                "default": [
                    {"text": "I'm glad to see your mood has improved.", "position": "prefix"},
                    {"text": "It's nice to see you're feeling better now.", "position": "prefix"}
                ]
            },
            "acknowledge_negative_shift": {
                "default": [
                    {"text": "I notice you seem to be feeling differently now.", "position": "prefix"},
                    {"text": "I can see this is affecting you.", "position": "prefix"}
                ]
            }
        }
        
        # Ensure the resources directory exists
        os.makedirs(os.path.dirname(templates_path), exist_ok=True)
        
        # Save the default templates for future use
        try:
            with open(templates_path, 'w', encoding='utf-8') as f:
                json.dump(default_templates, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default response templates: {e}")
        
        return default_templates


# Create singleton instance
_response_generator = None

def initialize() -> EmotionResponseGenerator:
    """Initialize the emotion response generator singleton"""
    global _response_generator
    if _response_generator is None:
        logger.info("Initializing emotion response generator...")
        try:
            _response_generator = EmotionResponseGenerator()
            logger.info("Emotion response generator initialized")
        except Exception as e:
            logger.error(f"Error initializing emotion response generator: {e}")
            return None
    return _response_generator

def get_instance() -> Optional[EmotionResponseGenerator]:
    """Get the emotion response generator singleton instance"""
    global _response_generator
    # Auto-initialize if not already done
    if _response_generator is None:
        return initialize()
    return _response_generator

# Helper functions for easy access
def enhance_response(response: str, user_input: str) -> str:
    """Enhance a response with emotional awareness"""
    generator = get_instance()
    if generator:
        return generator.enhance_response(response, user_input)
    return response

def enhance_prompt(prompt: str, user_input: str) -> str:
    """Enhance a prompt with emotional context"""
    generator = get_instance()
    if generator:
        return generator.enhance_prompt(prompt, user_input)
    return prompt

# Auto-initialize the module
try:
    initialize()
except Exception as e:
    logger.error(f"Error during auto-initialization of emotion response generator: {e}")


if __name__ == "__main__":
    # Simple test
    generator = get_instance()
    
    test_cases = [
        {
            "input": "I'm so happy about my promotion today!",
            "response": "Congratulations on your promotion."
        },
        {
            "input": "This makes me so angry, nothing is working!",
            "response": "Let me help you troubleshoot the issue."
        },
        {
            "input": "I'm feeling really sad and overwhelmed right now.",
            "response": "There are several ways to approach this problem."
        },
        {
            "input": "I'm a bit nervous about this presentation tomorrow.",
            "response": "Preparation is key for presentations."
        },
        {
            "input": "Wow! I can't believe they approved our project!",
            "response": "The project will start next week."
        }
    ]
    
    print("=== Emotion Response Enhancement Test ===")
    for case in test_cases:
        user_input = case["input"]
        original = case["response"]
        
        # Get emotions
        if EMOTION_MODULES_AVAILABLE:
            emotions = get_emotion_detector().detect_emotions(user_input)
            print(f"\nInput: {user_input}")
            print(f"Detected emotion: {emotions.get('dominant_emotion', 'neutral')} ({emotions.get('dominant_intensity', 'low')})")
            
            # Add to emotion memory
            if get_emotion_memory():
                get_emotion_memory().add_emotion_record(user_input, emotions)
        
        # Enhance response
        enhanced = generator.enhance_response(original, user_input)
        
        print(f"Original: {original}")
        print(f"Enhanced: {enhanced}")
        
        # Check for empathetic response
        empathetic = generator.generate_empathetic_response(user_input)
        if empathetic:
            print(f"Empathetic: {empathetic}")
