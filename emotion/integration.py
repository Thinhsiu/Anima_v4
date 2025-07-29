"""
Emotion Integration - Connects emotion systems with the main application
Provides unified interface for emotion awareness and response enhancement
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import threading

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
    from emotion.response_generator import get_instance as get_response_generator
    
    EMOTION_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Not all emotion components are available: {e}")
    EMOTION_COMPONENTS_AVAILABLE = False

# Try to import voice-emotion integration if available
try:
    from emotion.voice_bridge import process_voice_audio, get_last_voice_emotion, connect_to_voice_system
    VOICE_EMOTION_AVAILABLE = True
except ImportError:
    logger.warning("Voice emotion analysis not available")
    VOICE_EMOTION_AVAILABLE = False
    
    # Create dummy functions
    def process_voice_audio(*args, **kwargs):
        return {"status": "error", "message": "Voice emotion not available"}
        
    def get_last_voice_emotion():
        return {}
        
    def connect_to_voice_system():
        return False

# Try to import voice modules for integration
try:
    from voice.integration import get_voice_processor
    VOICE_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("Voice integration not available for emotion analysis")
    VOICE_INTEGRATION_AVAILABLE = False
    
# Try to import memory bridge for enhanced memory integration
try:
    from core.memory_bridge import get_memory_bridge, MEMORY_BRIDGE_AVAILABLE
    logger.info("Memory bridge available for emotion integration")
except ImportError:
    logger.warning("Memory bridge not available for emotion integration")
    MEMORY_BRIDGE_AVAILABLE = False
    
    # Create a dummy function
    def get_memory_bridge():
        return None

class EmotionIntegration:
    """
    Integrates emotion awareness into the main application flow
    Coordinates between emotion detection, memory, and response generation
    """
    
    def __init__(self):
        """Initialize the emotion integration system"""
        # Initialize emotion components
        self.emotion_detector = get_emotion_detector() if EMOTION_COMPONENTS_AVAILABLE else None
        self.emotion_memory = get_emotion_memory() if EMOTION_COMPONENTS_AVAILABLE else None
        self.response_generator = get_response_generator() if EMOTION_COMPONENTS_AVAILABLE else None
        
        # Track whether system is fully initialized
        self.is_initialized = (
            self.emotion_detector is not None and
            self.emotion_memory is not None and
            self.response_generator is not None
        )
        
        # Configure settings
        self.settings = {
            "emotional_awareness": True,  # Whether to use emotion awareness
            "response_enhancement": True,  # Whether to enhance responses with emotion
            "prompt_enhancement": True,   # Whether to enhance prompts with emotion
            "empathetic_responses": True  # Whether to generate empathetic responses
        }
        
        # Statistics
        self.stats = {
            "conversations_analyzed": 0,
            "responses_enhanced": 0,
            "prompts_enhanced": 0,
            "empathetic_responses": 0,
            "voice_emotions_integrated": 0
        }
        
        logger.info("Emotion integration initialized")
        
    def initialize(self):
        """Complete initialization of the emotion integration system"""
        if not self.is_initialized:
            logger.warning("Cannot initialize emotion system: components unavailable")
            return False
            
        try:
            # Initialize emotion components
            if self.emotion_detector:
                if hasattr(self.emotion_detector, 'initialize'):
                    self.emotion_detector.initialize()
                
            if self.emotion_memory:
                if hasattr(self.emotion_memory, 'initialize'):
                    self.emotion_memory.initialize()
                
            if self.response_generator:
                if hasattr(self.response_generator, 'initialize'):
                    self.response_generator.initialize()
            
            # Connect to voice system if available
            if VOICE_EMOTION_AVAILABLE:
                connected = connect_to_voice_system()
                if connected:
                    logger.info("Voice-emotion bridge connected successfully")
                else:
                    logger.warning("Could not connect voice-emotion bridge")
                
            logger.info("Emotion integration system fully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing emotion integration: {e}")
            return False
    
    def analyze_user_input(self, 
                          user_input: str, 
                          profile_id: str = "default") -> Dict[str, Any]:
        """
        Analyze user input for emotional content
        
        Args:
            user_input: User input text
            profile_id: Emotional profile ID
            
        Returns:
            Analysis results
        """
        # Check if emotion system is available and enabled
        if not self.is_initialized or not self.settings["emotional_awareness"]:
            return {"status": "emotion_system_unavailable"}
            
        try:
            # Basic emotion analysis
            emotions = self.emotion_detector.detect_emotions(user_input)
            
            # Record in emotion memory
            if self.emotion_memory:
                record = self.emotion_memory.add_emotion_record(
                    user_input,
                    emotions,
                    profile_id=profile_id
                )
                
                # Check for emotional shifts
                shift = self.emotion_memory.detect_emotional_shift(
                    user_input,
                    profile_id=profile_id
                )
                
                # Add shift information
                emotions["shift_detected"] = shift.get("shift_detected", False)
                emotions["shift_magnitude"] = shift.get("magnitude", 0.0)
            
            # Update statistics
            self.stats["conversations_analyzed"] += 1
            
            # Create result object
            result = {
                "status": "success",
                "emotions": emotions,
                "profile_id": profile_id,
                "timestamp": time.time()
            }
            
            # Store in memory bridge if available
            if MEMORY_BRIDGE_AVAILABLE:
                try:
                    memory_bridge = get_memory_bridge()
                    if memory_bridge:
                        memory_bridge.add_emotional_data(result)
                except Exception as e:
                    logger.warning(f"Error storing emotional data in memory: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing emotions in user input: {e}")
            return {"status": "error", "error": str(e)}
    
    def enhance_response(self, 
                       response: str, 
                       user_input: str,
                       profile_id: str = "default") -> str:
        """
        Enhance response with emotional awareness
        
        Args:
            response: Original response text
            user_input: User input that triggered the response
            profile_id: Emotional profile ID
            
        Returns:
            Enhanced response
        """
        # Check if emotion system is available and enhancement is enabled
        if not self.is_initialized or not self.settings["response_enhancement"]:
            return response
            
        try:
            # Check for empathetic response opportunity
            if self.settings["empathetic_responses"] and self.response_generator:
                empathetic = self.response_generator.generate_empathetic_response(
                    user_input,
                    profile_id=profile_id
                )
                
                # If empathetic response generated, consider using it
                if empathetic:
                    self.stats["empathetic_responses"] += 1
                    # For high emotion situations, use empathetic response as prefix
                    emotions = self.emotion_detector.detect_emotions(user_input)
                    if emotions.get("dominant_intensity", "low") == "high":
                        response = f"{empathetic} {response}"
            
            # Enhance the response if available
            if self.response_generator:
                enhanced = self.response_generator.enhance_response(
                    response,
                    user_input,
                    profile_id=profile_id
                )
                
                # Update statistics if changed
                if enhanced != response:
                    self.stats["responses_enhanced"] += 1
                    
                return enhanced
                
            return response
            
        except Exception as e:
            logger.error(f"Error enhancing response with emotion: {e}")
            return response
    
    def enhance_prompt(self, 
                     prompt: str, 
                     user_input: str,
                     profile_id: str = "default") -> str:
        """
        Enhance LLM prompt with emotional context
        
        Args:
            prompt: Original prompt
            user_input: User input for context
            profile_id: Emotional profile ID
            
        Returns:
            Enhanced prompt
        """
        # Check if emotion system is available and prompt enhancement is enabled
        if not self.is_initialized or not self.settings["prompt_enhancement"]:
            return prompt
            
        try:
            # Enhance prompt if response generator is available
            if self.response_generator:
                enhanced = self.response_generator.enhance_prompt(
                    prompt,
                    user_input,
                    profile_id=profile_id
                )
                
                # Update statistics
                self.stats["prompts_enhanced"] += 1
                
                return enhanced
                
            return prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt with emotion: {e}")
            return prompt
    
    def integrate_voice_emotion(self, 
                              voice_data: Dict[str, Any], 
                              text: str,
                              profile_id: str = "default") -> Dict[str, Any]:
        """
        Integrate voice emotion analysis with text emotion analysis
        
        Args:
            voice_data: Voice analysis data
            text: Transcribed text
            profile_id: Emotional profile ID
            
        Returns:
            Integrated emotion analysis
        """
        # Check if emotion system and voice integration are available
        if not self.is_initialized or not VOICE_INTEGRATION_AVAILABLE:
            return {"status": "integration_unavailable"}
            
        try:
            # Get text-based emotion analysis
            text_emotions = self.emotion_detector.detect_emotions(text)
            
            # Extract voice emotion data if available
            voice_emotion = voice_data.get("emotion", {})
            voice_features = voice_data.get("acoustic_features", {})
            
            # Combine text and voice emotions
            combined = self._combine_text_voice_emotions(
                text_emotions,
                voice_emotion,
                voice_features
            )
            
            # Record in emotion memory
            if self.emotion_memory:
                context = {"source": "voice_and_text"}
                self.emotion_memory.add_emotion_record(
                    text,
                    combined,
                    profile_id=profile_id,
                    context=context
                )
            
            # Update statistics
            self.stats["voice_emotions_integrated"] += 1
            
            return {
                "status": "success",
                "emotions": combined,
                "text_emotions": text_emotions,
                "voice_emotions": voice_emotion,
                "profile_id": profile_id
            }
            
        except Exception as e:
            logger.error(f"Error integrating voice and text emotions: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_emotional_profile(self, profile_id: str = "default") -> Dict[str, Any]:
        """
        Get emotional profile information
        
        Args:
            profile_id: Emotional profile ID
            
        Returns:
            Emotional profile data
        """
        # Check if emotion memory is available
        if not self.is_initialized or not self.emotion_memory:
            return {"status": "emotion_memory_unavailable"}
            
        try:
            # Get profile from emotion memory
            profile = self.emotion_memory.get_emotion_profile(profile_id)
            
            # Get emotional history summary
            summary = self.emotion_memory.summarize_emotional_history(profile_id)
            
            # Combine profile and summary
            result = {
                "status": "success",
                "profile": profile,
                "summary": summary
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting emotional profile: {e}")
            return {"status": "error", "error": str(e)}
    
    def update_settings(self, settings: Dict[str, bool]) -> Dict[str, Any]:
        """
        Update emotion integration settings
        
        Args:
            settings: Dictionary of settings to update
            
        Returns:
            Updated settings
        """
        # Update only provided settings
        for key, value in settings.items():
            if key in self.settings:
                self.settings[key] = value
                
        return {
            "status": "success",
            "settings": self.settings
        }
    
    def get_help(self) -> str:
        """Get help text for emotion integration"""
        if not self.is_initialized:
            return "Emotion awareness system is not fully initialized."
            
        # Start with basic help
        help_text = """
🔸 Emotion Awareness System 🔸

I can understand and respond to emotions in our conversations. My emotion features include:

- Emotion detection: I identify emotions in your messages
- Emotional memory: I track emotional patterns over time
- Empathetic responses: I adapt my responses based on emotional context
- Voice-emotion integration: I can analyze emotions in your voice (when voice mode is active)

You can use these commands:
- "emotion help" - Show this help information
- "emotion profile" - View your emotional profile
- "adjust emotion awareness" - Configure emotion settings

Current settings:
"""
        
        # Add current settings
        for setting, value in self.settings.items():
            status = "enabled" if value else "disabled"
            help_text += f"- {setting.replace('_', ' ').title()}: {status}\n"
            
        # Add statistics if available
        help_text += "\nEmotion System Statistics:\n"
        for stat, value in self.stats.items():
            help_text += f"- {stat.replace('_', ' ').title()}: {value}\n"
            
        return help_text
    
    def _combine_text_voice_emotions(self,
                                  text_emotions: Dict[str, Any],
                                  voice_emotion: Dict[str, Any],
                                  voice_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine text and voice emotion analyses
        
        Args:
            text_emotions: Text-based emotion analysis
            voice_emotion: Voice-based emotion analysis
            voice_features: Acoustic features from voice
            
        Returns:
            Combined emotion analysis
        """
        # Start with text emotions
        combined = dict(text_emotions)
        
        # Extract key information
        text_dominant = text_emotions.get("dominant_emotion", "neutral")
        voice_dominant = voice_emotion.get("dominant_emotion", "neutral")
        
        # If dominants differ, adjust based on confidence
        if text_dominant != voice_dominant:
            text_confidence = text_emotions.get("confidence", 0.5)
            voice_confidence = voice_emotion.get("confidence", 0.5)
            
            # Use the more confident analysis
            if voice_confidence > text_confidence + 0.2:
                combined["dominant_emotion"] = voice_dominant
                combined["dominant_intensity"] = voice_emotion.get("dominant_intensity", "medium")
                combined["confidence"] = voice_confidence
                combined["source"] = "primarily_voice"
            else:
                # Default to text but note the discrepancy
                combined["voice_emotion_mismatch"] = True
                combined["voice_dominant"] = voice_dominant
                combined["source"] = "primarily_text"
        else:
            # Same dominant emotion, increase confidence
            combined["confidence"] = min(
                1.0, 
                text_emotions.get("confidence", 0.5) * 1.25
            )
            combined["source"] = "text_voice_agreement"
            
        # Integrate acoustic features if available
        if voice_features:
            # Speed can indicate excitement or anxiety
            if voice_features.get("speech_rate", 0) > 1.2:
                if text_dominant in ["joy", "surprise"]:
                    combined["dominant_intensity"] = "high"
                elif text_dominant in ["fear", "anger"]:
                    combined["dominant_intensity"] = "high"
                    
            # Volume can indicate intensity
            if voice_features.get("volume", 0) > 1.2:
                combined["dominant_intensity"] = "high"
                
            # Pauses can indicate thoughtfulness or hesitation
            if voice_features.get("pauses", 0) > 1.5:
                if text_dominant in ["fear", "sadness"]:
                    combined["hesitation"] = True
            
        return combined


# Create singleton instance
_emotion_integration = None

def initialize() -> EmotionIntegration:
    """Initialize the emotion integration singleton"""
    global _emotion_integration
    try:
        if _emotion_integration is None:
            logger.info("Initializing emotion integration...")
            # Initialize main integration object
            _emotion_integration = EmotionIntegration()
            logger.info("Emotion integration initialized")
        return _emotion_integration
    except Exception as e:
        logger.error(f"Error initializing emotion integration: {e}")
        return None

# Initialize the instance
_emotion_integration = initialize()

# Export for direct import by other modules
emotion_integration = _emotion_integration

# Export these for other modules
__all__ = ["EmotionIntegration", "add_emotional_context", "process_text", "emotion_integration"]

def get_instance() -> Optional[EmotionIntegration]:
    """Get the emotion integration singleton instance"""
    global _emotion_integration
    # Auto-initialize if not already done
    if _emotion_integration is None:
        return initialize()
    return _emotion_integration

# Helper functions for easy access
def analyze_user_input(user_input: str) -> Dict[str, Any]:
    """Analyze user input for emotional content"""
    integration = get_instance()
    if integration:
        return integration.analyze_user_input(user_input)
    return {"status": "integration_unavailable"}

def enhance_response(response: str, user_input: str) -> str:
    """Enhance response with emotional awareness"""
    integration = get_instance()
    if integration:
        return integration.enhance_response(response, user_input)
    return response

def enhance_prompt(prompt: str, user_input: str) -> str:
    """Enhance prompt with emotional context"""
    integration = get_instance()
    if integration:
        return integration.enhance_prompt(prompt, user_input)
    return prompt

def get_emotion_help() -> str:
    """Get help text for emotion system"""
    integration = get_instance()
    if integration:
        return integration.get_help()
    return "Emotion awareness system is not available."

# Auto-initialize the module
try:
    initialize()
except Exception as e:
    logger.error(f"Error during auto-initialization of emotion integration: {e}")
