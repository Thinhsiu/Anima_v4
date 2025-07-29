"""
Voice-Emotion Bridge for Anima AI

Connects the voice interaction system with the emotion analysis components
to extract emotional features from voice input alongside text analysis.
"""

import os
import sys
import logging
import threading
import tempfile
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice_emotion")

# Global state tracking
_voice_emotion_enabled = True
_last_voice_emotion = {}
_voice_audio_buffer = None
_processing_lock = threading.Lock()

MAX_AUDIO_BUFFER_SIZE = 10  # Define the maximum audio buffer size

class VoiceEmotionBridge:
    """Bridge between voice interaction and emotion analysis systems."""
    
    def __init__(self):
        """Initialize the voice-emotion bridge."""
        self.enabled = True
        self.voice_features = {}
        self.audio_buffer = None
        self.temp_files = []
        self.voice_connected = False
        self.last_audio = None
        self.audio_buffer = deque(maxlen=MAX_AUDIO_BUFFER_SIZE)
        self.processing_lock = threading.Lock()
        
        # Connect to memory bridge
        try:
            from core.memory_bridge import memory_bridge
            self.memory_bridge = memory_bridge
            logger.info("Voice-emotion bridge connected to memory bridge")
        except ImportError:
            logger.warning("Memory bridge not available for voice-emotion bridge")
            self.memory_bridge = None
        
        # Statistics
        self.stats = {
            "audio_processed": 0,
            "emotions_detected": 0,
            "confidence_avg": 0.0,
        }
        
    def extract_emotion_features(self, audio_data: bytes) -> Dict[str, Any]:
        """Extract emotional features from audio data.
        
        Args:
            audio_data: Raw audio bytes to analyze
            
        Returns:
            Dictionary with extracted emotional features
        """
        if not audio_data or not self.enabled:
            return {}
            
        try:
            # For now, we'll implement a simple placeholder that detects
            # basic voice features like volume, speech rate, and pauses
            return self._extract_basic_features(audio_data)
            
        except Exception as e:
            logger.error(f"Error extracting emotion features: {e}")
            return {}
    
    def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process audio data to extract emotional features.
        
        Args:
            audio_data: Raw audio bytes to analyze
            
        Returns:
            Dictionary with emotional features extracted from voice
        """
        if not audio_data or not self.enabled:
            return {"status": "error", "message": "No audio data or system disabled"}
            
        try:
            # Store the audio buffer for potential future processing
            self.audio_buffer = audio_data
            
            # Extract emotion features
            features = self.extract_emotion_features(audio_data)
            
            # Update statistics
            self.stats["audio_processed"] += 1
            if features.get("emotion_detected", False):
                self.stats["emotions_detected"] += 1
                self.stats["confidence_avg"] = (
                    (self.stats["confidence_avg"] * (self.stats["emotions_detected"] - 1) + 
                     features.get("confidence", 0.0)) / self.stats["emotions_detected"]
                )
                
            return {
                "status": "success",
                "voice_features": features,
                "message": "Voice features extracted"
            }
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            return {"status": "error", "message": str(e)}
            
    def _extract_basic_features(self, audio_data: bytes) -> Dict[str, Any]:
        """Extract emotion features from audio data using ML models.
        
        Uses advanced ML-based voice emotion detection with fallback to
        simpler models if dependencies aren't available.
        
        Args:
            audio_data: Raw audio bytes to analyze
            
        Returns:
            Dictionary of extracted features with emotion classification
        """
        try:
            # Use our ML-based emotion classifier
            from emotion.voice_ml_model import get_classifier
            
            # Get the best available classifier
            classifier = get_classifier()
            logger.info(f"Using voice emotion classifier: {classifier.__class__.__name__}")
            
            # Process audio data and extract emotions
            features = classifier.predict_emotion(audio_data)
            
            # Add audio length for backward compatibility
            audio_length = len(audio_data)
            if "voice_metrics" in features and isinstance(features["voice_metrics"], dict):
                features["voice_metrics"]["audio_duration_ms"] = audio_length // 100
            
            # Ensure we have all expected fields
            if "dominant_emotion" not in features:
                features["dominant_emotion"] = "neutral"
            if "confidence" not in features:
                features["confidence"] = 0.5
            if "emotion_detected" not in features:
                features["emotion_detected"] = True
                
        except Exception as e:
            # Log the error and fall back to basic heuristics
            logger.error(f"Error in ML voice emotion extraction: {e}")
            logger.warning("Falling back to basic voice emotion heuristics")
            
            # Fallback to basic heuristics based on audio length
            audio_length = len(audio_data)
            
            # Simple heuristics based on audio buffer size
            volume = min(1.0, audio_length / 1000000)
            speech_rate = 0.5 + (audio_length % 100000) / 100000
            
            # Determine a simulated emotion based on these features
            dominant_emotion = "neutral"
            if volume > 0.8:
                if speech_rate > 0.7:
                    dominant_emotion = "excited"
                else:
                    dominant_emotion = "confident"
            elif volume < 0.3:
                if speech_rate < 0.4:
                    dominant_emotion = "sad"
                else:
                    dominant_emotion = "thoughtful"
                    
            # Create features dictionary
            features = {
                "emotion_detected": True,
                "dominant_emotion": dominant_emotion,
                "confidence": 0.6,  # Placeholder confidence value
                "voice_metrics": {
                    "volume": volume,
                    "speech_rate": speech_rate,
                    "pauses_detected": audio_length // 200000,
                    "audio_duration_ms": audio_length // 100
                }
            }
            
        # Store emotional data in memory if available
        if hasattr(self, 'memory_bridge') and self.memory_bridge:
            try:
                memory_data = {
                    "emotion": features["dominant_emotion"],
                    "confidence": features["confidence"],
                    "source": "voice_analysis",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "metrics": features["voice_metrics"]
                }
                
                # Add emotion scores if available
                if "emotion_scores" in features:
                    memory_data["emotion_scores"] = features["emotion_scores"]
                
                success = self.memory_bridge.add_emotional_data(memory_data)
                if success:
                    logger.info(f"Stored voice emotion in memory: {features['dominant_emotion']} (confidence: {features['confidence']:.2f})")
                else:
                    logger.warning("Failed to store voice emotion in memory")
            except Exception as e:
                logger.error(f"Error storing voice emotion in memory: {e}")
                
        # Return the features dictionary
        return features
        
    def get_last_voice_features(self) -> Dict[str, Any]:
        """Get the most recently extracted voice features.
        
        Returns:
            Dictionary of voice features or empty dict if none available
        """
        return self.voice_features
        
    def enable(self) -> bool:
        """Enable voice emotion processing.
        
        Returns:
            True if successfully enabled
        """
        self.enabled = True
        return True
        
    def disable(self) -> bool:
        """Disable voice emotion processing.
        
        Returns:
            True if successfully disabled
        """
        self.enabled = False
        return True
        
    def cleanup(self):
        """Clean up any temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Error cleaning up temp file {temp_file}: {e}")
        self.temp_files = []

# Global instance for easy access
_voice_emotion_bridge = None

def get_voice_emotion_bridge() -> VoiceEmotionBridge:
    """Get or create the global voice emotion bridge instance.
    
    Returns:
        VoiceEmotionBridge instance
    """
    global _voice_emotion_bridge
    if _voice_emotion_bridge is None:
        _voice_emotion_bridge = VoiceEmotionBridge()
    return _voice_emotion_bridge
    
# Alias for backward compatibility with test scripts
def get_bridge_instance() -> VoiceEmotionBridge:
    """Get the singleton instance of the voice-emotion bridge.
    
    Returns:
        VoiceEmotionBridge instance
    """
    # This must return the EXACT SAME instance as get_voice_emotion_bridge()
    # for the singleton pattern to work correctly
    return get_voice_emotion_bridge()

def process_voice_audio(audio_data: bytes) -> Dict[str, Any]:
    """Process voice audio data for emotional features.
    
    Args:
        audio_data: Raw audio bytes to analyze
        
    Returns:
        Dictionary with emotional features or error info
    """
    bridge = get_voice_emotion_bridge()
    return bridge.process_audio(audio_data)

# Alias for backward compatibility
process_audio = process_voice_audio

# Hook function to connect to voice interaction system
def connect_to_voice_system():
    """Connect the voice-emotion bridge to the voice interaction system.
    
    This function hooks into the voice interaction callbacks to receive
    audio data for emotion analysis.
    
    Returns:
        True if successfully connected, False otherwise
    """
    try:
        # Import the voice interaction system
        from voice_interaction import get_voice_interaction
        
        # Get voice interaction instance
        voice_system = get_voice_interaction()
        if not voice_system:
            logger.error("Failed to get voice interaction instance")
            return False
            
        # Store the original speech_end callback to chain them
        original_speech_end = voice_system.voice_controller.on_speech_end
        
        # Create a new callback that processes emotion and then calls the original
        def enhanced_speech_end(audio_buffer):
            # Process the audio buffer for emotions if available
            if audio_buffer and len(audio_buffer) > 0:
                with _processing_lock:
                    global _voice_audio_buffer
                    _voice_audio_buffer = audio_buffer
                    
                    # Process the audio in the background
                    threading.Thread(
                        target=_process_audio_background,
                        args=(audio_buffer,),
                        daemon=True
                    ).start()
            
            # Call the original callback
            if original_speech_end:
                original_speech_end(audio_buffer)
                
        # Replace the speech_end callback
        voice_system.voice_controller.on_speech_end = enhanced_speech_end
        
        logger.info("Successfully connected voice-emotion bridge to voice system")
        return True
    
    except ImportError:
        logger.warning("Voice interaction system not available, voice emotion features limited")
        return False
    except Exception as e:
        logger.error(f"Error connecting to voice system: {e}")
        return False
        
def _process_audio_background(audio_buffer):
    """Process audio data in the background and store results.
    
    Args:
        audio_buffer: Raw audio bytes to process
    """
    try:
        bridge = get_voice_emotion_bridge()
        result = bridge.process_audio(audio_buffer)
        
        # Store the results globally
        global _last_voice_emotion
        if result.get("status") == "success":
            _last_voice_emotion = result.get("voice_features", {})
            logger.debug(f"Voice emotion detected: {_last_voice_emotion.get('dominant_emotion', 'unknown')}")
    except Exception as e:
        logger.error(f"Error in background audio processing: {e}")

def get_last_voice_emotion() -> Dict[str, Any]:
    """Get the most recently detected voice emotion features.
    
    Returns:
        Dictionary of voice emotion features or empty dict if none available
    """
    return _last_voice_emotion

# Initialize by trying to connect to the voice system
try:
    # Don't connect automatically on import, let the main system decide when to connect
    pass
except Exception as e:
    logger.error(f"Error during voice-emotion bridge initialization: {e}")

# Export for other modules
__all__ = [
    "VoiceEmotionBridge", 
    "get_voice_emotion_bridge",
    "get_bridge_instance",
    "process_audio",
    "process_voice_audio",
    "connect_to_voice_system"
]
