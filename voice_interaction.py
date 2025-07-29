"""
Voice Interaction System for anima_v4

Integrates streaming STT, enhanced VAD, and efficient TTS for a smooth
voice conversation experience similar to ChatGPT/Copilot.
"""

import os
import sys
import time
import threading
import tempfile
import logging
import atexit
from typing import Optional, Callable, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice_interaction")

# Import our voice components
try:
    from stt.voice_controller import get_voice_controller, listen_for_speech
    from stt.voice_controller import start_voice_monitoring, stop_voice_monitoring
except ImportError:
    logger.error("Failed to import required modules. Make sure all voice modules are available.")
    raise

# Voice interaction configuration
VOICE_INTERACTION_CONFIG = {
    "auto_enable_smart_duplex": True,      # Use VAD-based interaction
    "visualize_voice_activity": True,      # Show visual feedback during recording
    "partial_transcripts": True,           # Show transcripts as they're being processed
    "auto_cleanup": True,                  # Clean up temp files on exit
    "min_transcript_confidence": 0.6,      # Minimum confidence for accepting transcript
    "max_listening_seconds": 30,           # Maximum listening duration
    "silence_timeout_seconds": 1.5,        # Silence duration before stopping recording
    "enable_wake_word": False,             # Future: Wake word detection
    "wake_word": "anima",                  # Future: Wake word to listen for
    "terminate_phrases": ["exit voice mode", "stop listening"],  # Phrases to exit voice mode
    "voice_feedback_sounds": False,        # Future: Audio feedback for events
    "tts_voice": "default",                # TTS voice to use
}

class VoiceInteraction:
    """Main voice interaction system for conversational voice interface."""
    
    def __init__(self, config=None):
        """Initialize the voice interaction system.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self.config = config or VOICE_INTERACTION_CONFIG.copy()
        
        # Initialize components
        self.voice_controller = get_voice_controller()
        self.temp_files = []
        
        # State tracking
        self.voice_mode_enabled = False
        self.is_listening = False
        self.is_speaking = False
        self.current_transcript = ""
        
        # Register callbacks
        self.voice_controller.on_speech_start = self._on_speech_start
        self.voice_controller.on_speech_end = self._on_speech_end
        self.voice_controller.on_partial_transcript = self._on_partial_transcript
        self.voice_controller.on_final_transcript = self._on_final_transcript
        
        # External callbacks
        self.on_voice_input = None  # Called when voice input is ready
        self.on_voice_start = None  # Called when user starts speaking
        self.on_voice_stop = None   # Called when user stops speaking
        self.on_voice_error = None  # Called on error
        self.on_voice_update = None # Called with status updates
        
        # Register cleanup
        if self.config.get("auto_cleanup", True):
            atexit.register(self._cleanup)
    
    def _on_speech_start(self):
        """Handle speech start event."""
        self.is_listening = True
        if self.config.get("partial_transcripts", True):
            self._update_status("Listening...")
        
        # External callback
        if self.on_voice_start:
            self.on_voice_start()
    
    def _on_speech_end(self, audio_buffer):
        """Handle speech end event."""
        self.is_listening = False
        self._update_status("Processing speech...")
        
        # Save audio buffer temporarily
        if audio_buffer and len(audio_buffer) > 0:
            try:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file.write(audio_buffer)
                temp_file.close()
                self.temp_files.append(temp_file.name)
                
                # For debugging: logger.info(f"Audio saved to {temp_file.name}")
            except Exception as e:
                logger.error(f"Error saving audio buffer: {e}")
        
        # External callback
        if self.on_voice_stop:
            self.on_voice_stop()
    
    def _on_partial_transcript(self, text):
        """Handle partial transcript updates."""
        if not text or not self.config.get("partial_transcripts", True):
            return
            
        self.current_transcript = text
        self._update_status(f"Transcribing: {text}")
    
    def _on_final_transcript(self, text):
        """Handle final transcript."""
        self.current_transcript = text
        
        # Check for termination phrases
        if self._check_terminate_phrases(text):
            self._update_status("Voice mode disabled by command")
            self.disable_voice_mode()
            return
            
        # Notify about final result
        self._update_status(f"Transcribed: {text}")
        
        # Pass to callback
        if text and self.on_voice_input:
            self.on_voice_input(text)
    
    def _update_status(self, message):
        """Update status via callback."""
        if self.on_voice_update:
            self.on_voice_update(message)
        else:
            # Default is to print to console when no callback provided
            print(f"\r{message}", end="", flush=True)
            if message.startswith("Transcribed:"):
                print()  # Add newline after complete transcription
    
    def _check_terminate_phrases(self, text):
        """Check if transcript contains termination phrases.
        
        Returns:
            True if termination phrase found, False otherwise
        """
        if not text:
            return False
            
        text = text.lower().strip()
        terminate_phrases = self.config.get("terminate_phrases", [])
        
        for phrase in terminate_phrases:
            if phrase.lower() in text:
                return True
                
        return False
    
    def _cleanup(self):
        """Clean up temporary files."""
        if not self.temp_files:
            return
            
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                # Just log errors during cleanup, don't raise
                logger.error(f"Error cleaning up temp file {file_path}: {e}")
        
        self.temp_files = []
    
    def enable_voice_mode(self, use_smart_duplex=True):
        """Enable voice mode for speech input.
        
        Args:
            use_smart_duplex: Whether to use VAD-based interaction
            
        Returns:
            True if enabled successfully, False otherwise
        """
        if self.voice_mode_enabled:
            logger.info("Voice mode already enabled")
            return True
            
        try:
            # Configure voice controller
            self.voice_controller.config["auto_start_streaming"] = True
            self.voice_controller.config["show_partial_results"] = self.config.get("partial_transcripts", True)
            
            # Start voice monitoring
            success = start_voice_monitoring(
                on_speech_start=self._on_speech_start,
                on_speech_end=self._on_speech_end,
                on_partial_transcript=self._on_partial_transcript,
                on_final_transcript=self._on_final_transcript
            )
            
            if not success:
                logger.error("Failed to start voice monitoring")
                if self.on_voice_error:
                    self.on_voice_error("Failed to enable voice mode")
                return False
                
            self.voice_mode_enabled = True
            self._update_status("Voice mode enabled. Speak naturally...")
            logger.info("Voice mode enabled")
            return True
            
        except Exception as e:
            logger.error(f"Error enabling voice mode: {e}")
            if self.on_voice_error:
                self.on_voice_error(f"Error: {e}")
            return False
    
    def disable_voice_mode(self):
        """Disable voice mode.
        
        Returns:
            True if disabled successfully, False otherwise
        """
        if not self.voice_mode_enabled:
            return True
            
        try:
            # Stop voice monitoring
            stop_voice_monitoring()
            
            self.voice_mode_enabled = False
            self.is_listening = False
            self.is_speaking = False
            self.current_transcript = ""
            
            self._update_status("Voice mode disabled")
            logger.info("Voice mode disabled")
            return True
            
        except Exception as e:
            logger.error(f"Error disabling voice mode: {e}")
            if self.on_voice_error:
                self.on_voice_error(f"Error: {e}")
            return False
    
    def toggle_voice_mode(self, use_smart_duplex=True):
        """Toggle voice mode on/off.
        
        Args:
            use_smart_duplex: Whether to use VAD-based interaction
            
        Returns:
            New voice mode state (True=enabled, False=disabled)
        """
        if self.voice_mode_enabled:
            self.disable_voice_mode()
            return False
        else:
            return self.enable_voice_mode(use_smart_duplex)
    
    def get_voice_input(self, timeout=None):
        """Get voice input (blocking).
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Transcribed text or None on error/timeout
        """
        # If voice mode is not enabled, enable temporarily
        was_enabled = self.voice_mode_enabled
        if not was_enabled:
            self.enable_voice_mode()
            
        try:
            # Wait for user to speak and get transcription
            timeout = timeout or self.config.get("max_listening_seconds", 30)
            self._update_status("Listening for speech...")
            transcript = listen_for_speech(timeout)
            
            # If we temporarily enabled voice mode, disable it again
            if not was_enabled:
                self.disable_voice_mode()
                
            # Return the transcript
            return transcript
            
        except Exception as e:
            logger.error(f"Error getting voice input: {e}")
            if self.on_voice_error:
                self.on_voice_error(f"Error: {e}")
                
            # Make sure to disable if we enabled it
            if not was_enabled and self.voice_mode_enabled:
                self.disable_voice_mode()
                
            return None
    
    def is_voice_mode_enabled(self):
        """Check if voice mode is enabled.
        
        Returns:
            True if voice mode is enabled, False otherwise
        """
        return self.voice_mode_enabled
        
    def cleanup(self):
        """Public method to clean up resources and stop voice processing.
        
        This should be called when the application exits to ensure
        all resources are properly released.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            # First disable voice mode if it's enabled
            if self.voice_mode_enabled:
                self.disable_voice_mode()
                
            # Clean up any temp files
            self._cleanup()
            
            # Clean up voice controller resources
            if self.voice_controller:
                self.voice_controller.cleanup()
                
            logger.info("Voice interaction resources cleaned up")
            return True
        except Exception as e:
            logger.error(f"Error during voice interaction cleanup: {e}")
            return False


# Global instance for easy access
_voice_interaction = None

def get_voice_interaction(config=None):
    """Get or create the global voice interaction instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        VoiceInteraction instance
    """
    global _voice_interaction
    if _voice_interaction is None:
        _voice_interaction = VoiceInteraction(config)
    return _voice_interaction

def enable_voice_mode(use_smart_duplex=True):
    """Enable voice mode.
    
    Args:
        use_smart_duplex: Whether to use VAD-based interaction
        
    Returns:
        True if enabled successfully, False otherwise
    """
    interaction = get_voice_interaction()
    return interaction.enable_voice_mode(use_smart_duplex)

def disable_voice_mode():
    """Disable voice mode.
    
    Returns:
        True if disabled successfully, False otherwise
    """
    interaction = get_voice_interaction()
    return interaction.disable_voice_mode()

def toggle_voice_mode(use_smart_duplex=True):
    """Toggle voice mode on/off.
    
    Args:
        use_smart_duplex: Whether to use VAD-based interaction
        
    Returns:
        New voice mode state (True=enabled, False=disabled)
    """
    interaction = get_voice_interaction()
    return interaction.toggle_voice_mode(use_smart_duplex)

def get_voice_input(timeout=None):
    """Get voice input (blocking).
    
    Args:
        timeout: Optional timeout in seconds
        
    Returns:
        Transcribed text or None on error/timeout
    """
    interaction = get_voice_interaction()
    return interaction.get_voice_input(timeout)
    
# Test function
def test_voice_interaction():
    """Test the voice interaction system."""
    print("Testing Voice Interaction System...")
    
    # Create interaction with feedback
    interaction = get_voice_interaction()
    
    def status_update(message):
        print(f"\r{message}", end="", flush=True)
        if "Transcribed:" in message:
            print()
            
    interaction.on_voice_update = status_update
    interaction.on_voice_error = lambda msg: print(f"\nError: {msg}")
    
    try:
        # Test toggling voice mode
        print("\n1. Testing voice mode toggle...")
        interaction.toggle_voice_mode()
        print("Voice mode enabled. Speak something and then wait for transcription.")
        print("(Voice mode will automatically listen for speech)")
        time.sleep(10)  # Allow time for speaking
        
        # Test direct input
        interaction.disable_voice_mode()
        print("\n2. Testing direct voice input...")
        print("Speak after the prompt...")
        transcript = interaction.get_voice_input(timeout=10)
        print(f"You said: {transcript}")
        
        print("\nTest complete!")
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        # Ensure voice mode is disabled
        interaction.disable_voice_mode()

# Run test if module is executed directly
if __name__ == "__main__":
    test_voice_interaction()
