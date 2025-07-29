"""
Voice Controller Module

Integrates streaming STT and enhanced VAD for a smooth voice interaction experience
with real-time feedback and improved latency.
"""

import os
import time
import threading
import queue
import tempfile
import wave
import logging
import pyaudio
from typing import Optional, Callable, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice_controller")

# Import our custom modules
try:
    from .enhanced_vad import get_speech_monitor, start_smart_listening, stop_smart_listening
    from .faster_whisper_streaming import get_streaming_stt
except ImportError:
    # Allow for direct imports when running as main module
    try:
        from enhanced_vad import get_speech_monitor, start_smart_listening, stop_smart_listening
        from faster_whisper_streaming import get_streaming_stt
    except ImportError:
        logger.error("Failed to import required modules. Make sure they're in the same directory.")
        raise

# Voice controller configuration
VOICE_CONFIG = {
    "auto_start_streaming": True,       # Start STT streaming on speech detection
    "show_partial_results": True,       # Show transcription while speaking
    "vad_speech_pad_ms": 300,           # Padding after speech end in ms
    "max_listen_seconds": 60,           # Maximum listening time before timeout
    "min_confidence": 0.6,              # Minimum confidence for accepting transcription
    "noise_filter": True,               # Filter out likely noise
    "energy_threshold": 0.01,           # Energy threshold for VAD
    "timeout_if_no_speech_seconds": 10  # Timeout if no speech detected
}

class VoiceController:
    """Controller for voice interactions with streaming transcription."""
    
    def __init__(self, config=None):
        """Initialize the voice controller.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self.config = config or VOICE_CONFIG.copy()
        
        # Initialize components
        self.stt = get_streaming_stt()
        
        # Complete VAD configuration with all required parameters
        self.vad_config = {
            # Audio parameters
            "sample_rate": 16000,  # Required by EnhancedVAD
            "frame_duration_ms": 30,  # Required by EnhancedVAD
            "frame_size": 480,  # = sample_rate * frame_duration_ms / 1000
            "format": pyaudio.paInt16,  # 16-bit audio format
            "channels": 1,  # Mono audio
            
            # Energy detection parameters
            "energy_threshold": self.config.get("energy_threshold", 0.01),
            "dynamic_threshold": True,
            "dynamic_threshold_percentile": 80,
            "noise_floor_samples": 100,
            
            # Speech segment parameters
            "min_speech_frames": 10,  # Min consecutive frames to trigger start (300ms)
            "speech_pad_frames": int(self.config.get("vad_speech_pad_ms", 500) / 30),  # Padding frames
            "min_silence_frames": 20,  # Min silence frames to end speech (600ms)
            "max_speech_seconds": 30,  # Max speech segment length
            
            # State parameters
            "speech_start_confidence": 3,  # Frames that must be speech to trigger
            "speech_end_confidence": 5,  # Frames that must be silence to end
            
            # Visualization
            "visualize": self.config.get("show_partial_results", True),
            "vis_width": 50,  # Width of visualization in terminal
        }
        
        # State tracking
        self.is_listening = False
        self.listening_thread = None
        self.listen_timeout = None
        self.speech_detected = False
        self._current_transcript = ""
        self._final_transcript = None
        self._transcript_ready = threading.Event()
        self._audio_buffer = None
        self._last_partial_time = 0
        
        # Callbacks
        self.on_listening_start = None
        self.on_listening_stop = None
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_partial_transcript = None
        self.on_final_transcript = None
        self.on_error = None
        
    def _handle_speech_start(self):
        """Handle speech start event from VAD."""
        logger.info("Speech detected")
        self.speech_detected = True
        
        # Reset streaming STT if needed
        if self.config.get("auto_start_streaming", True):
            self._current_transcript = ""
            self.stt.start_streaming()
            
        # Call user callback
        if self.on_speech_start:
            self.on_speech_start()
    
    def _handle_speech_end(self, audio_buffer):
        """Handle speech end event from VAD.
        
        Args:
            audio_buffer: Raw audio data as bytes
        """
        if not audio_buffer:
            logger.warning("Speech ended with no audio buffer")
            self._final_transcript = None
            self._transcript_ready.set()
            return
            
        buffer_size = len(audio_buffer)
        logger.info(f"Speech ended, buffer size: {buffer_size} bytes (~{buffer_size/32000:.2f}s of audio)")
        
        # Store the audio buffer for possible saving/debugging
        self._audio_buffer = audio_buffer
        
        # Minimum viable audio length (in bytes) for reasonable transcription
        # Lower threshold to ~0.1s of 16kHz 16-bit audio = 3200 bytes
        min_viable_bytes = 3200
        
        # Check if the audio segment is extremely short (likely just noise)
        if buffer_size < min_viable_bytes:
            logger.info(f"Speech segment very short ({buffer_size/32000:.2f}s), but attempting transcription anyway")
            # Continue with transcription - don't return early
            # This gives short utterances a chance to be transcribed
            
        try:
            # First try to get the streaming result which is faster
            streaming_result = self._current_transcript
            
            # Stop streaming but give it a moment to generate final results
            self.stt.stop_streaming()
            
            # Wait a brief moment for any pending streaming results
            # Use a longer wait for longer audio segments
            wait_time = min(0.3, buffer_size / 160000)  # Max 300ms wait
            time.sleep(wait_time)
            
            # Refresh streaming result after waiting
            streaming_result = self._current_transcript
            
            # Use streaming results if they seem valid
            if streaming_result and len(streaming_result.strip()) > 2:
                final_transcript = streaming_result
                self._final_transcript = final_transcript
                self._transcript_ready.set()
                logger.info(f"Using streaming result: '{final_transcript}'")
            else:
                # No good streaming result, fall back to full transcription
                logger.info("No good streaming result, using full transcription")
                
                # For short segments, we need special handling but can't pass custom params
                # directly to transcribe_buffer as it doesn't support them
                if len(audio_buffer) < 16000:  # Less than 0.5 seconds
                    logger.info("Short audio segment detected - might be noise rather than speech")
                    
                    # For very short segments that are likely noise, ignore them
                    if len(audio_buffer) < 5000:  # ~0.15 seconds
                        logger.info("Audio segment too short, likely noise - ignoring")
                        self._final_transcript = None
                        self._transcript_ready.set()
                        return
                        
                # Transcribe the audio segment with default parameters
                final_transcript = self.stt.transcribe_buffer(audio_buffer)
                
                if final_transcript and len(final_transcript.strip()) > 0:
                    self._final_transcript = final_transcript
                    logger.info(f"Full transcription: '{final_transcript}'")
                else:
                    logger.warning("Transcription returned empty or None result")
                    self._final_transcript = None
                    
                self._transcript_ready.set()
                
            # Call user callback with the final transcript (if any)
            if self._final_transcript and self.on_final_transcript:
                self.on_final_transcript(self._final_transcript)
                
        except Exception as e:
            logger.error(f"Error in speech end handler: {e}")
            self._final_transcript = None
            self._transcript_ready.set()
        
        # Always call user callback
        if self.on_speech_end:
            self.on_speech_end(audio_buffer)
    
    def _handle_partial_transcript(self, text):
        """Handle partial transcript from streaming STT.
        
        Args:
            text: Partial transcript text
        """
        self._current_transcript = text
        
        # Throttle updates to avoid flooding the console
        now = time.time()
        if now - self._last_partial_time > 0.3:  # Update at most every 300ms
            self._last_partial_time = now
            # Call user callback
            if self.on_partial_transcript:
                self.on_partial_transcript(text)
                
    def _handle_final_transcript(self, text):
        """Handle final transcript from streaming STT.
        
        Args:
            text: Final transcript text
        """
        self._final_transcript = text
        self._transcript_ready.set()
        
        # Call user callback if it hasn't been called by speech_end
        if self.on_final_transcript and self._final_transcript != text:
            self.on_final_transcript(text)
    
    def _listening_thread_func(self):
        """Background thread for listening operation."""
        try:
            # Setup callbacks for streaming STT
            self.stt.setup_callbacks(
                on_partial_transcript=self._handle_partial_transcript,
                on_final_transcript=self._handle_final_transcript
            )
            
            # Start VAD monitoring
            success = start_smart_listening(
                on_speech_start=self._handle_speech_start,
                on_speech_end=self._handle_speech_end,
                config=self.vad_config
            )
            
            if not success:
                logger.error("Failed to start smart listening")
                if self.on_error:
                    self.on_error("Failed to start audio monitoring")
                return
                
            # Main listening loop
            start_time = time.time()
            timeout = self.config.get("max_listen_seconds", 60)
            no_speech_timeout = self.config.get("timeout_if_no_speech_seconds", 10)
            
            while self.is_listening:
                # Check for max timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.info(f"Maximum listening time reached ({timeout}s)")
                    break
                    
                # Check for no-speech timeout
                if not self.speech_detected and elapsed > no_speech_timeout:
                    logger.info(f"No speech detected after {no_speech_timeout}s")
                    break
                    
                # Check if transcript is ready
                if self._transcript_ready.is_set():
                    logger.info("Transcript ready, ending listening")
                    break
                    
                # Short sleep to prevent CPU spinning
                time.sleep(0.1)
                
            # Stop VAD monitoring
            stop_smart_listening()
            
        except Exception as e:
            logger.error(f"Error in listening thread: {e}")
            if self.on_error:
                self.on_error(str(e))
        finally:
            self.is_listening = False
            
            # Call callback
            if self.on_listening_stop:
                self.on_listening_stop()
    
    def start_listening(self, timeout=None):
        """Start listening for speech with optional timeout.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if listening started successfully, False otherwise
        """
        if self.is_listening:
            logger.warning("Already listening")
            return False
            
        # Reset state
        self.speech_detected = False
        self._current_transcript = ""
        self._final_transcript = None
        self._audio_buffer = None
        self._transcript_ready.clear()
        self.listen_timeout = timeout or self.config.get("max_listen_seconds", 60)
        
        # Start listening thread
        self.is_listening = True
        self.listening_thread = threading.Thread(target=self._listening_thread_func)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
        # Call callback
        if self.on_listening_start:
            self.on_listening_start()
            
        logger.info("Started listening for speech")
        return True
    
    def stop_listening(self):
        """Stop listening for speech."""
        if not self.is_listening:
            return
            
        self.is_listening = False
        
        # Wait for thread to finish
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1.0)
            
        # Stop VAD and STT if needed
        stop_smart_listening()
        self.stt.stop_streaming()
        
        logger.info("Stopped listening")
        
    def cleanup(self):
        """Clean up resources used by the voice controller."""
        logger.info("Cleaning up voice controller resources")
        self.stop_listening()
        
        # Clean up STT resources
        if self.stt:
            try:
                self.stt.cleanup()
                logger.info("STT resources cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up STT: {e}")
        
        # Clean up VAD resources        
        try:
            stop_smart_listening()
            logger.info("VAD resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up VAD: {e}")
        
        # Reset all state variables
        self._current_transcript = ""
        self._final_transcript = None
        self._transcript_ready = threading.Event()
    
    def listen(self, timeout=None):
        """Listen for speech and return transcription (blocking).
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Transcribed text or None if timeout/interrupted
        """
        # Start listening
        if not self.start_listening():
            return None
            
        try:
            # Wait for transcription or timeout
            timeout = timeout or self.config.get("max_listen_seconds", 60)
            transcription_ready = self._transcript_ready.wait(timeout)
            
            # Stop listening
            self.stop_listening()
            
            if not transcription_ready:
                logger.warning("Listening timed out without transcription")
                return None
                
            # Get and post-process transcript
            transcript = self._final_transcript
            
            # Filter likely noise
            if self.config.get("noise_filter", True) and transcript:
                if len(transcript.strip()) <= 1:
                    return None
                    
                # Filter out common noise transcriptions
                noise_patterns = [".", "...", "um", "uh", "hmm"]
                if transcript.lower().strip() in noise_patterns:
                    return None
            
            return transcript
            
        except KeyboardInterrupt:
            logger.info("Listening interrupted by user")
            self.stop_listening()
            return None
        except Exception as e:
            logger.error(f"Error in listen method: {e}")
            self.stop_listening()
            return None
    
    def save_last_audio(self, filepath):
        """Save the last audio buffer to a file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        if not self._audio_buffer:
            logger.warning("No audio buffer available")
            return False
            
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # 16kHz
                wf.writeframes(self._audio_buffer)
                
            return True
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return False
    
    def get_current_transcript(self):
        """Get the current partial transcript."""
        return self._current_transcript
    
    def get_final_transcript(self):
        """Get the final transcript if available."""
        return self._final_transcript


# Global instance for easy access
_voice_controller = None

def get_voice_controller(config=None):
    """Get or create the global voice controller instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        VoiceController instance
    """
    global _voice_controller
    if _voice_controller is None:
        _voice_controller = VoiceController(config)
    return _voice_controller

def listen_for_speech(timeout=None, config=None):
    """Listen for speech and return transcription (blocking).
    
    Args:
        timeout: Optional timeout in seconds
        config: Optional configuration
        
    Returns:
        Transcribed text or None on error/timeout
    """
    controller = get_voice_controller(config)
    return controller.listen(timeout)

def start_voice_monitoring(on_speech_start=None, on_speech_end=None, 
                          on_partial_transcript=None, on_final_transcript=None,
                          config=None):
    """Start monitoring for speech with callbacks.
    
    Args:
        on_speech_start: Called when speech starts
        on_speech_end: Called when speech ends with audio buffer
        on_partial_transcript: Called with partial transcripts
        on_final_transcript: Called with final transcripts
        config: Optional configuration
        
    Returns:
        True if started successfully, False otherwise
    """
    controller = get_voice_controller(config)
    
    # Set callbacks
    controller.on_speech_start = on_speech_start
    controller.on_speech_end = on_speech_end
    controller.on_partial_transcript = on_partial_transcript
    controller.on_final_transcript = on_final_transcript
    
    # Start listening
    return controller.start_listening()

def stop_voice_monitoring():
    """Stop monitoring for speech."""
    controller = get_voice_controller()
    controller.stop_listening()

# Simple test function
def test_voice_controller():
    """Test the voice controller functionality."""
    import sys
    print("Testing Voice Controller...")
    
    # Define callbacks
    def on_listening_start():
        print("Listening started... Speak now")
    
    def on_speech_start():
        print("\nSpeech detected!")
    
    def on_partial_transcript(text):
        print(f"\rPartial: {text}", end="", flush=True)
    
    def on_final_transcript(text):
        print(f"\nFinal transcript: {text}")
    
    def on_speech_end(audio):
        print(f"\nSpeech ended, {len(audio)} bytes recorded")
        # Save to temp file
        filename = f"test_voice_{time.time()}.wav"
        with open(filename, "wb") as f:
            f.write(audio)
        print(f"Saved audio to {filename}")
    
    # Create controller with visualization
    config = VOICE_CONFIG.copy()
    config["show_partial_results"] = True
    controller = VoiceController(config)
    
    # Set callbacks
    controller.on_listening_start = on_listening_start
    controller.on_speech_start = on_speech_start
    controller.on_speech_end = on_speech_end
    controller.on_partial_transcript = on_partial_transcript
    controller.on_final_transcript = on_final_transcript
    
    try:
        # Method 1: Non-blocking with callbacks
        print("\nMethod 1: Voice monitoring with callbacks")
        controller.start_listening(timeout=15)
        
        # Wait until stopped or Ctrl+C
        while controller.is_listening:
            time.sleep(0.1)
            
        # Method 2: Blocking call
        print("\nMethod 2: Blocking call")
        print("Say something...")
        transcript = controller.listen(timeout=10)
        print(f"You said: {transcript}")
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        controller.stop_listening()
        print("Test complete")


# Run test if module is executed directly
if __name__ == "__main__":
    test_voice_controller()
