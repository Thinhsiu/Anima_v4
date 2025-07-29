"""
Enhanced Voice Activity Detection (VAD) module

Provides improved speech detection for real-time voice interaction
with configurable parameters and visual feedback.
"""

import os
import io
import time
import queue
import numpy as np
import pyaudio
import threading
import wave
import collections
from typing import Optional, Callable, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_vad")

# Try to import visualization libraries, optional
try:
    from colorama import Fore, Style, init
    init()  # Initialize colorama
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    logger.warning("colorama not available, install for colored terminal output")

# Enhanced VAD configuration
VAD_CONFIG = {
    "sample_rate": 16000,               # Sample rate in Hz
    "frame_duration_ms": 30,            # Frame duration in milliseconds
    "frame_size": 480,                  # Frame size (samples) = sample_rate * frame_duration_ms / 1000
    "format": pyaudio.paInt16,          # Audio format
    "channels": 1,                      # Mono audio
    
    # Speech detection parameters
    "energy_threshold": 0.015,          # Minimum energy level to detect as speech (0-1) - INCREASED to reduce false activations
    "dynamic_threshold": True,          # Dynamically adjust threshold based on noise floor
    "dynamic_threshold_percentile": 90, # Percentile of energy history to set threshold - INCREASED for better noise filtering
    "noise_floor_samples": 150,         # Number of samples to keep for noise floor estimation - INCREASED for better baseline
    
    # Speech segment parameters
    "min_speech_frames": 10,            # Minimum consecutive speech frames to trigger start (300ms) - INCREASED from 5 to 10
    "speech_pad_frames": 8,             # Padding frames added to start/end of speech (240ms) - REDUCED from 10 to 8
    "min_silence_frames": 15,           # Minimum consecutive silence frames to end speech (450ms) - UNCHANGED
    "max_speech_seconds": 30,           # Maximum allowed speech segment in seconds - UNCHANGED
    
    # State parameters
    "speech_start_confidence": 7,       # Number of recent frames that must be speech to trigger start - INCREASED from 3 to 7
    "speech_end_confidence": 5,         # Number of recent frames that must be silence to trigger end - UNCHANGED
    
    # Visualization
    "visualize": True,                  # Enable real-time visualization in terminal
    "vis_width": 50,                    # Width of the visualization
}

class EnhancedVAD:
    """Enhanced Voice Activity Detector with configurable parameters and visualization."""
    
    def __init__(self, config=None):
        """Initialize the enhanced VAD.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self.config = config or VAD_CONFIG.copy()
        
        # Derived config values
        if "frame_size" not in self.config:
            self.config["frame_size"] = int(self.config["sample_rate"] * self.config["frame_duration_ms"] / 1000)
        
        # Initialize state
        self._energy_history = collections.deque(maxlen=self.config["noise_floor_samples"])
        self._is_speech_history = collections.deque(maxlen=max(
            self.config["speech_start_confidence"], 
            self.config["speech_end_confidence"]
        ))
        self._reset_state()
        
        # Runtime objects
        self._audio = None
        self._stream = None
        self._running = False
        self._audio_buffer = bytearray()
        self._silent_chunks = 0
        self._speech_chunks = 0
        
        # Callbacks
        self.on_speech_start = None
        self.on_speech_end = None
        self.on_audio_frame = None
        
    def _reset_state(self):
        """Reset the detector state."""
        self._is_speaking = False
        self._energy_history.clear()
        self._is_speech_history.clear()
        self._current_energy = 0
        self._speech_start_time = None
        self._audio_buffer = bytearray()
        self._silent_chunks = 0
        self._speech_chunks = 0
        # Fill history with False (silence)
        for _ in range(max(self.config["speech_start_confidence"], self.config["speech_end_confidence"])):
            self._is_speech_history.append(False)

    def _calculate_energy(self, audio_chunk):
        """Calculate the energy level of an audio chunk.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            Energy level (0-1)
        """
        if not audio_chunk:
            return 0
            
        # Convert bytes to numpy array of int16
        try:
            data = np.frombuffer(audio_chunk, dtype=np.int16)
            # Convert to float32 and normalize to -1.0 to 1.0
            data = data.astype(np.float32) / 32768.0
            # Calculate RMS energy
            energy = np.sqrt(np.mean(data**2))
            return float(energy)
        except Exception as e:
            logger.error(f"Error calculating energy: {e}")
            return 0

    def _get_threshold(self):
        """Get the current energy threshold for speech detection.
        
        Returns:
            Current threshold value
        """
        if self.config["dynamic_threshold"] and len(self._energy_history) > 0:
            # Use percentile of energy history as dynamic threshold
            threshold = np.percentile(list(self._energy_history), 
                                     self.config["dynamic_threshold_percentile"])
            # Ensure minimum threshold
            return max(threshold, self.config["energy_threshold"])
        else:
            return self.config["energy_threshold"]

    def _is_speech(self, energy):
        """Determine if the given energy level constitutes speech.
        
        Args:
            energy: Energy level of audio frame
            
        Returns:
            True if the frame is considered speech
        """
        threshold = self._get_threshold()
        
        # Apply hysteresis - once we've detected speech, lower the threshold
        # to avoid cutting off speech too early
        if self._is_speaking:
            # Reduce threshold by 30% during active speech to maintain detection
            adjusted_threshold = threshold * 0.7
        else:
            # Normal threshold when not speaking
            adjusted_threshold = threshold
            
        # Use adaptive detection - if we have enough energy history, consider
        # significant changes from average as potential speech onset
        if len(self._energy_history) > 10:
            # Convert deque to list before slicing to avoid TypeError
            recent_energies = list(self._energy_history)
            recent_energies = recent_energies[-10:]
            avg_energy = sum(recent_energies) / len(recent_energies)
            energy_spike = energy > (avg_energy * 1.5) and energy > (threshold * 0.7)
        else:
            energy_spike = False
            
        return energy > adjusted_threshold or energy_spike

    def _update_speech_state(self, is_speech_frame):
        """Update the overall speech state based on frame history.
        
        Args:
            is_speech_frame: Whether the current frame contains speech
        
        Returns:
            Tuple of (speech_start, speech_end) booleans
        """
        # Update history
        self._is_speech_history.append(is_speech_frame)
        
        # Count speech vs silence frames in history
        recent_speech_count = sum(1 for frame in self._is_speech_history if frame)
        recent_silence_count = len(self._is_speech_history) - recent_speech_count
        
        speech_start = False
        speech_end = False
        
        # Detect speech start
        if not self._is_speaking:
            if recent_speech_count >= self.config["speech_start_confidence"]:
                self._is_speaking = True
                speech_start = True
                self._speech_start_time = time.time()
                self._speech_chunks = 0
                
        # Detect speech end
        else:
            # End if silence duration exceeds threshold
            recent_window = list(self._is_speech_history)[-self.config["speech_end_confidence"]:]
            if sum(1 for frame in recent_window if not frame) >= self.config["speech_end_confidence"]:
                self._is_speaking = False
                speech_end = True
            
            # Or if maximum speech duration exceeded
            elif (self._speech_start_time and 
                  time.time() - self._speech_start_time > self.config["max_speech_seconds"]):
                self._is_speaking = False
                speech_end = True
                logger.info("Maximum speech duration reached")
                
        return speech_start, speech_end

    def process_audio_frame(self, audio_frame):
        """Process a single audio frame and update speech detection state.
        
        Args:
            audio_frame: Raw audio data as bytes
            
        Returns:
            Tuple of (is_speech, speech_start, speech_end)
        """
        if not audio_frame or len(audio_frame) < self.config["frame_size"]:
            return False, False, False
            
        # Calculate energy and determine if frame is speech
        energy = self._calculate_energy(audio_frame)
        self._current_energy = energy
        
        # Update energy history for dynamic threshold
        self._energy_history.append(energy)
        
        # Check if frame is speech
        is_speech_frame = self._is_speech(energy)
        
        # Update counters
        if is_speech_frame:
            self._speech_chunks += 1
            self._silent_chunks = 0
        else:
            self._silent_chunks += 1
            
        # Update speech state
        speech_start, speech_end = self._update_speech_state(is_speech_frame)
        
        # Trigger callbacks if state changed
        if speech_start and self.on_speech_start:
            self.on_speech_start()
            
        # Collect audio during speech with padding
        if self._is_speaking or self._speech_chunks > 0:
            self._audio_buffer.extend(audio_frame)
            
        # Handle speech end
        if speech_end:
            audio_data = bytes(self._audio_buffer)
            self._audio_buffer = bytearray()
            if self.on_speech_end:
                self.on_speech_end(audio_data)
                
        # Visualize if enabled
        if self.config["visualize"]:
            self._visualize(is_speech_frame, energy)
            
        # Notify about the frame
        if self.on_audio_frame:
            self.on_audio_frame(audio_frame, is_speech_frame)
            
        return is_speech_frame, speech_start, speech_end
            
    def _visualize(self, is_speech, energy):
        """Visualize the current audio energy and speech state in terminal.
        
        Args:
            is_speech: Whether current frame is speech
            energy: Current energy level
        """
        if not COLORAMA_AVAILABLE:
            return
            
        threshold = self._get_threshold()
        width = self.config["vis_width"]
        
        # Scale energy to visualization width
        energy_level = min(int(energy * width * 10), width)
        threshold_pos = min(int(threshold * width * 10), width)
        
        # Choose color based on speech state
        if is_speech:
            bar_color = Fore.GREEN
            label = "SPEECH"
        else:
            bar_color = Fore.BLUE
            label = "silence"
            
        # Create visualization bar
        bar = bar_color + "█" * energy_level + Style.RESET_ALL
        
        # Add threshold marker
        vis = list(" " * width)
        for i in range(min(energy_level, width)):
            vis[i] = "█"
        if threshold_pos < width:
            vis[threshold_pos] = Fore.RED + "▌" + Style.RESET_ALL
            
        # Print with carriage return to overwrite previous line
        state = f"{label} {'[RECORDING]' if self._is_speaking else ''}"
        print(f"\r{state:12} |{''.join(vis)}| {energy:.5f}", end="", flush=True)

    def start_monitoring(self):
        """Start monitoring audio input for speech detection."""
        if self._running:
            logger.warning("Already monitoring audio")
            return False
            
        try:
            # Initialize PyAudio if needed
            if not self._audio:
                self._audio = pyaudio.PyAudio()
                
            # Open audio stream
            self._stream = self._audio.open(
                format=self.config["format"],
                channels=self.config["channels"],
                rate=self.config["sample_rate"],
                input=True,
                frames_per_buffer=self.config["frame_size"],
                stream_callback=self._audio_callback
            )
            
            # Reset state and start stream
            self._reset_state()
            self._running = True
            self._stream.start_stream()
            
            logger.info("Started audio monitoring for speech")
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio monitoring: {e}")
            self.stop_monitoring()
            return False
            
    def stop_monitoring(self):
        """Stop monitoring audio input."""
        self._running = False
        
        # Close stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            self._stream = None
            
        # Close PyAudio
        if self._audio:
            try:
                self._audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            self._audio = None
            
        logger.info("Stopped audio monitoring")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for processing incoming audio frames.
        
        Args:
            in_data: Raw audio data
            frame_count: Number of frames
            time_info: Time information
            status: Status flags
            
        Returns:
            Tuple of (None, flag) as required by PyAudio
        """
        if not self._running:
            return None, pyaudio.paComplete
            
        # Process the audio frame
        self.process_audio_frame(in_data)
        
        # Continue streaming
        return None, pyaudio.paContinue
    
    def get_wav_data(self, audio_buffer):
        """Convert raw audio buffer to WAV format.
        
        Args:
            audio_buffer: Raw audio data as bytes
            
        Returns:
            WAV data as bytes
        """
        try:
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(self.config["channels"])
                wf.setsampwidth(pyaudio.get_sample_size(self.config["format"]))
                wf.setframerate(self.config["sample_rate"])
                wf.writeframes(audio_buffer)
                
            wav_io.seek(0)
            return wav_io.read()
        except Exception as e:
            logger.error(f"Error converting to WAV: {e}")
            return None
            
    def save_wav_file(self, audio_buffer, filepath):
        """Save audio buffer to WAV file.
        
        Args:
            audio_buffer: Raw audio data as bytes
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.config["channels"])
                wf.setsampwidth(pyaudio.get_sample_size(self.config["format"]))
                wf.setframerate(self.config["sample_rate"])
                wf.writeframes(audio_buffer)
                
            return True
        except Exception as e:
            logger.error(f"Error saving WAV file: {e}")
            return False


class SpeechMonitor:
    """High-level speech monitoring system with enhanced VAD."""
    
    def __init__(self, on_speech_start=None, on_speech_end=None, config=None):
        """Initialize speech monitor.
        
        Args:
            on_speech_start: Function to call when speech starts
            on_speech_end: Function to call when speech ends, gets audio buffer
            config: VAD configuration or None to use defaults
        """
        self.vad = EnhancedVAD(config)
        
        # Register callbacks
        self.vad.on_speech_start = on_speech_start
        self.vad.on_speech_end = on_speech_end
        
        # State tracking
        self.is_monitoring = False
        self.pre_buffer_size = 10  # Number of frames to keep in pre-buffer
        self.pre_buffer = collections.deque(maxlen=self.pre_buffer_size)
        
    def start(self):
        """Start speech monitoring."""
        if self.is_monitoring:
            return False
            
        self.is_monitoring = True
        return self.vad.start_monitoring()
        
    def stop(self):
        """Stop speech monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        self.vad.stop_monitoring()


# Global instance for easy access
_speech_monitor = None

def get_speech_monitor(config=None):
    """Get or create the global speech monitor instance.
    
    Args:
        config: Optional VAD configuration
        
    Returns:
        SpeechMonitor instance
    """
    global _speech_monitor
    if _speech_monitor is None:
        _speech_monitor = SpeechMonitor(config=config)
    return _speech_monitor

def start_smart_listening(on_speech_start=None, on_speech_end=None, config=None):
    """Start smart listening with callbacks.
    
    Args:
        on_speech_start: Function to call when speech starts
        on_speech_end: Function to call when speech ends, gets audio buffer
        config: Optional VAD configuration
        
    Returns:
        True if started successfully, False otherwise
    """
    monitor = get_speech_monitor(config)
    monitor.vad.on_speech_start = on_speech_start
    monitor.vad.on_speech_end = on_speech_end
    return monitor.start()

def stop_smart_listening():
    """Stop smart listening."""
    monitor = get_speech_monitor()
    monitor.stop()

# Simple test function
def test_vad():
    """Test the VAD functionality."""
    print("Testing Enhanced VAD...")
    
    # Define callbacks
    def on_speech_start():
        print("\nSpeech started")
        
    def on_speech_end(audio_buffer):
        print("\nSpeech ended", len(audio_buffer), "bytes")
        # Save to temp file
        filename = f"test_speech_{time.time()}.wav"
        vad.save_wav_file(audio_buffer, filename)
        print(f"Saved to {filename}")
    
    # Create VAD with visualization
    config = VAD_CONFIG.copy()
    config["visualize"] = True
    vad = EnhancedVAD(config)
    vad.on_speech_start = on_speech_start
    vad.on_speech_end = on_speech_end
    
    # Start monitoring
    vad.start_monitoring()
    
    print("Listening for speech. Speak now...")
    print("Press Ctrl+C to stop")
    
    try:
        # Run for a set duration or until Ctrl+C
        time.sleep(20)
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        vad.stop_monitoring()
        print("Test complete")

# Run test if module is executed directly
if __name__ == "__main__":
    test_vad()
