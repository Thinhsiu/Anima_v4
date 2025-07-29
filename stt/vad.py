"""
Voice Activity Detection (VAD) module using Silero VAD
Provides real-time speech detection for smart conversation flow
"""
import os
import io
import torch
import time
import queue
import numpy as np
import pyaudio
import threading
import wave
from typing import Optional, Callable, List, Tuple

# Configuration for VAD system
VAD_CONFIG = {
    "sample_rate": 16000,        # Hz (Silero works best at 16kHz)
    "frame_size": 512,           # Audio frame size
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "../models/silero_vad.jit"),
    "threshold": 0.5,            # Speech detection threshold (0-1)
    "min_speech_duration_ms": 250,  # Min speech duration to trigger detection
    "min_silence_duration_ms": 700, # Min silence duration to end speech
    "speech_pad_ms": 300,        # Padding to add after speech ends
    "channels": 1,               # Mono audio
    "format": pyaudio.paInt16,   # Audio format
    
    # Additional parameters for improved detection
    "energy_threshold": 300,     # Minimum audio energy to consider frame for VAD
    "max_silence_sec": 2.0,      # Maximum silence before ending speech capture
    "max_speech_sec": 30.0,      # Maximum speech duration for safety
    "dynamic_energy_adjustment": True, # Adjust energy threshold based on environment
}

# Initialize download directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class SileroVAD:
    """Voice Activity Detection using Silero VAD model"""
    
    def __init__(self, config=None):
        self.config = config or VAD_CONFIG
        self._model = None
        self.sample_rate = self.config["sample_rate"]
        self._ensure_model()
    
    def _ensure_model(self):
        """Download or load the Silero VAD model"""
        model_path = self.config["model_path"]
        
        if os.path.exists(model_path):
            print("Loading existing Silero VAD model...")
            self._model = torch.jit.load(model_path)
            return
            
        print("Downloading Silero VAD model (this will happen only once)...")
        try:
            import urllib.request
            import zipfile
            
            # URLs for the model files
            model_url = "https://github.com/snakers4/silero-vad/archive/refs/tags/v3.1.zip"
            zip_path = os.path.join(MODELS_DIR, "silero_vad.zip")
            
            # Download the model
            urllib.request.urlretrieve(model_url, zip_path)
            
            # Extract the model
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(MODELS_DIR)
            
            # Load and save the model in JIT format
            import sys
            sys.path.append(os.path.join(MODELS_DIR, 'silero-vad-3.1'))
            
            model, utils = torch.hub.load(
                repo_or_dir=os.path.join(MODELS_DIR, 'silero-vad-3.1'),
                model='silero_vad',
                source='local',
                force_reload=True,
                onnx=False
            )
            
            # Convert to JIT format for faster loading
            model.eval()
            example_input = torch.randn(1, 8000)  # 0.5 second at 16kHz
            traced_model = torch.jit.trace(model, example_input)
            torch.jit.save(traced_model, model_path)
            
            self._model = traced_model
            
            # Cleanup
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            print("Silero VAD model downloaded and saved successfully.")
            
        except Exception as e:
            print(f"Error downloading Silero VAD model: {e}")
            # Fallback: Try direct torch hub load
            try:
                self._model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=True,
                    onnx=False
                )
                # Save for future use
                torch.jit.save(self._model, model_path)
                print("Downloaded VAD model directly through torch hub.")
            except Exception as e2:
                print(f"Failed to download model through fallback: {e2}")
                raise
    
    def is_speech(self, audio_chunk):
        """
        Detect if an audio chunk contains speech
        
        Args:
            audio_chunk: Raw audio bytes at the configured format
            
        Returns:
            bool: True if speech detected, False otherwise
        """
        if self._model is None:
            return False
            
        try:
            # Convert audio bytes to float tensor
            waveform = self._bytes_to_tensor(audio_chunk)
            
            # Get speech probability
            # Pass both the waveform and sample rate to the model
            with torch.no_grad():
                speech_prob = self._model(waveform, self.sample_rate).item()
            
            return speech_prob >= self.config["threshold"]
        except Exception as e:
            print(f"VAD error: {e}")
            return False
    
    def _bytes_to_tensor(self, audio_bytes):
        """Convert audio bytes to PyTorch tensor"""
        # Convert to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Normalize to float between -1 and 1
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Resample if needed (PyAudio sample rate -> VAD model sample rate)
        # For simplicity assuming same rate here, add resampling if needed
        
        # Convert to PyTorch tensor with correct shape [1, samples]
        return torch.FloatTensor(audio_float).unsqueeze(0)


class SpeechMonitor:
    """Real-time speech monitor using VAD for automatic mic control"""
    
    def __init__(self, 
                 on_speech_start=None, 
                 on_speech_end=None,
                 config=None):
        """
        Initialize speech monitor
        
        Args:
            on_speech_start: Function to call when speech starts
            on_speech_end: Function to call when speech ends
            config: Optional config override
        """
        self.config = config or VAD_CONFIG
        self.vad = SileroVAD(self.config)
        
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        
        self._audio = None
        self._stream = None
        self._running = False
        self._speech_detected = False
        self._last_detection_time = 0
        self._audio_queue = queue.Queue()
        self._monitor_thread = None
        
        # State tracking
        self.is_recording = False
        self.recording_buffer = b''
    
    def start(self):
        """Start speech monitoring"""
        if self._running:
            return
            
        self._running = True
        self._speech_detected = False
        self._audio = pyaudio.PyAudio()
        
        # Start audio stream
        self._stream = self._audio.open(
            format=self.config["format"],
            channels=self.config["channels"],
            rate=self.config["sample_rate"],
            input=True,
            frames_per_buffer=self.config["frame_size"],
            stream_callback=self._audio_callback
        )
        
        # Start monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            daemon=True
        )
        self._monitor_thread.start()
        
        print("🎧 Smart listening activated (VAD)...")
    
    def stop(self):
        """Stop speech monitoring"""
        if not self._running:
            return
            
        self._running = False
        
        # Stop audio stream
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
                
        # Wait for monitor thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
            
        # Clean up PyAudio
        if self._audio:
            try:
                self._audio.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
                
        self._audio = None
        self._stream = None
        
        print("🔇 Smart listening deactivated")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback - receives audio data from microphone
        """
        if self._running:
            self._audio_queue.put(in_data)
            
            # Add to recording buffer if we're actively recording
            if self.is_recording:
                self.recording_buffer += in_data
                
        return (in_data, pyaudio.paContinue)
    
    def _monitor_loop(self):
        """Background thread that processes audio frames and detects speech with enhanced energy filtering"""
        
        # Dynamic energy threshold adjustment variables
        energy_adjustment_damping = 0.95  # Smoothing factor
        energy_threshold = self.config["energy_threshold"]
        ambient_energy = energy_threshold * 0.5  # Initial ambient energy estimate
        
        # Speech duration tracking
        speech_start_time = 0
        total_silent_frames = 0
        silence_streak = 0
        
        while self._running:
            try:
                # Get audio frame from queue
                try:
                    audio_data = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                
                # Calculate energy of the frame
                frame_energy = np.abs(audio_array).mean() * 32767.0  # Scale back to int16 range
                
                # Update ambient energy estimate
                if self.config["dynamic_energy_adjustment"] and not self._speech_detected:
                    # Only adjust when not in active speech
                    ambient_energy = ambient_energy * energy_adjustment_damping + \
                                     frame_energy * (1 - energy_adjustment_damping)
                    # Dynamic threshold is ambient + margin
                    energy_threshold = max(self.config["energy_threshold"], ambient_energy * 2.0)
                
                # Skip low energy frames for efficiency and noise reduction
                if frame_energy < energy_threshold * 0.8:  # Small margin below threshold
                    if self._speech_detected:
                        silence_streak += 1
                    total_silent_frames += 1
                    if total_silent_frames > 100:  # Reset ambient occasionally during long silence
                        ambient_energy = energy_threshold * 0.5
                        total_silent_frames = 0
                    # Only pass every few low-energy frames to the VAD for efficiency
                    if total_silent_frames % 3 != 0 and not self._speech_detected:
                        continue
                else:
                    silence_streak = 0
                    total_silent_frames = 0
                
                # Run VAD on the audio frame
                speech_prob = self.vad.is_speech(audio_data)
                
                # Check if speech was detected in this frame
                is_speech = speech_prob > self.config["threshold"]
                
                # Safety check for maximum speech duration
                if self._speech_detected:
                    speech_duration = time.time() - speech_start_time
                    if speech_duration > self.config["max_speech_sec"]:
                        print("Maximum speech duration reached, ending capture")
                        self._speech_detected = False
                        self.is_recording = False
                        if self.on_speech_end and len(self.recording_buffer) > 0:
                            self.on_speech_end(self.recording_buffer)
                        continue
                
                # Speech start detection
                if is_speech and not self._speech_detected and frame_energy > energy_threshold:
                    self._speech_detected = True
                    self._last_detection_time = time.time()
                    speech_start_time = time.time()  # Track when speech started
                    self.recording_buffer = b''  # Clear previous recording
                    
                    # Start recording
                    self.is_recording = True
                    
                    # Log speech detection with energy level
                    print(f" Speech detected (energy: {frame_energy:.1f}, threshold: {energy_threshold:.1f})")
                    
                    # Call callback if provided
                    if self.on_speech_start:
                        self.on_speech_start()
                
                # Maintain speech detection state
                if is_speech and frame_energy > energy_threshold * 0.7:  # Lower threshold for continuing speech
                    self._last_detection_time = time.time()
                    silence_streak = 0
                
                # Accumulate audio while recording
                if self.is_recording:
                    self.recording_buffer += audio_data
                
                # Speech end detection - either by VAD silence or consecutive low energy frames
                silence_duration_ms = int((time.time() - self._last_detection_time) * 1000)
                if self._speech_detected and (
                    silence_duration_ms > self.config["min_silence_duration_ms"] or
                    silence_streak > int(self.config["max_silence_sec"] * 1000 / 50)  # 50ms frames
                ):
                    self._speech_detected = False
                    self.is_recording = False
                    
                    # Add a small padding to the end for better results
                    padding_ms = self.config["speech_pad_ms"]
                    padding_frames = int(padding_ms * self.config["sample_rate"] / 1000 / self.config["frame_size"]) 
                    for _ in range(min(padding_frames, 5)):  # Cap at 5 frames max
                        try:
                            padding_data = self._audio_queue.get(timeout=0.1)
                            self.recording_buffer += padding_data
                        except queue.Empty:
                            break
                    
                    # Log speech end
                    duration = time.time() - speech_start_time
                    print(f" Speech ended after {duration:.1f} seconds")
                    
                    # Call callback with the complete recording
                    if self.on_speech_end and len(self.recording_buffer) > 0:
                        self.on_speech_end(self.recording_buffer)
            
            except Exception as e:
                print(f"Error in speech monitor: {e}")
                time.sleep(0.1)  # Avoid tight loop on errors
    
    def get_wav_data(self, audio_data):
        """
        Convert raw audio data to WAV format
        """
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav:
                wav.setnchannels(self.config["channels"])
                wav.setsampwidth(pyaudio.get_sample_size(self.config["format"]))
                wav.setframerate(self.config["sample_rate"])
                wav.writeframes(audio_data)
            return wav_io.getvalue()

# Global instance for easy access
_speech_monitor = None

def get_speech_monitor():
    """Get or create the global speech monitor instance"""
    global _speech_monitor
    if _speech_monitor is None:
        _speech_monitor = SpeechMonitor()
    return _speech_monitor

def start_smart_listening(on_speech_start=None, on_speech_end=None):
    """Start smart listening with callbacks"""
    monitor = get_speech_monitor()
    monitor.on_speech_start = on_speech_start
    monitor.on_speech_end = on_speech_end
    monitor.start()
    return monitor

def stop_smart_listening():
    """Stop smart listening"""
    monitor = get_speech_monitor()
    monitor.stop()
