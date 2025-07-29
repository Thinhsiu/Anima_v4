"""
Streaming Speech-to-Text using Faster-Whisper

This module provides real-time streaming speech recognition using Faster-Whisper,
a more efficient implementation of OpenAI's Whisper model.
"""

import os
import time
import tempfile
import threading
import queue
import numpy as np
import torch
import wave
from typing import Optional, Callable, List, Dict, Any, Iterator
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("faster_whisper_streaming")

# Try importing faster_whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("faster-whisper not available. Run 'pip install faster-whisper'")
    FASTER_WHISPER_AVAILABLE = False

# Audio settings for consistent processing
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT_BYTES_PER_SAMPLE = 2  # 16-bit audio

# Whisper model configuration (can be overridden)
WHISPER_CONFIG = {
    "model_size": "small",       # Options: tiny, base, small, medium, large-v1, large-v2, large-v3
    "device": "auto",           # Options: cpu, cuda, auto
    "compute_type": "default",   # Options: default, float16, int8
    "cpu_threads": 4,           # Number of CPU threads to use
    "download_root": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
    "language": "en",           # Default language
    "beam_size": 5,             # Beam size for faster inference
    "vad_filter": True,         # Apply VAD filter to remove silence
    "initial_prompt": "",       # Optional prompt to guide transcription
    "streaming_chunk_size": 24,  # Audio chunk size in streaming mode (in model frames)
}

class WhisperStreamingSTT:
    """Streaming Speech-to-Text using Faster-Whisper with real-time feedback."""
    
    def __init__(self, config=None):
        """Initialize the WhisperStreamingSTT with configuration.
        
        Args:
            config: Configuration dictionary or None to use defaults
        """
        self.config = config or WHISPER_CONFIG.copy()
        self._model = None
        self._streaming_queue = queue.Queue()
        self._audio_buffer = []
        self._transcript_buffer = []
        self._transcript_text = ""
        self._is_streaming = False
        self._streaming_thread = None
        self._on_partial_transcript = None
        self._on_final_transcript = None
        self._current_segment_id = 0
        self._load_model()

    def _load_model(self):
        """Load the Whisper model with specified configuration."""
        if not FASTER_WHISPER_AVAILABLE:
            logger.error("Faster-Whisper is not available. Cannot load model.")
            return False
        
        try:
            device = self.config.get("device", "auto")
            compute_type = self.config.get("compute_type", "default")
            model_size = self.config.get("model_size", "small")
            download_root = self.config.get("download_root", None)
            cpu_threads = self.config.get("cpu_threads", 4)
            
            # Determine if CUDA is available and set device accordingly
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Determine compute type based on device
            if compute_type == "default":
                compute_type = "float16" if device == "cuda" else "int8"
                
            logger.info(f"Loading Faster-Whisper {model_size} model on {device} with {compute_type}...")
            
            # Create the model with optimal settings for the device
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
                cpu_threads=cpu_threads,
            )
            
            logger.info(f"Whisper {model_size} model loaded successfully on {device}")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return False

    def setup_callbacks(self, on_partial_transcript=None, on_final_transcript=None):
        """Register callbacks for partial and final transcription results.
        
        Args:
            on_partial_transcript: Callback for interim results
            on_final_transcript: Callback for final results
        """
        self._on_partial_transcript = on_partial_transcript
        self._on_final_transcript = on_final_transcript

    def _get_transcription_options(self):
        """Get options for the transcription process."""
        return {
            "language": self.config.get("language", "en"),
            "beam_size": self.config.get("beam_size", 5),
            "initial_prompt": self.config.get("initial_prompt", ""),
            "vad_filter": self.config.get("vad_filter", True),
            "vad_parameters": {
                "min_silence_duration_ms": 500,  # Don't include silences longer than this
            },
        }

    def start_streaming(self):
        """Start streaming audio processing in a background thread."""
        if self._is_streaming:
            logger.warning("Streaming already active")
            return False
        
        if not self._model:
            logger.error("Model not loaded")
            return False
        
        self._is_streaming = True
        self._audio_buffer = []
        self._transcript_buffer = []
        self._transcript_text = ""
        self._current_segment_id = 0
        
        # Start processing thread
        self._streaming_thread = threading.Thread(target=self._streaming_worker)
        self._streaming_thread.daemon = True
        self._streaming_thread.start()
        
        logger.info("Streaming STT started")
        return True

    def stop_streaming(self):
        """Stop streaming and process any remaining audio."""
        if not self._is_streaming:
            return
        
        self._is_streaming = False
        
        # Process any remaining audio
        if self._audio_buffer:
            self._process_final_transcript()
        
        # Wait for thread to finish
        if self._streaming_thread:
            self._streaming_thread.join(timeout=2.0)
            self._streaming_thread = None
            
        logger.info("Streaming STT stopped")
    
    def _streaming_worker(self):
        """Background thread that processes audio chunks and generates transcripts."""
        while self._is_streaming:
            try:
                # Wait for audio data with a timeout
                try:
                    audio_chunk = self._streaming_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Add to buffer
                self._audio_buffer.append(audio_chunk)
                
                # Process when buffer reaches substantial size (1 second of audio)
                buffer_size = len(self._audio_buffer) * CHUNK_SIZE
                if buffer_size >= SAMPLE_RATE:
                    self._process_partial_transcript()
                    
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
                break
    
    def _process_partial_transcript(self):
        """Process accumulated audio buffer and update with partial transcript."""
        if not self._audio_buffer:
            return
        
        try:
            # Convert audio buffer to numpy array
            audio_data = np.concatenate(self._audio_buffer)
            
            # Get partial transcription
            options = self._get_transcription_options()
            segments_iterator = self._model.transcribe(
                audio_data, 
                word_timestamps=True,
                **options
            )
            
            # Process segments
            partial_text = ""
            for segment in segments_iterator:
                # Update the transcript buffer with new segment
                segment_id = self._current_segment_id
                self._current_segment_id += 1
                
                # Extract segment data
                segment_data = {
                    "id": segment_id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [(word.word, word.start, word.end) for word in segment.words],
                    "confidence": segment.avg_logprob
                }
                
                # Add to buffer and update text
                self._transcript_buffer.append(segment_data)
                partial_text += segment_data["text"] + " "
            
            # Update current transcript
            if partial_text:
                self._transcript_text = partial_text.strip()
                
                # Trigger callback if available
                if self._on_partial_transcript:
                    self._on_partial_transcript(self._transcript_text)
                    
        except Exception as e:
            logger.error(f"Error processing partial transcript: {e}")
    
    def _process_final_transcript(self):
        """Process the complete audio and generate final transcript."""
        if not self._audio_buffer:
            return
        
        try:
            # Convert audio buffer to numpy array
            audio_data = np.concatenate(self._audio_buffer)
            
            # Get final transcription with high-quality settings
            options = self._get_transcription_options()
            segments, _ = self._model.transcribe(
                audio_data, 
                beam_size=options.get("beam_size", 5) + 1,  # Use larger beam for final
                word_timestamps=True,
                **options
            )
            
            # Collect all segment texts
            final_text = ""
            for segment in segments:
                final_text += segment.text + " "
            
            # Clean up and set final result
            final_text = final_text.strip()
            self._transcript_text = final_text
            
            # Clear buffers
            self._audio_buffer = []
            
            # Trigger callback if available
            if self._on_final_transcript:
                self._on_final_transcript(final_text)
            
            return final_text
            
        except Exception as e:
            logger.error(f"Error processing final transcript: {e}")
            return self._transcript_text  # Return whatever we have so far
    
    def add_audio_chunk(self, audio_chunk):
        """Add a chunk of audio data to the streaming queue.
        
        Args:
            audio_chunk: Audio data as numpy array or bytes
        """
        if not self._is_streaming:
            return False
        
        # Convert bytes to numpy if needed
        if isinstance(audio_chunk, bytes):
            # Convert 16-bit PCM bytes to numpy float32 array
            chunk_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_chunk, np.ndarray):
            chunk_np = audio_chunk
        else:
            logger.error(f"Unsupported audio chunk type: {type(audio_chunk)}")
            return False
        
        # Add to queue
        self._streaming_queue.put(chunk_np)
        return True
    
    def get_current_transcript(self):
        """Get the current transcript text."""
        return self._transcript_text
    
    def transcribe_file(self, file_path):
        """Transcribe audio from a file (non-streaming).
        
        Args:
            file_path: Path to audio file
        
        Returns:
            Transcribed text
        """
        if not self._model:
            logger.error("Model not loaded")
        try:
            # Log durations to debug short audio segments
            import librosa
            try:
                duration = librosa.get_duration(path=file_path)
                logger.info(f"Processing audio with duration {duration:.2f}s")
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {e}")
            
            # Get all segments and combine them
            segments, info = self._model.transcribe(file_path, beam_size=5)
            segments_list = list(segments)  # Convert generator to list
            
            if segments_list:
                # Combine all segments into one transcript
                combined_text = ' '.join([segment.text for segment in segments_list])
                return combined_text.strip()
            else:
                logger.warning("No speech segments detected in audio")
                return None
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return None

    def cleanup(self):
        """Clean up resources used by this instance."""
        try:
            # Stop any ongoing streaming
            self.stop_streaming()
            
            # Clear buffers and queues
            self._streaming_queue = queue.Queue()
            self._audio_buffer = []
            self._transcript_buffer = []
            self._transcript_text = ""
            
            # Unregister callbacks
            self._on_partial_transcript = None
            self._on_final_transcript = None
            
            # Clear the model if it's using GPU memory
            if self._model is not None and self.config.get('device', '').startswith('cuda'):
                try:
                    # Free up CUDA memory
                    torch.cuda.empty_cache()
                    logger.info("CUDA memory cache emptied")
                except Exception as e:
                    logger.warning(f"Failed to clear CUDA memory: {e}")
            
            logger.info("WhisperStreamingSTT resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def transcribe_buffer(self, buffer_data):
        """Transcribe audio from a bytes buffer.
        
        Args:
            buffer_data: Audio buffer as bytes or numpy array
        
        Returns:
            Transcribed text
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # Always create a proper WAV file with headers
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(FORMAT_BYTES_PER_SAMPLE)
                wf.setframerate(SAMPLE_RATE)
                
                # Process different input types
                if isinstance(buffer_data, bytes):
                    # Raw PCM bytes from VAD - assume 16-bit audio
                    wf.writeframes(buffer_data)
                elif isinstance(buffer_data, np.ndarray):
                    # Convert numpy array to 16-bit PCM
                    wf.writeframes((buffer_data * 32768).astype(np.int16).tobytes())
                else:
                    logger.error(f"Unsupported buffer type: {type(buffer_data)}")
                    return None
                
            # Transcribe the temp file
            result = self.transcribe_file(temp_filename)
            
            # Clean up
            try:
                os.unlink(temp_filename)
            except:
                pass
                
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing from buffer: {e}")
            return None


# Create a global instance for easy access
_whisper_streaming_stt = None

def get_streaming_stt():
    """Get or create the global WhisperStreamingSTT instance."""
    global _whisper_streaming_stt
    if _whisper_streaming_stt is None:
        _whisper_streaming_stt = WhisperStreamingSTT()
    return _whisper_streaming_stt

def transcribe_file(file_path, language=None):
    """Transcribe audio from a file path.
    
    Args:
        file_path: Path to audio file
        language: Language code (optional)
    
    Returns:
        Transcribed text
    """
    stt = get_streaming_stt()
    options = stt._get_transcription_options()
    if language:
        options["language"] = language
    return stt.transcribe_file(file_path)

def transcribe_buffer(buffer_data, language=None):
    """Transcribe audio from a buffer.
    
    Args:
        buffer_data: Audio buffer data
        language: Language code (optional)
    
    Returns:
        Transcribed text
    """
    stt = get_streaming_stt()
    options = stt._get_transcription_options()
    if language:
        options["language"] = language
    return stt.transcribe_buffer(buffer_data)

# Simple test function
def test_streaming_stt():
    """Test the streaming STT functionality."""
    import pyaudio
    
    print("Testing streaming STT...")
    
    # Initialize STT
    stt = get_streaming_stt()
    
    if not stt._model:
        print("Failed to load model. Exiting test.")
        return
        
    # Setup callbacks
    def on_partial(text):
        print(f"\rPartial: {text}", end="", flush=True)
        
    def on_final(text):
        print(f"\nFinal: {text}")
    
    stt.setup_callbacks(on_partial, on_final)
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Start streaming
    stt.start_streaming()
    
    # Open stream for 5 seconds
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    
    print("\nListening... Speak now. (5 seconds)")
    for _ in range(int(SAMPLE_RATE / CHUNK_SIZE * 5)):
        audio_data = stream.read(CHUNK_SIZE)
        stt.add_audio_chunk(audio_data)
        
    # Close stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Stop streaming and get final result
    stt.stop_streaming()
    final_text = stt.get_current_transcript()
    print(f"Test complete. Final transcript: {final_text}")

# Run test if this module is executed directly
if __name__ == "__main__":
    test_streaming_stt()
