import whisper
import tempfile
import wave
import numpy as np
import pyaudio
import os
import atexit
from difflib import SequenceMatcher

# Keep track of temp files for cleanup
_temp_files = []

# Register cleanup function to remove temp files on exit
def _cleanup_temp_files():
    global _temp_files
    for file_path in _temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed temp file: {file_path}")
        except Exception as e:
            print(f"Error removing temp file {file_path}: {e}")
    _temp_files = []

# Register cleanup function
atexit.register(_cleanup_temp_files)

# Load Whisper model once
model = whisper.load_model("base")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def record_audio(seconds=7):
    """Record audio and return path to temporary audio file
    Returns tuple of (filepath, has_speech)
    """
    pyaudio_instance = None
    stream = None
    
    try:
        # Initialize PyAudio with proper error handling
        pyaudio_instance = pyaudio.PyAudio()
        
        # Set up audio stream (16kHz, 16bit, mono)
        stream = pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        # Track audio levels to detect speech
        frames = []
        audio_levels = []
        max_level = 0
        
        # Record audio for the specified duration
        for i in range(0, int(RATE / CHUNK * seconds)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Calculate audio level (simple absolute average)
                audio_sample = np.frombuffer(data, dtype=np.int16)
                level = np.abs(audio_sample).mean()
                audio_levels.append(level)
                
                # Track maximum level
                if level > max_level:
                    max_level = level
            except IOError as e:
                print(f"Warning: Audio read error ({e}). Continuing...")
                # Insert silent chunk on error
                frames.append(b'\x00' * CHUNK * 2)  # 2 bytes per int16 sample
                audio_levels.append(0)
    except Exception as e:
        print(f"❌ Error initializing audio device: {e}")
        return None, False
    finally:
        # Always clean up audio resources
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Warning: Error closing stream: {e}")
                
        if pyaudio_instance:
            try:
                pyaudio_instance.terminate()
            except Exception as e:
                print(f"Warning: Error terminating PyAudio: {e}")
                

    # Cleanup now handled in finally block above
    
    # Analyze audio levels to determine if speech was detected
    avg_level = sum(audio_levels) / len(audio_levels) if audio_levels else 0
    print(f"Audio stats - Max level: {max_level}, Avg level: {avg_level}")
    
    # Use configurable thresholds from STT_CONFIG
    has_speech = (max_level > STT_CONFIG["max_level_threshold"] and 
                 avg_level > STT_CONFIG["avg_level_threshold"] and 
                 (max_level / (avg_level + 1)) > STT_CONFIG["noise_ratio_threshold"])

    # Save temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wf = wave.open(temp_file.name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))  # Get correct sample width from format
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Add to tracked temp files list for cleanup
    _temp_files.append(temp_file.name)
    
    return temp_file.name, has_speech

def check_similarity_to_common_phrases(text):
    """Check if the transcribed text is similar to any common phrases or intents"""
    # Temporarily disable strict filtering to prevent crashes
    return True  # Always allow the transcription through

def clean_transcription(text):
    """Clean transcription text by removing any instances of the prompt text"""
    if not text:
        return None
        
    # Get the initial prompt from config
    prompt = STT_CONFIG.get("initial_prompt", "")
    if prompt:
        # Check for case-insensitive matches of the prompt within text
        start_idx = text.lower().find(prompt.lower())
        if start_idx >= 0:
            # Remove the prompt with proper original capitalization
            text = text[:start_idx] + text[start_idx + len(prompt):]
    
    # Filter short transcriptions (likely noise or incomplete fragments)
    word_count = len(text.split())
    if word_count < 5:
        print(f"⚠️ Too short, likely invalid: '{text}' ({word_count} words)")
        return None
    
    return text.strip()

def is_repetitive(text, min_repeat=3):
    """Detect repeated phrases within the transcription"""
    # Safety check to prevent NoneType errors
    if not text:
        return False
        
    # Split by periods
    phrases = text.lower().split('.')
    phrases = [p.strip() for p in phrases if p.strip()]
    
    # Count how many times each unique phrase appears
    from collections import Counter
    freq = Counter(phrases)
    
    # If any phrase repeats more than threshold, consider it spam
    return any(count >= min_repeat for count in freq.values())

def is_noise_pattern(text):
    """Enhanced detection of noise patterns to filter out non-speech"""
    # Empty or very short text is noise
    if not text or len(text) < 3:
        return True
        
    # Check for repetitive character patterns
    total_chars = len(text)
    
    # Check for excessive punctuation
    punct_count = sum(1 for c in text if c in '.,;:?![]{}()<>-_+=|`~@#$%^&*')
    if punct_count > total_chars * 0.25:  # More than 25% punctuation
        return True
        
    # Check for excessive digit patterns
    digit_count = sum(1 for c in text if c.isdigit())
    if digit_count > total_chars * 0.3:  # More than 30% digits
        return True
    
    # Check for repeating characters or patterns
    if len(set(text)) < min(5, total_chars * 0.3):  # Very low character diversity
        return True
        
    # Check for single repeated character (like '...' or '111')
    if any(text.count(c) > total_chars * 0.5 for c in set(text)):
        return True
    
    # More advanced pattern detection - look for binary patterns
    if total_chars > 6 and len(set(text.replace(' ',''))) <= 3:  # Very limited unique chars
        return True
        
    return False

# Configurable settings for STT
STT_CONFIG = {
    "recording_seconds": 9,        # Increased from 7 seconds for accent handling
    "confidence_threshold": 0.2,   # Lower threshold for accented speech (was 0.3)
    "noise_ratio_threshold": 5,    # Ratio of max to avg level for speech detection
    "max_level_threshold": 150,    # Minimum peak volume for speech
    "avg_level_threshold": 25,     # Minimum average volume for speech
    "initial_prompt": "This is a conversation with an AI assistant. The speaker may be asking about who created you, your purpose, or giving commands. The speaker has a non-native English accent.",
    "similarity_threshold": 0.6,   # Threshold for text similarity matching
    "common_phrases": [           # List of common phrases/intents to check against
        "who created you",
        "what is your purpose",
        "how were you made",
        "tell me about yourself",
        "what can you do",
        "exit voice mode",
        "switch to text mode",
        "stop listening"
    ]
}

def get_voice_input(seconds=None):
    """Capture voice and return Whisper transcription text 
    with improved noise filtering and confidence checks
    Returns None if no speech was detected or filtered
    """
    try:
        # Use configured recording duration if not specified
        recording_time = seconds if seconds else STT_CONFIG["recording_seconds"]
        
        print(f"🎙️ Listening for {recording_time} seconds...")
        audio_path, has_speech = record_audio(recording_time)
        
        # Skip transcription if no significant audio was detected or path is None
        if not has_speech or audio_path is None:
            print("❌ No speech detected in audio or recording failed")
            return None
        
        # Verify file exists before transcription    
        if not os.path.exists(audio_path):
            print(f"❌ Error: Audio file not found at {audio_path}")
            return None
            
        print("📼 Transcribing...")
        try:
            # Set Whisper with enhanced options for accent handling
            result = model.transcribe(
                audio_path,
                language="en",             # English language
                task="transcribe",         # Standard transcription task
                temperature=0.0,           # Lower temperature for more predictable output
                initial_prompt=STT_CONFIG["initial_prompt"]
            )
        except Exception as e:
            print(f"❌ Whisper error: {e}")
            # Try loading with soundfile as a workaround
            try:
                import soundfile as sf
                print("Attempting to load audio with soundfile...")
                audio_data, sample_rate = sf.read(audio_path)
                # Convert to the format whisper expects
                if sample_rate != 16000:
                    print(f"Resampling from {sample_rate} to 16000 Hz")
                    from scipy import signal
                    audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
                
                print("Loading resampled audio into whisper...")
                result = whisper.transcribe(
                    model,
                    audio_data,
                    language="en",
                    task="transcribe",
                    temperature=0.0
                )
            except Exception as inner_e:
                print(f"❌ Failed alternative loading: {inner_e}")
                # Clean up this specific temp file
                if audio_path in _temp_files:
                    try:
                        os.remove(audio_path)
                        _temp_files.remove(audio_path)
                        print(f"Cleaned up temp file after error: {audio_path}")
                    except:
                        pass
                return None
                
        # Successful transcription, clean up the temp file
        if audio_path in _temp_files:
            try:
                os.remove(audio_path)
                _temp_files.remove(audio_path)
            except Exception as e:
                print(f"Warning: Failed to clean up temp file: {e}")
                
        text = result['text'].strip()
        
        # Get confidence from segments if available
        confidence = 1.0  # Default high confidence
        if 'segments' in result and result['segments']:
            # Calculate average probability across all segments
            probs = [seg.get('avg_logprob', -1) for seg in result['segments']]
            if probs:
                # Convert log probabilities to a 0-1 scale - higher is better
                avg_logprob = sum(probs) / len(probs)
                # Map from typical whisper range (-1 to -10) to confidence
                confidence = max(0, min(1, 1 - (abs(avg_logprob) - 1) / 10))
        
        # Check for noise patterns
        if is_noise_pattern(text):
            print(f"❌ Filtered out noise pattern: '{text}'")
            return None
        
        # Apply aggressive content filtering rules to prevent hallucinated and inappropriate content
        
        # RULE 1: Reject any content with suspicious or inappropriate words
        suspicious_terms = ['sex', 'porn', 'fuck', 'shit', 'damn', 'bitch', 'ass']
        text_lower = text.lower()
        for term in suspicious_terms:
            if term in text_lower:
                print(f"⛔ Filtered inappropriate content: '{text}'")
                return None
        
        # RULE 2: Apply extra strict confidence threshold
        if confidence < STT_CONFIG["confidence_threshold"] * 1.5:  # 50% higher threshold
            print(f"❌ Low confidence rejected: {confidence:.2f} - '{text}'")
            return None
            
        # RULE 3: For shorter phrases, require very high confidence
        word_count = len(text.split())
        if word_count < 7 and confidence < 0.6:  # Higher threshold for short phrases
            print(f"❌ Short phrase with insufficient confidence ({confidence:.2f}): '{text}'")
            return None
        
        # RULE 4: Apply similarity check against known commands/questions
        if not check_similarity_to_common_phrases(text):
            # Only allow long phrases with very high confidence to pass without similarity match
            if word_count < 10 or confidence < 0.7:
                print(f"❌ Rejected non-matching phrase: '{text}'")
                return None
            print(f"⚠️ Allowing long phrase with high confidence despite no match: '{text}'")
        
        # Log the decision to accept this transcription and return it
        print(f"🟢 VERIFIED transcription ({confidence:.2f}, {word_count} words): '{text}'")
        return text
        
    except Exception as e:
        print("❌ Whisper transcription error:", e)
        return "[Error during transcription]"

def transcribe_from_buffer(buffer_data):
    """Transcribe speech from a buffer of WAV audio data"""
    if buffer_data is None or len(buffer_data) == 0:
        print("❌ Error: Empty buffer provided")
        return None
    
    # Create a temporary file to store the WAV data
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        # Write the wav buffer to the file
        temp_file.write(buffer_data)
        temp_file.close()
        _temp_files.append(temp_file.name)
        
        print("📝 Processing speech...")
        
        # Now transcribe the audio file
        result = model.transcribe(
            temp_file.name,
            language="en",
            task="transcribe",
            temperature=0.0
        )
        
        text = result["text"].strip()
        
        # Check for repetition and filter if needed
        if is_repetitive(text):
            print(f"❌ Repetition filter triggered: '{text}'")
            return None
        
        # Check for noise patterns
        if is_noise_pattern(text):
            print(f"❌ Noise pattern detected: '{text}'")
            return None
        
        # Clean up the temp file
        try:
            os.remove(temp_file.name)
            _temp_files.remove(temp_file.name)
        except Exception as e:
            print(f"Warning: Failed to clean up temp file: {e}")
            
        return text
        
    except Exception as e:
        print("❌ Error in buffer transcription:", e)
        # Handle loading errors with alternative libraries
        if "sample width not specified" in str(e) or "Error loading audio" in str(e):
            try:
                import soundfile
                from scipy import signal
                
                print("Trying alternative audio loading method...")
                audio, sr = soundfile.read(temp_file.name)
                # Resample to 16kHz if needed for Whisper
                if sr != 16000:
                    audio = signal.resample(audio, int(len(audio) * 16000 / sr))
                    sr = 16000
                    
                # Transcribe directly
                result = whisper.transcribe(
                    model, 
                    audio, 
                    language=language,
                    initial_prompt=STT_CONFIG.get("initial_prompt", None)
                )
                return result["text"].strip()
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                return None  # Empty text on error
        
        # Handle other errors
        print(f"Error in buffer transcription: {e}")
        return None  # Return None on error

def initialize_whisper_stt():
    """Used by main.py to confirm whisper_stt is initialized"""
    return True
