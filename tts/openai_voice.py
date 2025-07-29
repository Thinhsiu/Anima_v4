import os
import time
import pygame
import tempfile
import openai
import hashlib
import threading
import requests
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Cache directory for storing generated audio
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "voice")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(text, voice="shimmer"):
    """Generate a cache file path based on the text content"""
    # Create a deterministic hash from the text and voice
    text_hash = hashlib.md5(f"{text}_{voice}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{text_hash}.mp3")

def initialize_api():
    """Initialize the OpenAI API with key"""
    try:
        # Try multiple ways to load the API key
        
        # Method 1: Check if it's already set in openai.api_key
        if hasattr(openai, 'api_key') and openai.api_key:
            return True
            
        # Method 2: Try loading from .env file in various locations
        try:
            from dotenv import load_dotenv
            
            # Try current directory first
            load_dotenv()
            
            # Try project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            load_dotenv(os.path.join(project_root, '.env'))
            
            # Try different key names used in the project
            for key_name in ["OPENAI_API_KEY", "OPENAI_35_API_KEY", "OPENAI_4_API_KEY"]:
                api_key = os.getenv(key_name)
                if api_key:
                    openai.api_key = api_key
                    return True
        except ImportError:
            pass  # dotenv not installed
            
        # Method 3: Check if API key is in environment variables directly
        for key_name in ["OPENAI_API_KEY", "OPENAI_35_API_KEY", "OPENAI_4_API_KEY"]:
            api_key = os.environ.get(key_name)
            if api_key:
                openai.api_key = api_key
                return True
            
        # Method 4: Check for API key in main.py's globals (if extracted there)
        try:
            import sys
            main_module = sys.modules.get('__main__')
            if main_module and hasattr(main_module, 'openai_api_key'):
                openai.api_key = main_module.openai_api_key
                return True
        except:
            pass
            
        # If we got here, no API key found
        print("Error: OPENAI_API_KEY not found. Please add it to your .env file or set it in your environment.")
        print("Checked: Current env vars, .env file in current directory and project root.")
        return False
    except Exception as e:
        print(f"Error initializing OpenAI API: {e}")
        return False

# Global thread pool for async operations
_thread_pool = ThreadPoolExecutor(max_workers=4)

# Global variables for async processing
_is_generating = False
_speech_queue = []  # Queue for upcoming speech chunks
_queue_lock = threading.Lock()  # Lock for thread-safe queue operations
_preload_event = threading.Event()  # Event to signal when preloading should start

def generate_speech(text, voice="shimmer", speed=1.0, max_timeout=30, max_retries=3):
    """Generate speech using OpenAI's API with retry logic and exponential backoff"""
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "tts-1",
        "voice": voice,
        "input": text,
        "speed": speed
    }
    
    # Retry with exponential backoff
    retry_count = 0
    base_timeout = 10  # Start with 10 seconds timeout
    
    while retry_count <= max_retries:
        current_timeout = min(base_timeout * (2 ** retry_count), max_timeout)
        
        try:
            # Suppressed console message
            # print(f"API request attempt {retry_count+1}/{max_retries+1} (timeout: {current_timeout}s)")
            response = requests.post(url, headers=headers, json=data, timeout=current_timeout)
            
            if response.status_code != 200:
                error_msg = f"Error from OpenAI API: {response.text}"
                print(error_msg)
                
                # Only retry on 5xx errors (server issues) or 429 (rate limit)
                if response.status_code >= 500 or response.status_code == 429:
                    retry_count += 1
                    if retry_count <= max_retries:
                        retry_delay = 0.5 * (2 ** retry_count)  # Exponential backoff
                        print(f"Retrying in {retry_delay:.1f}s...")
                        time.sleep(retry_delay)
                        continue
                # Don't retry on other errors (4xx client errors)
                raise Exception(f"OpenAI API error: {response.status_code}")
            
            return response.content
        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count <= max_retries:
                retry_delay = 0.5 * (2 ** retry_count)  # Exponential backoff
                print(f"Request timed out after {current_timeout}s. Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
            else:
                raise TimeoutError(f"OpenAI API request timed out after {max_retries+1} attempts")
        except requests.exceptions.ConnectionError as e:
            retry_count += 1
            if retry_count <= max_retries:
                retry_delay = 0.5 * (2 ** retry_count)  # Exponential backoff
                print(f"Connection error: {e}. Retrying in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
            else:
                raise ConnectionError(f"Failed to connect to OpenAI API after {max_retries+1} attempts: {e}")

def play_audio_file(audio_path, blocking=True):
    """Play an audio file with optional blocking"""
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.set_volume(1.0)
        pygame.mixer.music.play()
        
        if blocking:
            # Use a safer wait method that can be interrupted gracefully
            start_time = time.time()
            timeout = 60  # Maximum wait time in seconds to prevent infinite loops
            
            try:
                # Wait for playback to finish with timeout and interruption handling
                while pygame.mixer.music.get_busy() and (time.time() - start_time < timeout):
                    # Shorter sleep reduces chance of KeyboardInterrupt issues
                    time.sleep(0.01)  
                    
                    # If we're waiting on playback and have a queue, signal preloading of next chunk
                    if len(_speech_queue) > 0:
                        _preload_event.set()
            except KeyboardInterrupt:
                # Gracefully handle interruption
                pygame.mixer.music.stop()
                print("\n⏹️ Playback interrupted")
                return False
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False
    
    return True


def preload_speech(text, voice="shimmer", speed=1.0):
    """Preload speech to prepare for seamless playback
    Returns the file path to the cached or generated audio file
    """
    # Check cache first
    cache_path = get_cache_path(text, voice)
    if os.path.exists(cache_path):
        return cache_path
    
    try:
        # Create a temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()
        
        # Generate speech with retry logic
        audio_content = generate_speech(text, voice, speed, max_timeout=20, max_retries=2)  
        
        # Save the audio to file
        with open(temp_path, "wb") as file:
            file.write(audio_content)
        
        # Move to cache location
        try:
            # Check if source and destination are on same drive
            if os.path.splitdrive(temp_path)[0] == os.path.splitdrive(cache_path)[0]:
                # Same drive, can use replace (move)
                os.replace(temp_path, cache_path)
            else:
                # Different drives, must copy then delete
                shutil.copy2(temp_path, cache_path)
                os.remove(temp_path)
            return cache_path
        except Exception as e:
            print(f"Warning: Could not cache audio: {e}")
            return temp_path  # Use the temp file instead
            
    except Exception as e:
        print(f"Error preloading speech: {e}")
        return None
        
def process_speech_queue():
    """Process and play items in the speech queue"""
    global _speech_queue
    
    while True:
        # Wait for preload event (signaled from play_audio_file or speak function)
        _preload_event.wait()
        _preload_event.clear()
        
        # Get next item from queue if available
        next_item = None
        with _queue_lock:
            if _speech_queue:
                next_item = _speech_queue.pop(0)
                
        # If no item, continue waiting
        if not next_item:
            continue
            
        text, voice, speed = next_item
        
        # Preload the speech file
        audio_path = preload_speech(text, voice, speed)
        
        if audio_path:
            # Wait until current playback finishes
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
                
            # Play the preloaded audio
            play_audio_file(audio_path, blocking=True)
            
            # Signal for next preload if queue not empty
            with _queue_lock:
                if _speech_queue:
                    _preload_event.set()

# Start the queue processor thread
_queue_processor = threading.Thread(target=process_speech_queue, daemon=True)
_queue_processor.start()

def cleanup_temp_file(file_path):
    """Clean up a temporary file"""
    if file_path and os.path.exists(file_path) and file_path.startswith(tempfile.gettempdir()):
        try:
            os.remove(file_path)
        except:
            pass

def split_into_sentences(text, max_chunk_size=100):
    """Split text into sentences for better TTS processing
    
    Args:
        text: Text to split into sentences
        max_chunk_size: Maximum size for any single chunk (reduced to improve API reliability)
    """
    # Basic sentence splitting on period, question mark, or exclamation mark
    # followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Filter out empty sentences and very short ones (likely not real sentences)
    sentences = [s for s in sentences if s and len(s) > 3]
    
    # If we have no sentences after filtering, return the original as one sentence
    if not sentences:
        sentences = [text]
    
    # Make sure the last sentence has proper punctuation
    if sentences and sentences[-1] and not sentences[-1][-1] in '.!?':
        sentences[-1] = sentences[-1] + '.'
    
    # Further chunk any sentences that are too long
    result = []
    for sentence in sentences:
        if len(sentence) <= max_chunk_size:
            # Sentence is already short enough
            result.append(sentence)
        else:
            # Split the long sentence by commas first
            comma_chunks = re.split(r'(?<=,)\s+', sentence)
            
            current_chunk = ""
            for chunk in comma_chunks:
                if len(current_chunk) + len(chunk) + 1 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += ", " + chunk
                    else:
                        current_chunk = chunk
                else:
                    # Add the current chunk to results if it's not empty
                    if current_chunk:
                        if not current_chunk[-1] in '.!?,;:':
                            current_chunk += '.'
                        result.append(current_chunk)
                    
                    # Start a new chunk - but if single chunk is too long, force split
                    if len(chunk) > max_chunk_size:
                        # Chunk is still too long, split by words
                        words = chunk.split()
                        word_chunk = ""
                        for word in words:
                            if len(word_chunk) + len(word) + 1 <= max_chunk_size:
                                if word_chunk:
                                    word_chunk += " " + word
                                else:
                                    word_chunk = word
                            else:
                                if word_chunk and not word_chunk[-1] in '.!?,;:':
                                    word_chunk += '.'
                                result.append(word_chunk)
                                word_chunk = word
                                
                        # Add the last word chunk
                        if word_chunk:
                            if not word_chunk[-1] in '.!?,;:':
                                word_chunk += '.'
                            result.append(word_chunk)
                    else:
                        current_chunk = chunk
            
            # Add the last chunk
            if current_chunk:
                if not current_chunk[-1] in '.!?,;:':
                    current_chunk += '.'
                result.append(current_chunk)
    
    return result

def speak(text, voice=None, speed=1.0, use_cache=True, blocking=True, timeout=None, chunk_long_text=True):
    """Generate speech using OpenAI's API and play it
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (default: uses VOICE_CONFIG["default"])
        speed: Speech speed multiplier (default: 1.0)
        use_cache: Whether to cache speech files (default: True)
        blocking: Whether to block until playback completes (default: True)
        timeout: Timeout in seconds for API call (default: uses VOICE_CONFIG["timeout"])
        chunk_long_text: Whether to split long text into sentences (default: True)
    """
    global _is_generating, _speech_queue, VOICE_CONFIG
    
    # Use config values if not explicitly specified
    voice = voice or VOICE_CONFIG["default"]
    timeout = timeout or VOICE_CONFIG["timeout"]
    max_chunk_size = VOICE_CONFIG["chunk_size"]
    
    # Return early if text is empty
    if not text or len(text.strip()) == 0:
        return
    
    # Clean up text
    text = text.strip()
    
    # Initialize API if needed
    if not initialize_api():
        print("Failed to initialize OpenAI API. Unable to generate speech.")
        return
    
    # 👇 Disable chunking entirely – send the full text in one request
    if False:  # Disabled chunking
        print(f"Using full text without chunking ({len(text)} characters)...")
        
        # Generate speech once
        audio_path = preload_speech(text, voice, speed)
        if audio_path:
            play_audio_file(audio_path, blocking=blocking)
            return
    elif chunk_long_text and len(text) > max_chunk_size:  # Use configured chunk size but never reached
        sentences = split_into_sentences(text)
        
        # If we have multiple sentences, process them individually
        if len(sentences) > 1:
            print(f"Splitting text into {len(sentences)} chunks for buffered playback")
            
            # Handle sentences with async buffering
            first_sentence = sentences[0]
            remaining_sentences = sentences[1:]
            
            # Play first sentence immediately (blocking) to start audio quickly
            speak(first_sentence, voice, speed, use_cache, 
                  blocking=True,  # First chunk is always blocking
                  timeout=timeout, chunk_long_text=False)
                
            # Queue remaining sentences for async processing
            with _queue_lock:
                # Clear existing queue if any
                _speech_queue.clear()
                
                # Add all sentences to the queue
                for sentence in remaining_sentences:
                    _speech_queue.append((sentence, voice, speed))
                
                # Signal to start processing the queue
                if _speech_queue:
                    _preload_event.set()
            
            # If blocking mode requested, wait for all chunks to finish
            if blocking:
                # Use a safer blocking wait with timeout to prevent deadlocks
                wait_start = time.time()
                max_wait_time = 120  # Maximum wait of 2 minutes as safety
                
                try:
                    while time.time() - wait_start < max_wait_time:
                        with _queue_lock:
                            queue_empty = not _speech_queue
                        
                        music_done = not pygame.mixer.music.get_busy()
                        
                        # Exit conditions
                        if queue_empty and music_done:
                            break
                            
                        # Brief sleep to prevent CPU spinning
                        time.sleep(0.05)
                except KeyboardInterrupt:
                    # Gracefully handle interruption
                    with _queue_lock:
                        _speech_queue.clear()
                    print("\n⚠️ Speech playback interrupted.")
                    return
            
            # All sentences processed or queued
            return
    
    # Single chunk processing (short text or a chunk of longer text)
    cache_path = get_cache_path(text, voice) if use_cache else None
    
    # For cached audio, play directly
    if use_cache and os.path.exists(cache_path):
        # Just play the cached audio
        if blocking:
            play_audio_file(cache_path, blocking=True)
        else:
            _thread_pool.submit(play_audio_file, cache_path, True)
        return
    
    # For non-cached audio that needs generation
    try:
        # Show generating message
        _is_generating = True
        # Suppressed console message
        # print(f"Generating speech with {voice} voice...")
        
        # Create temp file path
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()
        
        # Generate speech with improved retry logic
        try:
            # Use our improved generate_speech with retry
            audio_content = generate_speech(text, voice, speed, 
                                           max_timeout=timeout, 
                                           max_retries=2)
            
            # Save the audio to file
            with open(temp_path, "wb") as file:
                file.write(audio_content)
            
            # Cache the audio if enabled
            audio_path = temp_path
            if use_cache:
                try:
                    # Check if source and destination are on same drive
                    if os.path.splitdrive(temp_path)[0] == os.path.splitdrive(cache_path)[0]:
                        # Same drive, can use replace (move)
                        os.replace(temp_path, cache_path)
                    else:
                        # Different drives, must copy then delete
                        shutil.copy2(temp_path, cache_path)
                        os.remove(temp_path)
                    audio_path = cache_path
                    # Cache file was successfully generated
                    # print("Audio cached successfully")
                except Exception as e:
                    print(f"Warning: Could not cache audio: {e}")
                    # Keep using the temp path
                    
        except Exception as e:
            print(f"Error generating speech: {e}")
            cleanup_temp_file(temp_path)
            _is_generating = False
            return
            
        finally:
            _is_generating = False
        
        # Play the audio
        if blocking:
            play_audio_file(audio_path, blocking=True)
            # Cleanup temp file if needed
            if audio_path == temp_path:
                cleanup_temp_file(temp_path)
        else:
            # Use a thread for non-blocking playback with cleanup
            def play_and_cleanup():
                play_audio_file(audio_path, blocking=True)
                if audio_path == temp_path:
                    cleanup_temp_file(temp_path)
            
            _thread_pool.submit(play_and_cleanup)
                    
    except Exception as e:
        print(f"Error in speech processing: {e}")
        _is_generating = False

# Configuration options
VOICE_CONFIG = {
    "default": "shimmer",  # Default voice to use
    "available_voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "chunk_size": 4096,     # Maximum characters per chunk (increased to prevent fallback chunking)
    "timeout": 25,         # Default timeout in seconds
    "max_retries": 3,      # Maximum retries on failure
}

def configure_voice(voice=None, chunk_size=None, timeout=None, max_retries=None):
    """Configure the voice system settings
    
    Args:
        voice: Voice to use (one of available_voices)
        chunk_size: Maximum characters per chunk
        timeout: Timeout in seconds for API calls
        max_retries: Maximum retries on failure
    
    Returns:
        Dict with current configuration
    """
    global VOICE_CONFIG
    
    if voice and voice in VOICE_CONFIG["available_voices"]:
        VOICE_CONFIG["default"] = voice
        print(f"Voice set to: {voice}")
    
    if chunk_size and isinstance(chunk_size, int) and 50 <= chunk_size <= 300:
        VOICE_CONFIG["chunk_size"] = chunk_size
        print(f"Chunk size set to: {chunk_size} characters")
    
    if timeout and isinstance(timeout, int) and 5 <= timeout <= 60:
        VOICE_CONFIG["timeout"] = timeout
        print(f"Timeout set to: {timeout} seconds")
        
    if max_retries and isinstance(max_retries, int) and 0 <= max_retries <= 5:
        VOICE_CONFIG["max_retries"] = max_retries
        print(f"Max retries set to: {max_retries}")
    
    return VOICE_CONFIG

# Simple test function
if __name__ == "__main__":
    test_text = "Hello! I'm speaking with the Shimmer voice from OpenAI. How do I sound?"
    speak(test_text)
