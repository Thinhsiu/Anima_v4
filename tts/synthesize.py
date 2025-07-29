import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import os
import hashlib
import pickle

# Trust all necessary config and model definitions
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

from TTS.api import TTS
import pygame
import time

# Check if CUDA is available for GPU acceleration
use_gpu = torch.cuda.is_available()

# Cache directory for storing generated speech
CACHE_DIR = "tts/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Path definitions
CUSTOM_VOICE_PATH = "tts/voices/anima_voice.wav"
SPEAKER_LANG = "en"

# Initialize XTTS with optimized settings
# Load the model only once during import, not every time speak() is called
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=use_gpu)

# Set global performance settings
if use_gpu:
    # Optimize for GPU inference
    torch.backends.cudnn.benchmark = True
    # Use lower precision for faster computation
    torch.set_float32_matmul_precision('medium')

# Pre-warm the model
tts.tts("Warming up the model", speaker_wav=CUSTOM_VOICE_PATH, language=SPEAKER_LANG)

def get_cache_path(text):
    """Generate a cache file path based on the text content"""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{text_hash}.wav")

def speak(text, speed=1.0, use_cache=True, fast_mode=False):
    """Speak the provided text with optional speed control and caching
    
    Args:
        text (str): The text to speak
        speed (float): Speed factor (0.5=slower, 1.0=normal, 1.5=faster)
        use_cache (bool): Whether to use cached audio files
        fast_mode (bool): If True, uses optimization techniques for faster generation
    """
    output_path = "tts_output.wav"
    
    # Remove any debug string representation artifacts
    # Check if the text contains Python list representation artifacts
    if "', '" in text or "']" in text or "['" in text:
        # This might be a string representation of a list - clean it up
        text = text.replace("', '", ". ").replace("[']", "").replace("[']", "")
        text = text.replace("['\n", "").replace("']", "").replace("['\n ", "") 
        text = text.replace("['\n", "").replace("['\n", "")
        text = text.replace("['\n", "").replace("['\n", "")
    
    # Split long text into sentences for better caching and faster processing
    # Use a regex pattern for better sentence splitting
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentence_audio_paths = []
    
    # Debug info suppressed for production
    # print(f"Processing {len(sentences)} sentences")
    
    for sentence in sentences:
        if not sentence:
            continue
            
        # Clean any remaining artifacts that might cause issues
        sentence = sentence.strip().replace("'", "").replace('"', "")
        
        # Add period back if it was removed during splitting and doesn't end with punctuation
        if not any(sentence.endswith(p) for p in ['.', '!', '?']):
            sentence += '.'
            
        # Check if this sentence is cached
        cache_path = get_cache_path(sentence)
        if use_cache and os.path.exists(cache_path):
            sentence_audio_paths.append(cache_path)
        else:
            # Use a temporary file for this sentence
            temp_path = f"temp_{len(sentence_audio_paths)}.wav"
            
            # Generate speech with optimized settings
            generation_settings = {}
            if fast_mode:
                # Use settings that trade some quality for speed
                generation_settings = {
                    "enable_text_splitting": True
                }
            
            # Generate the audio
            tts.tts_to_file(
                text=sentence,
                speaker_wav=CUSTOM_VOICE_PATH,
                language=SPEAKER_LANG,
                file_path=temp_path,
                **generation_settings
            )
            
            # Save to cache if caching is enabled
            if use_cache:
                try:
                    os.replace(temp_path, cache_path)
                    sentence_audio_paths.append(cache_path)
                except Exception as e:
                    # Debug info suppressed
                    pass  # Silently continue on cache failures
                    sentence_audio_paths.append(temp_path)
            else:
                sentence_audio_paths.append(temp_path)
    
    # Play the audio files sequentially
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Print timing information 
        start_time = time.time()
        
        # Play each sentence audio file sequentially
        for idx, audio_path in enumerate(sentence_audio_paths):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.set_volume(1.0)  # Set to maximum volume
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
                
            # Delete temporary files after playing
            if audio_path.startswith('temp_') and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if len(sentence_audio_paths) > 0:
            # Debug info suppressed for production
            # print(f" > Processing time: {processing_time}")
            # Calculate real-time factor (time to generate / time to play)
            # if processing_time > 0:
            #     rtf = len(text) / (100 * processing_time)  # Approximate chars per second
            #     print(f" > Real-time factor: {rtf}")
            pass
            
    except Exception as e:
        # Debug info suppressed
        # print(f"Error playing audio: {e}")
        
        # Clean up any temporary files
        for path in sentence_audio_paths:
            if path.startswith('temp_') and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        # Force clean up in case of error
        try:
            pygame.mixer.music.stop()
        except:
            pass
