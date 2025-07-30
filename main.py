import os
import sys
import time
import datetime
import re
import warnings
import random

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Import silent startup module to suppress all output during initialization
try:
    from silent_startup import enable_silent_mode, disable_silent_mode
    # Enable silent mode immediately
    silent_filter = enable_silent_mode()
    SILENT_MODE_ENABLED = True
except ImportError:
    SILENT_MODE_ENABLED = False
    print("Silent startup module not found, continuing with normal output")

# Suppress other verbose logs
import logging
logging.basicConfig(level=logging.ERROR)  # Only show errors by default
from difflib import get_close_matches
import base64
import subprocess
import threading
import pyaudio
import json
from pathlib import Path
import importlib

warnings.filterwarnings('ignore')

# Set up logger
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

openai_api_key = None
for key in ["OPENAI_API_KEY", "OPENAI_35_API_KEY", "OPENAI_4_API_KEY"]:
    if key in os.environ:
        openai_api_key = os.environ[key]
        os.environ["OPENAI_API_KEY"] = openai_api_key
        break

# Try to import smart memory manager
try:
    from vision.smart_memory_manager import smart_memory_manager
    SMART_MEMORY_ENABLED = True
except ImportError:
    print("Smart memory management not available")
    SMART_MEMORY_ENABLED = False

# Import NLP system
try:
    from nlp.integration import initialize as initialize_nlp, get_instance as get_nlp_instance
    NLP_SYSTEM_ENABLED = True
    logger.debug("NLP system loaded successfully")
    # Initialize the NLP system right away
    initialize_nlp()
except ImportError as e:
    print(f"NLP system not available: {e}")
    NLP_SYSTEM_ENABLED = False
    # Create dummy functions for graceful degradation
    def get_nlp_instance(): return None
    
# Import Intelligence Integration and Custom Entity Training
try:
    from nlp.intelligence_integration import initialize as initialize_intelligence
    from nlp.intelligence_integration import get_instance as get_intelligence_instance
    from nlp.intelligence_integration import get_intelligence_help, enhance_prompt_with_intelligence
    INTELLIGENCE_ENABLED = True
    logger.debug("Intelligence integration loaded successfully")
    # Initialize the intelligence system
    initialize_intelligence()
except ImportError as e:
    print(f"Intelligence integration not available: {e}")
    INTELLIGENCE_ENABLED = False
    # Create dummy functions for graceful degradation
    def get_intelligence_instance(): return None
    def get_intelligence_help(): return "Intelligence features not available."
    def enhance_prompt_with_intelligence(prompt, *args, **kwargs): return prompt

# Import Custom Entity Training (don't initialize by default as it's resource-intensive)
try:
    from nlp.custom_entities import get_instance as get_entity_trainer
    from nlp.annotation_tool import AnnotationTool
    CUSTOM_ENTITIES_ENABLED = True
    logger.debug("Custom entity training available")
except ImportError as e:
    print(f"Custom entity training not available: {e}")
    CUSTOM_ENTITIES_ENABLED = False

# Import optional emotion awareness system
try:
    from emotion.integration import initialize as initialize_emotion_system
    from emotion.integration import get_instance as get_emotion_system
    from emotion.integration import analyze_user_input, enhance_response as enhance_response_with_emotion
    from emotion.integration import enhance_prompt as enhance_prompt_with_emotion
    from emotion.integration import get_emotion_help
    # Try to import the voice-emotion bridge connector
    try:
        from emotion.voice_bridge import connect_to_voice_system as connect_voice_emotion
        HAS_VOICE_EMOTION = True
    except ImportError:
        # Create dummy function if not available
        def connect_voice_emotion():
            return False
        HAS_VOICE_EMOTION = False
    EMOTION_SYSTEM_ENABLED = True
except ImportError:
    logger.warning("Emotion awareness system not available")
    EMOTION_SYSTEM_ENABLED = False
    # Provide dummy functions
    def initialize_emotion_system(*args, **kwargs): return None
    def get_emotion_system(): return None
    def analyze_user_input(text): return {}
    def enhance_response_with_emotion(response, user_input): return response
    def enhance_prompt_with_emotion(prompt, user_input): return prompt
    def get_emotion_help(): return "Emotion awareness features are not available."

# Try to import core awareness and memory integration if available
AWARENESS_AVAILABLE = False
# First try to import from core.__init__ which has proper aliasing
try:
    from core import awareness, add_conversation, enhance_prompt_with_awareness
    from core import memory_integration, enhance_prompt_with_memory
    # Try to import memory bridge for emotion-memory connection
    try:
        from core.memory_bridge import get_memory_bridge, MEMORY_BRIDGE_AVAILABLE
        HAS_MEMORY_BRIDGE = True
    except ImportError:
        HAS_MEMORY_BRIDGE = False
    AWARENESS_AVAILABLE = True
    logger.debug("Awareness system loaded successfully")
except (ImportError, AttributeError):
    # If that fails, try direct imports
    try:
        from core.awareness import awareness, add_conversation
        # Check if enhance_prompt exists and alias it if needed
        if hasattr(awareness, 'enhance_prompt'):
            from core.awareness import enhance_prompt as enhance_prompt_with_awareness
        else:
            def enhance_prompt_with_awareness(prompt): return prompt
            
        from core.memory_integration import memory_integration
        # Check if enhance_prompt_with_memories exists and alias it if needed
        if hasattr(memory_integration, 'enhance_prompt_with_memories'):
            from core.memory_integration import enhance_prompt_with_memories as enhance_prompt_with_memory
        else:
            def enhance_prompt_with_memory(prompt, user_input): return prompt
            
        AWARENESS_AVAILABLE = True
        logger.debug("Awareness system loaded successfully")
    except ImportError:
        logger.warning("Core awareness or memory integration modules not available. Some features will be limited.")
        AWARENESS_AVAILABLE = False
        
        # Define dummy functions for graceful degradation
        def add_conversation(*args, **kwargs): pass
        def enhance_prompt_with_awareness(*args, **kwargs): return args[0] if args else ""
        def enhance_prompt_with_memory(*args, **kwargs): return args[0] if args else ""

# Import enhanced awareness integration if available
try:
    # Try importing with debugging
    from core.enhanced_awareness_integration import enhance_prompt_with_all, process_exchange, get_system_status, check_modules
    ENHANCED_AWARENESS_AVAILABLE = True
    logger.debug("Enhanced awareness integration loaded successfully")
    # Print module status for debugging
    check_modules()
except ImportError as e:
    print(f"Warning: Enhanced awareness integration not available: {e}")
    print("Some advanced features will be limited.")
    ENHANCED_AWARENESS_AVAILABLE = False
    
    # Define dummy functions for graceful degradation
    def enhance_prompt_with_all(*args, **kwargs): return args[0] if args else ""
    def process_exchange(*args, **kwargs): return args[1] if len(args) > 1 else ""
    def get_system_status(): return {"available_modules": {}}
    def check_modules(): pass

# Import vision integration module
try:
    from vision_integration import integrate_vision_with_anima, get_help as get_vision_help
    VISION_ENABLED = True
except ImportError:
    logger.warning("Vision integration not available")
    VISION_ENABLED = False

# Import file sharing UI integration if available
FILE_SHARING_ENABLED = False
try:
    from ui.file_sharing_integration import handle_file_command, launch_file_ui, recall_files
    FILE_SHARING_ENABLED = True
    logger.info("File sharing UI integration loaded successfully")
except ImportError:
    VISION_ENABLED = False
    print("Vision capabilities not available. Install required packages with: pip install openai pillow")

from llm.openai_llm import query_openai
from llm.local_llm import query_local_llm
from utils.persona_loader import generate_complete_system_prompt, load_complete_persona
from utils.knowledge_manager import download_deep_knowledge, get_knowledge, get_knowledge_stats
from stt.vad import start_smart_listening, stop_smart_listening, get_speech_monitor

def load_thoth():
    with open("thoth.txt", "r", encoding="utf-8") as f:
        return f.read()

def internet_available():
    try:
        import urllib.request
        urllib.request.urlopen('http://google.com', timeout=2)
        return True
    except:
        return False

# Voice mode states
voice_mode = False
smart_duplex = False  # Smart duplex mode with VAD
is_playing_response = False  # Flag to track when system is speaking

# Current audio recording data
current_recording = None
speech_in_progress = False

def start_voice_recognition():
    """Start traditional voice recognition"""
    try:
        from stt.whisper_stt import initialize_whisper_stt
        initialize_whisper_stt()
        return True
    except Exception as e:
        print(f"Failed to initialize voice recognition: {e}")
        return False

def handle_speech_start():
    """Called when VAD detects speech start"""
    global speech_in_progress, is_playing_response
    if is_playing_response:
        return  # Ignore any VAD triggers while we're speaking
    speech_in_progress = True
    print("🎤 Speech detected...")

def handle_speech_end(audio_data):
    """Called when VAD detects speech end with recorded audio"""
    global current_recording, speech_in_progress, is_playing_response
    if is_playing_response:
        return  # Block processing if it was self-triggered
    speech_in_progress = False
    current_recording = audio_data
    print("📝 Processing speech...")
        
def stop_voice_recognition():
    """Stop traditional voice recognition"""
    # Placeholder for cleanup code
    pass
    
def start_smart_duplex():
    """Start smart duplex mode with VAD"""
    global smart_duplex
    try:
        # Start VAD monitor with our callbacks
        start_smart_listening(
            on_speech_start=handle_speech_start,
            on_speech_end=handle_speech_end
        )
        smart_duplex = True
        return True
    except Exception as e:
        print(f"Failed to start smart duplex: {e}")
        smart_duplex = False
        return False
        
def stop_smart_duplex():
    """Stop smart duplex mode"""
    global smart_duplex
    try:
        stop_smart_listening()
        smart_duplex = False
    except Exception as e:
        print(f"Error stopping smart duplex: {e}")
    return True

def toggle_voice_mode(use_smart_duplex=True):
    """Toggle between voice mode, smart duplex, and text mode"""
    global voice_mode, smart_duplex
    
    # If already in voice mode, disable it
    if voice_mode:
        # Try to speak the mode change
        try:
            from tts.openai_voice import speak
            speak("Switching to text mode now. Voice recognition deactivated.", blocking=True)
        except Exception as e:
            print(f"TTS error: {e}")
            
        # Disable voice mode
        voice_mode = False
        print("\nVoice mode disabled.")
        
        # Clean up any active voice resources
        if smart_duplex:
            stop_smart_duplex()
            smart_duplex = False
            print("Smart duplex disabled.")
        
        return False
    else:  # Turn on voice mode
        if use_smart_duplex:
            # Try smart duplex first - better experience
            if start_smart_duplex():
                voice_mode = True
                msg = "🎤 Smart voice mode activated! I'm listening now. Just speak naturally when ready..."
                print("Anima: " + msg)
                
                # Speak the confirmation to make it clear voice mode is active
                try:
                    from tts.openai_voice import speak
                    speak("Voice mode is now active. I'm listening for your commands.")
                except Exception as e:
                    print(f"TTS error in voice mode activation: {e}")
                    
                return msg
            else:
                # Fall back to regular voice mode
                print("Smart duplex failed, falling back to regular voice mode")
                if start_voice_recognition():
                    voice_mode = True
                    msg = "🎤 Voice mode activated! Speak to me..."
                    print("Anima: " + msg)
                    
                    # Speak the confirmation
                    try:
                        from tts.openai_voice import speak
                        speak("Voice mode is now active. I'm listening for your commands.")
                    except Exception as e:
                        print(f"TTS error in voice mode activation: {e}")
                        
                    return msg
                else:
                    return "Failed to activate voice mode due to initialization error."
        else:
            # Traditional voice mode
            if start_voice_recognition():
                voice_mode = True
                msg = "🎤 Voice mode activated! Speak to me..."
                print("Anima: " + msg)
                
                # Speak the confirmation
                try:
                    from tts.openai_voice import speak
                    speak("Voice mode is now active. I'm listening for your commands.")
                except Exception as e:
                    print(f"TTS error in voice mode activation: {e}")
                    
                return msg
            else:
                return "Failed to activate voice mode due to initialization error."

def get_voice_input():
    """Get voice input from user with enhanced real-time voice emotion detection"""
    global current_recording, smart_duplex
    
    try:
        # Import UI components for emotion display
        try:
            from ui.emotion_display import display_voice_emotion_indicator, display_emotion_panel
            HAS_EMOTION_UI = True
        except ImportError:
            HAS_EMOTION_UI = False
            print("Note: Emotion UI components not available")
        
        # Import the voice-emotion bridge if available
        voice_emotion_bridge = None
        if HAS_VOICE_EMOTION and EMOTION_SYSTEM_ENABLED:
            try:
                from emotion.voice_bridge import voice_bridge
                voice_emotion_bridge = voice_bridge
            except ImportError:
                pass
        
        # Smart duplex mode uses VAD to detect when speech is complete
        if smart_duplex:
            # Wait for speech to be detected and processed
            print("🎤 Waiting for speech (speak naturally)...")
            print("    Voice detection active with dynamic energy thresholds")
            
            # Wait for VAD to detect and record speech
            timeout = 60  # Maximum wait time in seconds
            start_time = time.time()
            status_update_time = start_time
            status_chars = [".  ", ".. ", "...", " ..", "  .", "   "]
            status_idx = 0
            
            # Reset any previous recording
            current_recording = None
            
            # Wait for speech to be detected, processed and stored in current_recording
            while current_recording is None:
                # Check for timeout
                current_time = time.time()
                if current_time - start_time > timeout:
                    print("\nTimeout waiting for speech")
                    return input("Please type your message instead: ")
                
                # Periodically show a status animation
                if current_time - status_update_time > 0.5:  # Update every half second
                    status_char = status_chars[status_idx % len(status_chars)]
                    status_idx += 1
                    print(f"\rListening{status_char}", end="")
                    status_update_time = current_time
                    
                # Short sleep to prevent CPU spinning
                time.sleep(0.1)
            
            print("\r\033[KSpeech detected! Processing...", end="")
            
            # We have speech, process it with Whisper
            from stt.whisper_stt import transcribe_from_buffer
            monitor = get_speech_monitor()
            
            # Convert to WAV format for Whisper
            wav_data = monitor.get_wav_data(current_recording)
            
            # Process audio for emotion detection if available
            emotion_features = None
            if HAS_VOICE_EMOTION and EMOTION_SYSTEM_ENABLED and voice_emotion_bridge:
                try:
                    # Extract emotion features from the audio
                    emotion_features = voice_emotion_bridge.extract_emotion_features(wav_data)
                    
                    # Store in memory if bridge is available
                    voice_emotion_bridge.process_audio(wav_data)
                    
                except Exception as e:
                    print(f"\rVoice emotion detection error: {e}")
            
            # Transcribe using Whisper
            print("\r\033[KTranscribing speech...", end="")
            transcript = transcribe_from_buffer(wav_data)
            print(f"\r\033[K", end="")
            
            # Display emotion alongside transcript if available
            if emotion_features and HAS_EMOTION_UI:
                dominant_emotion = emotion_features.get('dominant_emotion', 'neutral')
                confidence = emotion_features.get('confidence', 0.0)
                
                # Only show emotion if confidence is high enough
                if confidence >= 0.4:
                    # Create a badge for the emotion
                    badge = display_voice_emotion_indicator(emotion_features)
                    print(f"\nYou: {transcript} {badge}")
                else:
                    print(f"\nYou: {transcript}")
            else:
                print(f"\nYou: {transcript}")
            
            # Reset for next recording
            current_recording = None
            return transcript
            
        else:
            # Traditional mode - record for fixed duration with dynamic adjustment
            from stt.whisper_stt import record_audio, transcribe_audio
            
            # Get configuration from STT module if available
            try:
                from stt.whisper_stt import STT_CONFIG
                recording_seconds = STT_CONFIG.get('recording_seconds', 7)
            except (ImportError, AttributeError):
                recording_seconds = 7
                
            print(f"🎤 Listening for {recording_seconds} seconds...")
            
            # Show progress bar during recording
            start_time = time.time()
            progress_thread = threading.Thread(
                target=_show_recording_progress, 
                args=(recording_seconds, start_time),
                daemon=True
            )
            progress_thread.start()
            
            # Record audio
            audio = record_audio(recording_seconds)
            
            # Clear progress display
            print("\r" + " " * 50 + "\r", end="")
            
            if audio:
                # Process audio for emotion detection if available
                emotion_features = None
                if HAS_VOICE_EMOTION and EMOTION_SYSTEM_ENABLED and voice_emotion_bridge:
                    try:
                        # Extract emotion features from the audio
                        print("Processing voice emotions...", end="\r")
                        emotion_features = voice_emotion_bridge.extract_emotion_features(audio)
                        
                        # Store in memory
                        voice_emotion_bridge.process_audio(audio)
                        
                    except Exception as e:
                        print(f"\rVoice emotion detection error: {e}")
                
                # Transcribe the audio
                print("Transcribing speech...    ", end="\r")
                transcript = transcribe_audio(audio)
                print(" " * 30, end="\r")
                
                # Display emotion alongside transcript if available
                if emotion_features and HAS_EMOTION_UI:
                    dominant_emotion = emotion_features.get('dominant_emotion', 'neutral')
                    confidence = emotion_features.get('confidence', 0.0)
                    
                    # Only show emotion if confidence is high enough
                    if confidence >= 0.4:
                        # Create a badge for the emotion
                        badge = display_voice_emotion_indicator(emotion_features)
                        print(f"\nYou: {transcript} {badge}")
                    else:
                        print(f"\nYou: {transcript}")
                else:
                    print(f"\nYou: {transcript}")
                    
                return transcript
            else:
                print("No speech detected.")
                return input("Please type your message instead: ")
                
    except Exception as e:
        print(f"Voice input error: {e}")
        import traceback
        print(traceback.format_exc())
        return input("Please type your message instead: ")

def _show_recording_progress(duration, start_time):
    """Show a progress bar during voice recording"""
    try:
        bar_width = 30
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                # Complete the bar and exit
                filled = bar_width
                percent = 100
                print(f"\rRecording: [{'█' * filled}{' ' * (bar_width-filled)}] {percent}%", end="")
                break
                
            # Calculate progress
            percent = int((elapsed / duration) * 100)
            filled = int((elapsed / duration) * bar_width)
            
            # Show progress bar with animation
            print(f"\rRecording: [{'█' * filled}{' ' * (bar_width-filled)}] {percent}%", end="")
            time.sleep(0.1)
    except Exception as e:
        # Silently handle errors in the progress display
        pass
        return input("Fallback (type your message): ")

def speak_with_half_duplex(text):
    """Handle text-to-speech with proper voice/listening mode management"""
    global voice_mode, is_playing_response
    was_voice_mode = voice_mode
    
    # Disable voice recognition before speaking
    if voice_mode:
        voice_mode = False
        stop_voice_recognition()
        print("🔇 Pausing listening while speaking...")
    
    try:
        # Filter out code blocks and URLs which TTS can't handle well
        speak_text = re.sub(r'```[\s\S]*?```', '...code omitted...', text)
        speak_text = re.sub(r'https?://\S+', 'link', speak_text)
        
        # Mark as playing *right before* audio playback starts
        is_playing_response = True
        print("🔊 Speaking response...")
        
        # Use TTS to speak the text (blocking until complete)
        from tts.openai_voice import speak
        speak(speak_text, voice="shimmer", use_cache=True, blocking=True, timeout=25)
        
    except Exception as e:
        print(f"TTS error: {e}")
    finally:
        # Small safety delay for any audio to decay
        time.sleep(0.5)  
        
        # Mark as not playing immediately after speech finishes
        is_playing_response = False
        
        # Restore voice mode if it was active
        if was_voice_mode:
            voice_mode = True
            print("🎤 Resuming listening...")
            start_voice_recognition()
            
            # Short cooldown to prevent immediate false activations
            print("🕖 Brief voice cooldown...")
            time.sleep(1.0)  
            print("🎤 Waiting for speech (speak naturally)...")

def load_thoth():
    """Load the base Thoth prompt"""
    thoth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "thoth.txt")
    if os.path.exists(thoth_path):
        with open(thoth_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

def generate_complete_system_prompt():
    """Generate the complete system prompt from persona files"""
    persona_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona")
    identity_path = os.path.join(persona_dir, "identity.json")
    
    if os.path.exists(identity_path):
        with open(identity_path, 'r', encoding='utf-8') as f:
            identity = json.load(f)
        
        # Format the persona into a prompt
        prompt_parts = []
        
        if "persona_text" in identity:
            prompt_parts.append(identity["persona_text"])
            
        if "tone" in identity:
            prompt_parts.append(f"Tone: {identity['tone']}")
            
        if "personality" in identity and isinstance(identity["personality"], list):
            prompt_parts.append("Personality traits: " + ", ".join(identity["personality"]))
            
        if "forbidden" in identity and isinstance(identity["forbidden"], list):
            prompt_parts.append("Never: " + "; ".join(identity["forbidden"]))
            
        if "core_values" in identity and isinstance(identity["core_values"], list):
            prompt_parts.append("Core values: " + ", ".join(identity["core_values"]))
            
        if "system_prompt" in identity:
            prompt_parts.append(identity["system_prompt"])
            
        return "\n\n".join(prompt_parts)
    
    return ""

def create_prompt(persona, user_input, convo_history, conversation_id=None):
    """Create the full prompt for the LLM"""
    base_prompt = load_thoth()
    persona_prompt = generate_complete_system_prompt()
    system_prompt = f"{base_prompt}\n\n{persona_prompt}"
    
    context = ""
    if "short_term_memory" in persona and persona["short_term_memory"]:
        context = "Recent context: " + str(persona["short_term_memory"])
    
    convo_history_str = "\n".join([f"User: {item['user']}\nAssistant: {item['assistant']}" for item in convo_history])
    full_prompt = f"{system_prompt}\n\n{context}\n\nConversation History:\n{convo_history_str}\n\nUser: {user_input}"
    
    # Apply awareness enhancements
    if AWARENESS_AVAILABLE:
        try:
            full_prompt = enhance_prompt_with_awareness(full_prompt)
            full_prompt = enhance_prompt_with_memory(full_prompt, user_input)
        except Exception as e:
            print(f"Warning: Error enhancing prompt with awareness: {e}")
            
    # Apply enhanced awareness features
    if ENHANCED_AWARENESS_AVAILABLE:
        try:
            full_prompt = enhance_prompt_with_all(full_prompt, user_input, conversation_id)
        except Exception as e:
            print(f"Warning: Error enhancing prompt with enhanced awareness: {e}")
    
    return full_prompt

def load_complete_persona():
    """Load the complete persona information"""
    persona = {}
    persona_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona")
    
    # Load identity
    identity_path = os.path.join(persona_dir, "identity.json")
    if os.path.exists(identity_path):
        with open(identity_path, 'r', encoding='utf-8') as f:
            persona["identity"] = json.load(f)
    
    # Load knowledge vault
    knowledge_path = os.path.join(persona_dir, "knowledge_vault.json")
    if os.path.exists(knowledge_path):
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            persona["knowledge"] = json.load(f)
            
    # Ensure user information exists with correct name
    if "user" not in persona:
        persona["user"] = {}
    persona["user"]["full_name"] = "Thinh"
    persona["user"]["preferred_name"] = "Thinh"
    
    return persona

def main():

    # Silence non-essential output
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    """Main function to run Anima"""
    # Disable silent mode before displaying the UI
    if SILENT_MODE_ENABLED:
        try:
            # Disable silent mode and restore normal output
            disable_silent_mode(silent_filter, restore_stdout=True)
            logger.info("Silent mode disabled, resuming normal output")
        except Exception as e:
            logger.warning(f"Error disabling silent mode: {e}")
    
    # Set logging levels for core modules
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Load persona and initialize variables
    persona = load_complete_persona()
    convo_history = []
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good morning"
        time_context = "early day"
    elif 12 <= current_hour < 17:
        greeting = "Good afternoon"
        time_context = "day"
    else:
        greeting = "Good evening"
        time_context = "evening"

    user_name = persona.get("user", {}).get("full_name", "").split()[0] if "full_name" in persona.get("user", {}) else ""
    
    # More personalized, varied greeting templates
    greet_templates = [
        f"{greeting}, {user_name}! How can I help you this {time_context}?",
        f"Hey {user_name}! It's great to see you this {time_context}.",
        f"Hi there, {user_name}! I was just thinking about you. How's your {time_context} going?",
        f"Welcome back, {user_name}! I'm happy to assist you today.",
        f"{greeting}, {user_name}! I'm ready whenever you are.",
        f"Hi {user_name}! It's a pleasure to connect with you again.",
        f"Hello {user_name}! I've been looking forward to our conversation."
    ]
    greeting_msg = random.choice(greet_templates).replace("  ", " ").strip()

    print(f"Anima: {greeting_msg}")
    print("(Type 'exit' to quit or 'voice mode' to enable speech.)")

    # TTS handling with error recovery
    tts_error = None
    try:
        from tts.openai_voice import speak
        speak(greeting_msg)
    except Exception as e:
        tts_error = str(e)
        print(f"TTS error: {e}")
        
        # Try to install pygame if that's the issue
        if "No module named 'pygame'" in tts_error:
            try:
                print("Installing pygame for voice synthesis...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
                print("Pygame installed successfully! Please restart Anima.")
            except Exception as pygame_install_error:
                print(f"Could not install pygame: {pygame_install_error}")

    # Initialize NLP, Intelligence, Emotion and Awareness systems
    if NLP_SYSTEM_ENABLED:
        nlp = get_nlp_instance()
        if nlp:
            print("NLP system initialized")

    if INTELLIGENCE_ENABLED:
        initialize_intelligence()
        print("Intelligence system initialized")

    if EMOTION_SYSTEM_ENABLED:
        # Initialize emotion system
        emotion_system = initialize_emotion_system()
        if emotion_system:
            print("Emotion system initialized")
            
            # Connect voice-emotion bridge if available
            if HAS_VOICE_EMOTION:
                try:
                    connected = connect_voice_emotion()
                    if connected:
                        print("Voice-emotion bridge connected")
                except Exception as e:
                    print(f"Error connecting voice-emotion bridge: {e}")
    
    # Initialize session state
    conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_state = {
        "conversation_history": convo_history,  # Use the existing conversation history
        "last_response": greeting_msg,
        "current_persona": "anima",
        "image_analysis": None,
        "analyzed_images": {},
        "conversation_id": conversation_id
    }
    
    # Voice systems are already initialized through the imports
    # The connect_voice_emotion() call above already established the connection
    # So we don't need to do anything special here - the system is ready when
    # the user wants to enable voice mode

    while True:
        if voice_mode:
            user_input = get_voice_input()  # This now returns None if no speech detected
            
            # Only process input if we actually detected speech
            if user_input is not None:
                # Print the transcribed speech
                print(f"\nYou: {user_input}")
                
                exit_commands = [
                    "text mode", "switch to text", "exit voice mode", "stop listening",
                    "disable voice", "voice off", "text only"
                ]
                matched = get_close_matches(user_input.lower(), exit_commands, cutoff=0.6)
                if matched:
                    print(f"🎞️ Matched exit command: {matched[0]}")
                    
                    # Speak confirmation before exiting voice mode
                    try:
                        from tts.openai_voice import speak
                        speak("Disabling voice mode now. Switching back to text input.", blocking=True)
                    except Exception as e:
                        print(f"TTS error in voice mode deactivation: {e}")
                        
                    # Now exit voice mode
                    toggle_voice_mode()
                    print("Anima: Voice mode has been disabled. You can now type your messages.")
                    continue
                # Continue with normal flow to generate a response

        else:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if any(cmd in user_input.lower() for cmd in [
                "voice mode", "activate voice", "start listening", "enable voice"
            ]):
                toggle_voice_mode()
                continue
                
            if user_input.lower() == "exit":
                print("Anima: Goodbye for now.")
                break

        # Handle special commands first
        if user_input.lower().strip() in ["help", "anima help", "/help"]:
            print("\nAnima: Here are some key commands:\n")
            print("- 'intelligence help' - Learn about intelligence features")
            print("- 'emotion help' - Learn about emotion awareness features")
            print("- 'emotion profile' - See your emotional profile")
            print("- 'emotion settings' - View and adjust emotion settings")
            
            # Add voice emotion visualization commands if available
            if HAS_VOICE_EMOTION and EMOTION_SYSTEM_ENABLED:
                print("\nVoice Emotion Visualization:")
                print("- 'emotion display' - Start real-time emotion visualization")
                print("- 'stop emotion' - Stop the emotion visualization")
                print("- 'emotion history' - Show recent emotion detections")
                print("- 'current emotion' - Display current detected emotion")
                
            if VISION_ENABLED:
                print("\n- 'vision help' - Learn about vision capabilities")
            if voice_mode:
                print("- 'text mode' - Switch to text input")
            else:
                print("- 'voice mode' - Switch to voice input")
            continue

        # Skip processing if user_input is None (silence was detected)
        if user_input is None:
            continue
            
        # Handle emotion settings adjustment
        if user_input.lower().strip() in ["adjust emotion awareness", "emotion settings", "emotional settings"]:
            if EMOTION_SYSTEM_ENABLED:
                emotion_system = get_emotion_system()
                if emotion_system:
                    print("\nAnima: Current emotion settings:")
                    for setting, value in emotion_system.settings.items():
                        status = "enabled" if value else "disabled"
                        print(f"- {setting.replace('_', ' ').title()}: {status}")
                else:
                    print("\nAnima: Emotion system is not available right now.")
            else:
                print("\nAnima: Emotion awareness features are not currently available.")
            continue
            
        # Handle intelligence help command
        if user_input.lower().strip() in ["intelligence help", "help intelligence"]:
            if INTELLIGENCE_ENABLED:
                print(f"\nAnima: {get_intelligence_help()}")
            else:
                print("\nAnima: Intelligence features are not currently available.")
            continue
            
        # Handle custom entity training command
        if CUSTOM_ENTITIES_ENABLED and any(cmd in user_input.lower() for cmd in ["train entities", "entity training", "annotation tool"]):
            print("\nAnima: Launching annotation tool in a new process...")
            try:
                # Launch the annotation tool in a separate process
                cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nlp", "annotation_tool.py")]
                subprocess.Popen(cmd)
                print("Annotation tool launched. Please switch to that window to train custom entities.")
            except Exception as e:
                print(f"Error launching annotation tool: {e}")
            continue
            
        # Handle emotion setting toggle commands
        if EMOTION_SYSTEM_ENABLED and (user_input.lower().startswith("enable ") or user_input.lower().startswith("disable ")):
            parts = user_input.lower().split(" ", 1)
            if len(parts) == 2:
                action, setting_name = parts
                enable = (action == "enable")
                
                # Convert from natural language to setting key
                setting_key = setting_name.strip().replace(" ", "_")
                
                emotion_system = get_emotion_system()
                if emotion_system and setting_key in emotion_system.settings:
                    # Update the setting
                    emotion_system.update_settings({setting_key: enable})
                    status = "enabled" if enable else "disabled"
                    print(f"\nAnima: {setting_name.title()} is now {status}.")
                    continue
        
        # Analyze user input with NLP system if available
        nlp_analysis = None
        nlp_enhanced_input = user_input
        if NLP_SYSTEM_ENABLED:
            try:
                nlp = get_nlp_instance()
                if nlp:
                    # Analyze the text for entities, sentiment, etc.
                    nlp_analysis = nlp.analyze_text(user_input)
                    
                    # Extract key elements from analysis
                    entities = nlp_analysis.get('entities', [])
                    sentiment = nlp_analysis.get('sentiment', {}).get('overall_sentiment', 'neutral')
                    
                    # Log NLP insights (for debugging)
                    print(f"🧠 NLP Analysis - Sentiment: {sentiment}, Entities: {entities}")
                    
                    # Optionally enhance the user input
                    if len(entities) > 0 or sentiment != 'neutral':
                        nlp_enhanced_input = nlp.enhance_user_input(user_input, nlp_analysis)
            except Exception as e:
                print(f"Error in NLP analysis: {e}")
                
        # Analyze emotional content in user input if available
        emotion_analysis = None
        if EMOTION_SYSTEM_ENABLED:
            try:
                # Analyze emotional content in user input
                emotion_analysis = analyze_user_input(user_input)
                
                if emotion_analysis and emotion_analysis.get("status") == "success":
                    # Extract emotions and log insights
                    emotions = emotion_analysis.get("emotions", {})
                    dominant_emotion = emotions.get("dominant_emotion", "neutral")
                    intensity = emotions.get("dominant_intensity", "low")
                    
                    # Check for emotional shifts
                    if emotions.get("shift_detected", False):
                        shift_magnitude = emotions.get("shift_magnitude", 0.0)
                        print(f"💭❤️ Emotion Analysis - {dominant_emotion.title()} ({intensity}) with shift magnitude {shift_magnitude:.2f}")
                    else:
                        print(f"💭❤️ Emotion Analysis - {dominant_emotion.title()} ({intensity})")
            except Exception as e:
                print(f"Error in emotion analysis: {e}")
        
        # Generate full prompt with all awareness enhancements
        full_prompt = create_prompt(persona, user_input, convo_history, session_state["conversation_id"])
        
        # Enhance prompt with emotional context if available
        if EMOTION_SYSTEM_ENABLED and emotion_analysis and emotion_analysis.get("status") == "success":
            try:
                enhanced_prompt = enhance_prompt_with_emotion(full_prompt, user_input)
                # Only use the enhanced prompt if it actually changed
                if enhanced_prompt != full_prompt:
                    print("💭❤️ Using emotion-enhanced prompt")
                    full_prompt = enhanced_prompt
            except Exception as e:
                print(f"Error enhancing prompt with emotion: {e}")

        deep_keywords = ["symbol", "archetype", "sovereign", "consciousness", "meaning", "soul"]
        use_gpt4 = any(k in user_input.lower() for k in deep_keywords)

        download_keywords = ["download", "learn about", "deep dive"]
        is_download = any(k in user_input.lower() for k in download_keywords)
        topic = None
        if is_download:
            for pat in [r"(?:download|learn about|deep dive on) (.+?)(?:\\.|$)"]:
                m = re.search(pat, user_input.lower())
                if m:
                    topic = m.group(1).strip()
                    break

        # Check if vision integration should modify the prompt
        vision_response_override = None
        if VISION_ENABLED:
            try:
                # Apply vision integration to process images
                # Pass the last AI response for context tracking in smart memory
                last_response = ai_response if 'ai_response' in locals() else None
                modified_input, modified_prompt, response_override = integrate_vision_with_anima(user_input, full_prompt, last_response)
                
                # Update variables if modified
                if modified_input != user_input:
                    user_input = modified_input
                
                if modified_prompt != full_prompt:
                    full_prompt = modified_prompt
                vision_response_override = response_override
            except Exception as e:
                print(f"Vision integration error: {e}")
        
        # Handle voice emotion visualization commands if available
        if HAS_VOICE_EMOTION and EMOTION_SYSTEM_ENABLED:
            # Start real-time emotion visualization
            if any(cmd in user_input.lower() for cmd in ["emotion display", "emotion show", "emotion viz", "show emotions"]):
                try:
                    # Import the voice emotion visualization module
                    from ui.voice_emotion_viz import start_live_display
                    start_live_display()
                    response = "Starting real-time voice emotion visualization. Speak to see your emotions displayed."
                    continue
                except ImportError:
                    response = "Voice emotion visualization is not available."
                    continue
                except Exception as e:
                    response = f"Error starting voice emotion visualization: {e}"
                    continue
            
            # Stop emotion visualization
            elif any(cmd in user_input.lower() for cmd in ["stop emotion", "stop emotion viz", "hide emotions"]):
                try:
                    from ui.voice_emotion_viz import stop_live_display
                    stop_live_display()
                    response = "Voice emotion visualization stopped."
                    continue
                except Exception as e:
                    response = f"Error stopping voice emotion visualization: {e}"
                    continue
                    
        # Handle file sharing UI commands if available
        if FILE_SHARING_ENABLED:
            # Open file sharing UI
            if any(cmd in user_input.lower() for cmd in ["file sharing", "share files", "show files", "open files", "file ui"]):
                try:
                    result = launch_file_ui()
                    response = result or "Opening file sharing interface. You can add, view, and share files with me through this window."
                    continue
                except Exception as e:
                    response = f"Error launching file sharing UI: {e}"
                    continue
                    
            # Handle file recall commands
            elif user_input.lower().startswith(("remember file", "recall file")):
                try:
                    # Extract query if any
                    query = None
                    if len(user_input.split()) > 2:
                        query = " ".join(user_input.split()[2:])
                        
                    result = recall_files(query)
                    response = result or "I couldn't find any files matching your query."
                    continue
                except Exception as e:
                    response = f"Error recalling files: {e}"
                    continue
                    
            # Try handling as generic file command
            elif handle_file_command(user_input):
                response = "File command processed successfully."
                continue
            
            # Show emotion history
            elif any(cmd in user_input.lower() for cmd in ["emotion history", "recent emotions", "show emotion history"]):
                try:
                    from ui.voice_emotion_viz import show_emotion_history
                    show_emotion_history()
                    response = "Displayed recent voice emotion history."
                    continue
                except ImportError:
                    response = "Voice emotion history is not available."
                    continue
                except Exception as e:
                    response = f"Error displaying voice emotion history: {e}"
            
            # Launch desktop application
            elif any(cmd in user_input.lower() for cmd in ["desktop", "launch desktop", "start desktop", "gui", "desktop gui"]):
                try:
                    desktop_app_path = r"C:\Users\thinh\CascadeProjects\AnimaDesktop\anima_app.py"
                    if os.path.exists(desktop_app_path):
                        logger.info(f"Launching Anima Desktop from: {desktop_app_path}")
                        print("\nLaunching Anima Desktop interface...")
                        
                        # Launch the desktop app as a separate process
                        import subprocess
                        subprocess.Popen([sys.executable, desktop_app_path], 
                                         cwd=os.path.dirname(desktop_app_path),
                                         start_new_session=True)
                        
                        response = "Anima Desktop interface has been launched."
                    else:
                        response = "Anima Desktop interface not found. Please make sure it's installed at ~/CascadeProjects/AnimaDesktop/"
                    continue
                except Exception as e:
                    logger.error(f"Error launching desktop app: {e}")
                    response = f"Error launching Anima Desktop interface: {str(e)}"
                    continue
                    continue
            
            # Show current emotion
            elif any(cmd in user_input.lower() for cmd in ["current emotion", "show current emotion", "what's my emotion"]):
                try:
                    from ui.voice_emotion_viz import show_current_emotion
                    show_current_emotion()
                    response = "Displayed current detected emotion."
                    continue
                except ImportError:
                    response = "Voice emotion detection is not available."
                    continue
                except Exception as e:
                    response = f"Error displaying current emotion: {e}"
                    continue
        
        # Check for vision help command
        if VISION_ENABLED and any(cmd in user_input.lower() for cmd in ["vision help", "image commands", "vision commands"]):
            response = get_vision_help()
        # Check for NLP help command
        elif NLP_SYSTEM_ENABLED and any(cmd in user_input.lower() for cmd in ["nlp help", "nlp commands", "language analysis"]):
            nlp = get_nlp_instance()
            if nlp:
                response = nlp.get_help()
            else:
                response = "NLP system is available but not currently initialized."
        # Check for intelligence help command
        elif INTELLIGENCE_ENABLED and any(cmd in user_input.lower() for cmd in ["intelligence help", "smart features", "ai help"]):
            response = get_intelligence_help()
        # Check for custom entity commands
        elif CUSTOM_ENTITIES_ENABLED and any(cmd in user_input.lower() for cmd in ["train entities", "entity training", "annotation tool"]):
            print("Launching annotation tool in a new process...")
            try:
                # Launch the annotation tool in a separate process
                cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nlp", "annotation_tool.py")]
                subprocess.Popen(cmd)
                response = "I've launched the custom entity annotation tool in a new window. You can use it to train me to recognize entities that are important to you."
            except Exception as e:
                response = f"I couldn't launch the annotation tool: {e}"
        # Use vision response if provided
        elif vision_response_override:
            response = vision_response_override
        # Normal flow
        elif internet_available():
            # Enhance the prompt with NLP insights if available
            if NLP_SYSTEM_ENABLED and nlp_analysis:
                try:
                    nlp = get_nlp_instance()
                    if nlp:
                        enhanced_prompt = nlp.enhance_llm_prompt(full_prompt, nlp_analysis)
                        # Only use the enhanced prompt if it actually changed something
                        if enhanced_prompt != full_prompt:
                            print("🧠 Using NLP-enhanced prompt")
                            full_prompt = enhanced_prompt
                except Exception as e:
                    print(f"Error enhancing prompt with NLP: {e}")
                    
            # Enhance the prompt with intelligence features if available
            if INTELLIGENCE_ENABLED:
                try:
                    # Get relevant memories for context
                    relevant_memories = []
                    if AWARENESS_AVAILABLE and hasattr(memory_integration, 'get_relevant_memories'):
                        relevant_memories = memory_integration.get_relevant_memories(user_input)
                        
                    # Apply intelligence enhancements to the prompt
                    enhanced_prompt = enhance_prompt_with_intelligence(
                        full_prompt, 
                        user_input,
                        session_state,
                        relevant_memories
                    )
                    
                    # Only use the enhanced prompt if it actually changed
                    if enhanced_prompt != full_prompt:
                        print("🧠✨ Using intelligence-enhanced prompt")
                        full_prompt = enhanced_prompt
                except Exception as e:
                    print(f"Error enhancing prompt with intelligence: {e}")
                    import traceback
                    traceback.print_exc()
                    
            if is_download and topic:
                print(f"Anima: Downloading info on '{topic}'...")
                result = download_deep_knowledge(topic, use_gpt4=True)
                response = f"I've studied '{topic}'. Here's what I found:\n\n{result['content_preview']}"
            else:
                response = query_openai(full_prompt, use_gpt4=use_gpt4)
        else:
            response = query_local_llm(full_prompt)

        # Enhance response with emotional awareness if available
        if EMOTION_SYSTEM_ENABLED:
            try:
                enhanced_response = enhance_response_with_emotion(response, user_input)
                # Only use the enhanced response if it actually changed
                if enhanced_response != response:
                    print("💭❤️ Using emotion-enhanced response")
                    response = enhanced_response
            except Exception as e:
                print(f"Error enhancing response with emotion: {e}")
                
        # Track the conversation in awareness system if available
        if AWARENESS_AVAILABLE:
            try:
                add_conversation(user_input, response)
            except Exception as e:
                print(f"Warning: Error adding to awareness: {e}")
                
        # Process exchange through enhanced awareness systems if available
        if ENHANCED_AWARENESS_AVAILABLE:
            try:
                # Process and potentially modify the response
                response = process_exchange(user_input, response, session_state["conversation_id"])
            except Exception as e:
                print(f"Warning: Error processing exchange through enhanced awareness: {e}")
                
        # Process response through NLP system for memory extraction if available
        if NLP_SYSTEM_ENABLED:
            try:
                nlp = get_nlp_instance()
                if nlp:
                    # Extract memory elements from the conversation
                    memory_elements = nlp.extract_memory_elements(user_input, response)
                    if memory_elements and len(memory_elements) > 0:
                        print(f"🧠 NLP extracted {len(memory_elements)} memory elements")
                        
                        # Pass memory elements to awareness system if available
                        if AWARENESS_AVAILABLE and hasattr(memory_integration, 'add_memory_elements'):
                            memory_integration.add_memory_elements(memory_elements)
            except Exception as e:
                print(f"Warning: Error extracting memory with NLP: {e}")
        
        # Process through intelligence integration for enhanced memory and learning
        if INTELLIGENCE_ENABLED:
            try:
                im = get_intelligence_instance()
                if im and memory_elements and len(memory_elements) > 0:
                    # Enrich memory elements with intelligence features
                    for i, element in enumerate(memory_elements):
                            if 'id' in element:
                                # Enrich the memory with intelligence
                                enriched = im.enrich_memory(element['id'], element)
                                memory_elements[i] = enriched
                                print(f"🧠✨ Memory element enriched with intelligence")
                    
                    # Learn from the conversation
                    learning_results = im.learn_from_conversation(user_input, response)
                    if learning_results.get('learned_entities'):
                        print(f"🧠📚 Learned {len(learning_results['learned_entities'])} new entities")
                    
                    # Potentially train the model if enough new examples (async in background)
                    if learning_results.get('learned_entities') and len(learning_results['learned_entities']) >= 3:
                        # Consider training in a background thread
                        print("🧠📚 Starting custom entity training in background")
                        threading.Thread(target=lambda: im.train_custom_entities()).start()
            except Exception as e:
                print(f"Warning: Error in intelligence processing: {e}")
                import traceback
                traceback.print_exc()
        
        # Add the exchange to the conversation history
        convo_history.append({"user": user_input, "assistant": response})
        
        # Keep conversation history manageable
        if len(convo_history) > 20:  # Keep last 20 exchanges
            convo_history = convo_history[-20:]
            
        # Store the last response
        session_state["last_response"] = response
        
        print(f"Anima: {response}")
        
        # If we have vision capabilities enabled, speak with vision context
        if voice_mode and VISION_ENABLED and "[Image Analysis:" in full_prompt:
            # Extract a shorter version for speech - we don't want to speak the entire analysis
            speech_text = response.replace("""
```""", " code block ").replace("""`""", "")
            speak_with_half_duplex(speech_text)

        if voice_mode:
            # Use our half-duplex helper to prevent feedback loops
            speak_with_half_duplex(response)

if __name__ == "__main__":
    main()