import os
import sys
import time
import datetime
import re
import warnings
import random
import logging
import json
import base64
import subprocess
import threading
from difflib import get_close_matches
from pathlib import Path
import importlib

# Suppress pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- Silent startup (kept) ---
try:
    from silent_startup import enable_silent_mode, disable_silent_mode
    silent_filter = enable_silent_mode()
    SILENT_MODE_ENABLED = True
except ImportError:
    SILENT_MODE_ENABLED = False
    print("Silent startup module not found, continuing with normal output")

# --- Environment setup ---
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

# --- Smart memory manager ---
try:
    from vision.smart_memory_manager import smart_memory_manager
    SMART_MEMORY_ENABLED = True
except ImportError:
    print("Smart memory management not available")
    SMART_MEMORY_ENABLED = False

# --- NLP system ---
try:
    from nlp.integration import initialize as initialize_nlp, get_instance as get_nlp_instance
    NLP_SYSTEM_ENABLED = True
    initialize_nlp()
except ImportError as e:
    print(f"NLP system not available: {e}")
    NLP_SYSTEM_ENABLED = False
    def get_nlp_instance(): return None

# --- Intelligence system ---
try:
    from nlp.intelligence_integration import (
        initialize as initialize_intelligence,
        get_instance as get_intelligence_instance,
        get_intelligence_help,
        enhance_prompt_with_intelligence,
    )
    INTELLIGENCE_ENABLED = True
    initialize_intelligence()
except ImportError:
    INTELLIGENCE_ENABLED = False
    def get_intelligence_instance(): return None
    def get_intelligence_help(): return "Intelligence features not available."
    def enhance_prompt_with_intelligence(p, *a, **kw): return p

# --- Custom entities ---
try:
    from nlp.custom_entities import get_instance as get_entity_trainer
    from nlp.annotation_tool import AnnotationTool
    CUSTOM_ENTITIES_ENABLED = True
except ImportError:
    CUSTOM_ENTITIES_ENABLED = False

# --- Emotion awareness ---
try:
    from emotion.integration import (
        initialize as initialize_emotion_system,
        get_instance as get_emotion_system,
        analyze_user_input,
        enhance_response as enhance_response_with_emotion,
        enhance_prompt as enhance_prompt_with_emotion,
        get_emotion_help,
    )
    EMOTION_SYSTEM_ENABLED = True
except ImportError:
    EMOTION_SYSTEM_ENABLED = False
    def initialize_emotion_system(*a, **kw): return None
    def get_emotion_system(): return None
    def analyze_user_input(t): return {}
    def enhance_response_with_emotion(r, u): return r
    def enhance_prompt_with_emotion(p, u): return p
    def get_emotion_help(): return "Emotion awareness features are not available."

# --- Awareness & memory integration ---
AWARENESS_AVAILABLE = False
try:
    from core import awareness, add_conversation, enhance_prompt_with_awareness
    from core import memory_integration, enhance_prompt_with_memory
    AWARENESS_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        from core.awareness import awareness, add_conversation
        from core.memory_integration import memory_integration
        if hasattr(awareness, 'enhance_prompt'):
            from core.awareness import enhance_prompt as enhance_prompt_with_awareness
        else:
            def enhance_prompt_with_awareness(p): return p
        if hasattr(memory_integration, 'enhance_prompt_with_memories'):
            from core.memory_integration import enhance_prompt_with_memories as enhance_prompt_with_memory
        else:
            def enhance_prompt_with_memory(p, u): return p
        AWARENESS_AVAILABLE = True
    except ImportError:
        def add_conversation(*a, **kw): pass
        def enhance_prompt_with_awareness(*a, **kw): return a[0] if a else ""
        def enhance_prompt_with_memory(*a, **kw): return a[0] if a else ""

# --- Enhanced awareness ---
try:
    from core.enhanced_awareness_integration import (
        enhance_prompt_with_all,
        process_exchange,
        get_system_status,
        check_modules,
    )
    ENHANCED_AWARENESS_AVAILABLE = True
    check_modules()
except ImportError:
    ENHANCED_AWARENESS_AVAILABLE = False
    def enhance_prompt_with_all(*a, **kw): return a[0] if a else ""
    def process_exchange(*a, **kw): return a[1] if len(a) > 1 else ""
    def get_system_status(): return {"available_modules": {}}
    def check_modules(): pass

# --- Vision integration ---
try:
    from vision_integration import integrate_vision_with_anima, get_help as get_vision_help
    VISION_ENABLED = True
except ImportError:
    VISION_ENABLED = False

# --- File sharing integration ---
FILE_SHARING_ENABLED = False
try:
    from ui.file_sharing_integration import handle_file_command, launch_file_ui, recall_files
    FILE_SHARING_ENABLED = True
except ImportError:
    FILE_SHARING_ENABLED = False

# --- LLMs & utilities ---
from llm.openai_llm import query_openai
from llm.local_llm import query_local_llm
from utils.persona_loader import generate_complete_system_prompt, load_complete_persona
from utils.knowledge_manager import download_deep_knowledge, get_knowledge, get_knowledge_stats

# --- Removed all voice / audio imports ---
# from stt.vad import start_smart_listening, stop_smart_listening, get_speech_monitor
# from tts.openai_voice import speak
# from emotion.voice_bridge import connect_to_voice_system
# import pyaudio

# --- Voice completely disabled ---
VOICE_SYSTEM_ENABLED = False
def toggle_voice_mode(*a, **kw):
    print("[Voice system disabled on Render-safe build]")
    return False
    # -------------------------------------------------------------------------
#  Voice and Audio System — STRIPPED for Render-safe build
# -------------------------------------------------------------------------

def start_voice_recognition():
    """Stub - voice recognition disabled."""
    print("[Voice recognition disabled]")
    return False

def stop_voice_recognition():
    """Stub - no cleanup needed."""
    pass

def start_smart_duplex():
    """Stub - smart duplex disabled."""
    print("[Smart duplex disabled]")
    return False

def stop_smart_duplex():
    """Stub - smart duplex disabled."""
    pass

def get_voice_input():
    """Fallback to text input since voice mode is removed."""
    return input("You: ")

def speak_with_half_duplex(text):
    """Stub - prints text instead of speaking."""
    print(f"Anima (text-only): {text}")

# -------------------------------------------------------------------------
#  Utility and Persona systems (intact)
# -------------------------------------------------------------------------

def load_thoth():
    """Load base Thoth prompt safely."""
    thoth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona", "thoth.txt")
    if os.path.exists(thoth_path):
        with open(thoth_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def internet_available():
    try:
        import urllib.request
        urllib.request.urlopen("http://google.com", timeout=2)
        return True
    except:
        return False

def generate_complete_system_prompt():
    """Rebuild persona system prompts (untouched)."""
    persona_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona")
    identity_path = os.path.join(persona_dir, "identity.json")
    thinh_profile_path = os.path.join(persona_dir, "thinh_profile.json")
    prompt_parts = []

    if os.path.exists(identity_path):
        with open(identity_path, "r", encoding="utf-8") as f:
            identity = json.load(f)
        if "persona" in identity:
            prompt_parts.append(identity["persona"])
        if "tone" in identity:
            prompt_parts.append(f"Tone: {identity['tone']}")
        if "personality" in identity:
            if isinstance(identity["personality"], list):
                prompt_parts.append("Personality traits: " + ", ".join(identity["personality"]))
            else:
                prompt_parts.append(f"Personality: {identity['personality']}")
        if "core_values" in identity:
            prompt_parts.append("Core values: " + ", ".join(identity["core_values"]))

    if os.path.exists(thinh_profile_path):
        with open(thinh_profile_path, "r", encoding="utf-8") as f:
            thinh_data = json.load(f)
        if "anima_fusion_date" in thinh_data:
            prompt_parts.append(f"Created on {thinh_data['anima_fusion_date']}")
        if "creator_of" in thinh_data:
            prompt_parts.append(f"You are {thinh_data['creator_of']}")

    return "\n\n".join(prompt_parts) if prompt_parts else "You are Anima, created by Thinh."

def load_complete_persona():
    """Full persona loader (unchanged)."""
    persona = {}
    persona_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persona")
    identity_path = os.path.join(persona_dir, "identity.json")
    knowledge_path = os.path.join(persona_dir, "knowledge_vault.json")
    thinh_profile_path = os.path.join(persona_dir, "thinh_profile.json")
    origin_story_path = os.path.join(persona_dir, "anima_origin_story.json")

    if os.path.exists(identity_path):
        with open(identity_path, "r", encoding="utf-8") as f:
            persona["identity"] = json.load(f)
    if os.path.exists(knowledge_path):
        with open(knowledge_path, "r", encoding="utf-8") as f:
            persona["knowledge"] = json.load(f)
    if os.path.exists(thinh_profile_path):
        with open(thinh_profile_path, "r", encoding="utf-8") as f:
            persona["thinh_profile"] = json.load(f)
            persona.update(persona["thinh_profile"])
    if os.path.exists(origin_story_path):
        with open(origin_story_path, "r", encoding="utf-8") as f:
            persona["origin_story"] = json.load(f)

    persona.setdefault("user", {"full_name": "Thinh"})
    return persona

def create_prompt(persona, user_input, convo_history, conversation_id=None):
    """Prompt creation preserved (no audio)."""
    base_prompt = load_thoth()
    persona_prompt = generate_complete_system_prompt()
    system_prompt = f"{base_prompt}\n\n{persona_prompt}"

    context = ""
    if "short_term_memory" in persona and persona["short_term_memory"]:
        context = "Recent context: " + str(persona["short_term_memory"])

    convo_history_str = "\n".join([
        f"User: {item['user']}\nAssistant: {item['assistant']}"
        for item in convo_history
    ])

    user_prompt = f"{context}\n\nConversation History:\n{convo_history_str}\n\nUser: {user_input}"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    if AWARENESS_AVAILABLE:
        try:
            full_prompt = enhance_prompt_with_awareness(full_prompt, user_input)
            full_prompt = enhance_prompt_with_memory(full_prompt, user_input)
        except Exception as e:
            print(f"Awareness enhance error: {e}")

    if ENHANCED_AWARENESS_AVAILABLE:
        try:
            full_prompt = enhance_prompt_with_all(full_prompt, user_input, conversation_id)
        except Exception as e:
            print(f"Enhanced awareness error: {e}")

    return {"system_prompt": system_prompt, "user_prompt": user_prompt, "full_prompt": full_prompt}

# -------------------------------------------------------------------------
#  Main program (Render-safe, text-only)
# -------------------------------------------------------------------------

def main():
    # Suppress noise during startup
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if SILENT_MODE_ENABLED:
        try:
            disable_silent_mode(silent_filter, restore_stdout=True)
        except Exception:
            pass

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

    user_name = persona.get("user", {}).get("full_name", "there").split()[0]
    greet_templates = [
        f"{greeting}, {user_name}! How can I help you this {time_context}?",
        f"Hey {user_name}! Hope your {time_context} is going well.",
        f"Hi {user_name}, ready to get started?",
        f"{greeting}, {user_name}! I’m here and focused."
    ]
    greeting_msg = random.choice(greet_templates)
    print(f"Anima: {greeting_msg}")
    print("(Type 'exit' to quit.)")

    # Initialize optional subsystems
    if NLP_SYSTEM_ENABLED:
        nlp = get_nlp_instance()
        if nlp:
            print("NLP system initialized.")
    if INTELLIGENCE_ENABLED:
        initialize_intelligence()
        print("Intelligence system initialized.")
    if EMOTION_SYSTEM_ENABLED:
        em = initialize_emotion_system()
        if em:
            print("Emotion system initialized.")

    session_state = {
        "conversation_history": convo_history,
        "last_response": greeting_msg,
        "current_persona": "anima",
        "conversation_id": f"conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    # ---------------------------------------------------------------------
    #  Main text loop
    # ---------------------------------------------------------------------
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Anima: Goodbye for now.")
            break

        # --- Built-in help ---
        if user_input.lower() in ["help", "/help", "anima help"]:
            print("\nAnima: Commands available:")
            print("- 'intelligence help' → details on intelligence features")
            print("- 'emotion help' → learn about emotion awareness")
            print("- 'vision help' → vision-related info (if enabled)")
            continue

        # --- Emotion system help ---
        if user_input.lower() == "emotion help":
            print(f"\nAnima: {get_emotion_help()}")
            continue

        # --- Intelligence system help ---
        if user_input.lower() == "intelligence help":
            print(f"\nAnima: {get_intelligence_help()}")
            continue

        # --- NLP analysis (if enabled) ---
        if NLP_SYSTEM_ENABLED:
            try:
                nlp = get_nlp_instance()
                if nlp:
                    analysis = nlp.analyze_text(user_input)
                    sentiment = analysis.get("sentiment", {}).get("overall_sentiment", "neutral")
                    print(f"🧠 Sentiment: {sentiment}")
            except Exception as e:
                print(f"NLP analysis error: {e}")

        # --- Emotion analysis (if enabled) ---
        if EMOTION_SYSTEM_ENABLED:
            try:
                emotions = analyze_user_input(user_input)
                if emotions and "emotions" in emotions:
                    dom = emotions["emotions"].get("dominant_emotion", "neutral")
                    print(f"💭 Emotion: {dom}")
            except Exception as e:
                print(f"Emotion analysis error: {e}")

        # --- Build prompt and get LLM response ---
        prompt_data = create_prompt(persona, user_input, convo_history, session_state["conversation_id"])
        full_prompt = prompt_data["full_prompt"]
        system_prompt = prompt_data["system_prompt"]

        use_gpt4 = any(k in user_input.lower() for k in ["symbol", "archetype", "meaning", "soul"])
        try:
            if internet_available():
                response = query_openai(prompt_data["user_prompt"], use_gpt4=use_gpt4, system_prompt=system_prompt)
            else:
                response = query_local_llm(full_prompt)
        except Exception as e:
            response = f"Error generating response: {e}"

        # --- Emotion enhancement ---
        if EMOTION_SYSTEM_ENABLED:
            try:
                response = enhance_response_with_emotion(response, user_input)
            except Exception:
                pass

        # --- Awareness logging ---
        if AWARENESS_AVAILABLE:
            try:
                add_conversation(user_input, response)
            except Exception:
                pass

        if ENHANCED_AWARENESS_AVAILABLE:
            try:
                response = process_exchange(user_input, response, session_state["conversation_id"])
            except Exception:
                pass

        convo_history.append({"user": user_input, "assistant": response})
        if len(convo_history) > 20:
            convo_history = convo_history[-20:]

        session_state["last_response"] = response
        print(f"Anima: {response}")


if __name__ == "__main__":
    main()
