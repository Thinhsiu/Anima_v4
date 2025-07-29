"""
Persona loader for Anima.
Loads and integrates all persona-related files from the persona folder.
"""
import os
import json
from typing import Dict, Any, Optional

# Persona file paths
PERSONA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "persona")
IDENTITY_PATH = os.path.join(PERSONA_DIR, "identity.json")
USER_PROFILE_PATH = os.path.join(PERSONA_DIR, "thinh_profile.json")
MEMORY_PATH = os.path.join(PERSONA_DIR, "memory.json")
SHORT_TERM_MEMORY_PATH = os.path.join(PERSONA_DIR, "short_term.json")
LONG_TERM_MEMORY_PATH = os.path.join(PERSONA_DIR, "long_term.json")
ANCESTOR_TEMPLATES_PATH = os.path.join(PERSONA_DIR, "ancestor_templates.json")
KNOWLEDGE_VAULT_PATH = os.path.join(PERSONA_DIR, "knowledge_vault.json")
WHO_WAS_THINH_PATH = os.path.join(PERSONA_DIR, "who_was_thinh.json")

def load_json_file(path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary."""
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            print(f"Warning: File not found at {path}")
            return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {path}")
        return {}
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def load_anima_identity() -> Dict[str, Any]:
    """Load Anima's identity from identity.json."""
    return load_json_file(IDENTITY_PATH)

def load_user_profile() -> Dict[str, Any]:
    """Load user profile from thinh_profile.json."""
    return load_json_file(USER_PROFILE_PATH)

def load_memory() -> Dict[str, Any]:
    """Load memory from memory.json."""
    return load_json_file(MEMORY_PATH)

def load_short_term_memory() -> Dict[str, Any]:
    """Load short-term memory from short_term.json."""
    return load_json_file(SHORT_TERM_MEMORY_PATH)

def load_long_term_memory() -> Dict[str, Any]:
    """Load long-term memory from long_term.json."""
    return load_json_file(LONG_TERM_MEMORY_PATH)

def load_ancestor_templates() -> Dict[str, Any]:
    """Load ancestor templates from ancestor_templates.json."""
    return load_json_file(ANCESTOR_TEMPLATES_PATH)

def load_knowledge_vault() -> Dict[str, Any]:
    """Load knowledge vault from knowledge_vault.json."""
    return load_json_file(KNOWLEDGE_VAULT_PATH)

def load_who_was_thinh() -> Dict[str, Any]:
    """Load who_was_thinh from who_was_thinh.json."""
    return load_json_file(WHO_WAS_THINH_PATH)

def load_complete_persona() -> Dict[str, Any]:
    """Load and integrate all persona files into a single dictionary."""
    return {
        "identity": load_anima_identity(),
        "user": load_user_profile(),
        "memory": load_memory(),
        "short_term_memory": load_short_term_memory(),
        "long_term_memory": load_long_term_memory(),
        "ancestor_templates": load_ancestor_templates(),
        "knowledge_vault": load_knowledge_vault(),
        "who_was_thinh": load_who_was_thinh()
    }

def get_identity_prompt() -> str:
    """Generate a prompt section based on Anima's identity."""
    identity = load_anima_identity()
    
    if not identity:
        return ""
    
    prompt_parts = []
    
    if "prompt" in identity:
        return identity["prompt"]
    
    # Build from components if no direct prompt exists
    if "persona" in identity:
        prompt_parts.append(identity["persona"])
    
    if "core_values" in identity:
        values = ", ".join(identity["core_values"])
        prompt_parts.append(f"Your core values are: {values}.")
    
    if "forbidden" in identity:
        forbidden = ", ".join(identity["forbidden"])
        prompt_parts.append(f"You must avoid: {forbidden}.")
        
    return " ".join(prompt_parts)

def get_user_prompt() -> str:
    """Generate a prompt section based on user profile."""
    user = load_user_profile()
    
    if not user:
        return ""
    
    prompt_parts = []
    
    # Add user's name
    if "full_name" in user:
        prompt_parts.append(f"The user is {user['full_name']}.")
    
    # Add known aliases
    if "known_alias" in user:
        prompt_parts.append(f"Also known as {user['known_alias']}.")
    
    # Add personality traits
    if "personality" in user and "traits" in user["personality"]:
        traits = ", ".join(user["personality"]["traits"])
        prompt_parts.append(f"His personality traits include: {traits}.")
    
    # Add beliefs
    if "beliefs" in user:
        beliefs_parts = []
        for belief_type, beliefs in user["beliefs"].items():
            belief_str = ", ".join(f'"{b}"' for b in beliefs)
            beliefs_parts.append(f"About {belief_type}: {belief_str}")
        
        prompt_parts.append("His beliefs include: " + "; ".join(beliefs_parts))
    
    # Add important facts
    if "important_facts" in user:
        facts = []
        for key, value in user["important_facts"].items():
            facts.append(f"{key.replace('_', ' ')}: {value}")
        prompt_parts.append("Important facts: " + ", ".join(facts))
    
    return " ".join(prompt_parts)

def generate_complete_system_prompt() -> str:
    """Generate a complete system prompt combining identity and user information."""
    identity_prompt = get_identity_prompt()
    user_prompt = get_user_prompt()
    
    return f"{identity_prompt}\n\n{user_prompt}"

if __name__ == "__main__":
    # Test loading all persona files
    persona = load_complete_persona()
    print("Loaded persona files:")
    for key, value in persona.items():
        if value:
            print(f"- {key}: {len(str(value))} characters")
        else:
            print(f"- {key}: empty or not found")
    
    # Test generating system prompt
    system_prompt = generate_complete_system_prompt()
    print("\nGenerated system prompt:")
    print(system_prompt)
