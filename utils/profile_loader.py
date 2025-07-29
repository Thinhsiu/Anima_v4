"""
Utility to load user profile information from a JSON file.
This information is used to personalize Anima's interactions.
"""
import os
import json

# Default location for the user profile
DEFAULT_PROFILE_PATH = "data/user_profile.json"

def load_user_profile(profile_path=None):
    """
    Load user profile information from a JSON file.
    
    Args:
        profile_path (str, optional): Path to the user profile JSON file. 
                                     Defaults to data/user_profile.json.
    
    Returns:
        dict: User profile data or empty dict if file not found.
    """
    path_to_use = profile_path or DEFAULT_PROFILE_PATH
    
    try:
        if os.path.exists(path_to_use):
            with open(path_to_use, 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            print(f"User profile not found at {path_to_use}. Using default persona.")
            return {}
    except json.JSONDecodeError:
        print(f"Error parsing user profile JSON at {path_to_use}. Using default persona.")
        return {}
    except Exception as e:
        print(f"Error loading user profile: {e}. Using default persona.")
        return {}

def get_profile_prompt(profile_data):
    """
    Generate a prompt section based on the user profile data.
    
    Args:
        profile_data (dict): User profile information.
        
    Returns:
        str: Formatted prompt section with user information.
    """
    if not profile_data:
        return ""
    
    # Create a formatted string from user profile data
    prompt_parts = []
    
    if 'name' in profile_data:
        prompt_parts.append(f"The user's name is {profile_data['name']}.")
    
    if 'preferences' in profile_data:
        prefs = profile_data['preferences']
        pref_items = [f"{key}: {value}" for key, value in prefs.items()]
        prompt_parts.append("User preferences: " + ", ".join(pref_items))
    
    if 'background' in profile_data:
        prompt_parts.append(f"User background: {profile_data['background']}")
    
    if 'interests' in profile_data:
        interests = profile_data['interests']
        if isinstance(interests, list):
            prompt_parts.append("User interests: " + ", ".join(interests))
        else:
            prompt_parts.append(f"User interests: {interests}")
    
    # Add any other custom fields
    for key, value in profile_data.items():
        if key not in ['name', 'preferences', 'background', 'interests']:
            if isinstance(value, (str, int, float, bool)):
                prompt_parts.append(f"{key.capitalize()}: {value}")
    
    return "\n".join(prompt_parts)

# Example of expected user profile format:
"""
{
    "name": "John Doe",
    "preferences": {
        "response_style": "detailed",
        "topics_of_interest": "technology, philosophy, science"
    },
    "background": "Software engineer with 10 years experience",
    "interests": ["AI", "robotics", "music production"],
    "custom_field": "Custom information about the user"
}
"""
