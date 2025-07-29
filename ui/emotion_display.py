"""
UI component for displaying real-time voice emotion data
within the Anima AI console interface.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ui.emotion")

# Color mappings for different emotions (ANSI color codes)
EMOTION_COLORS = {
    "happy": "\033[38;5;220m",  # Yellow
    "sad": "\033[38;5;33m",     # Blue
    "angry": "\033[38;5;196m",  # Red
    "surprised": "\033[38;5;214m", # Orange
    "fearful": "\033[38;5;135m", # Purple
    "disgusted": "\033[38;5;34m", # Green
    "neutral": "\033[38;5;250m", # Light Gray
    "calm": "\033[38;5;39m",    # Light Blue
    # Default color for unknown emotions
    "default": "\033[38;5;255m"  # White
}

# Reset color code
RESET_COLOR = "\033[0m"

# Emotion symbols for visualization
EMOTION_SYMBOLS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "surprised": "😮",
    "fearful": "😨",
    "disgusted": "🤢",
    "neutral": "😐",
    "calm": "😌",
    # Default symbol for unknown emotions
    "default": "❓"
}

# Intensity indicators
INTENSITY_INDICATORS = {
    "very_low": "▁",
    "low": "▂",
    "medium": "▃▄",
    "high": "▅▆",
    "very_high": "▇█"
}

def map_confidence_to_intensity(confidence: float) -> str:
    """Map confidence score to intensity level"""
    if confidence < 0.2:
        return "very_low"
    elif confidence < 0.4:
        return "low"
    elif confidence < 0.6:
        return "medium"
    elif confidence < 0.8:
        return "high"
    else:
        return "very_high"

def get_emotion_color(emotion: str) -> str:
    """Get the ANSI color code for a given emotion"""
    return EMOTION_COLORS.get(emotion.lower(), EMOTION_COLORS["default"])

def get_emotion_symbol(emotion: str) -> str:
    """Get the emoji symbol for a given emotion"""
    return EMOTION_SYMBOLS.get(emotion.lower(), EMOTION_SYMBOLS["default"])

def display_emotion_badge(emotion: str, confidence: float, duration_ms: Optional[int] = None) -> str:
    """
    Create a colorful badge displaying the emotion and its confidence.
    
    Args:
        emotion: The detected emotion label
        confidence: Confidence score (0.0-1.0)
        duration_ms: Optional duration of the audio in milliseconds
        
    Returns:
        Formatted string for console display
    """
    color = get_emotion_color(emotion)
    symbol = get_emotion_symbol(emotion)
    intensity = map_confidence_to_intensity(confidence)
    intensity_bar = INTENSITY_INDICATORS[intensity]
    
    # Format confidence as percentage
    conf_pct = int(confidence * 100)
    
    # Create badge with emotion name, symbol, and confidence
    badge = f"{color}{symbol} {emotion.title()} {intensity_bar} {conf_pct}%{RESET_COLOR}"
    
    # Add duration if provided
    if duration_ms is not None:
        duration_sec = duration_ms / 1000
        badge += f" ({duration_sec:.1f}s)"
        
    return badge

def display_emotion_panel(emotion_data: Dict[str, Any]) -> str:
    """
    Create a multi-line panel displaying detailed emotion analysis.
    
    Args:
        emotion_data: Dictionary containing emotion analysis data
        
    Returns:
        Multi-line string for console display
    """
    if not emotion_data or not isinstance(emotion_data, dict):
        return ""
        
    dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
    confidence = emotion_data.get("confidence", 0.0)
    top_scores = emotion_data.get("top_scores", {})
    
    # Create header with dominant emotion
    header = display_emotion_badge(dominant_emotion, confidence)
    
    # Create bars for top emotions
    lines = [header, ""]
    
    # Sort emotions by confidence score
    sorted_emotions = sorted(top_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Display top 3 emotions with colored bars
    for emotion, score in sorted_emotions[:3]:
        color = get_emotion_color(emotion)
        symbol = get_emotion_symbol(emotion)
        bar_length = int(score * 20)  # Scale to 20 characters max
        bar = "█" * bar_length
        pct = int(score * 100)
        lines.append(f"{color}{symbol} {emotion.title().ljust(10)} {bar} {pct}%{RESET_COLOR}")
    
    return "\n".join(lines)

def display_voice_emotion_indicator(emotion_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Display a real-time voice emotion indicator in the console.
    
    Args:
        emotion_data: Dictionary containing emotion analysis data
        
    This function prints the emotion indicator directly to the console.
    """
    if not emotion_data:
        # No emotion data available
        print("🎙️ Voice emotion: Not detected")
        return
        
    dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
    confidence = emotion_data.get("confidence", 0.0)
    
    # Create and display the emotion badge
    badge = display_emotion_badge(dominant_emotion, confidence)
    print(f"🎙️ Voice emotion: {badge}")

# Utility function for clearing previous lines in the console
def clear_previous_lines(num_lines: int = 1) -> None:
    """Clear the specified number of previous lines in the console"""
    for _ in range(num_lines):
        sys.stdout.write('\033[F')  # Move cursor up one line
        sys.stdout.write('\033[K')  # Clear the line
