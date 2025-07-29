"""
Emotion module for Anima
Provides emotional intelligence capabilities for enhanced understanding and response
"""

# Import core components to make them available through the package
try:
    from .emotion_analyzer import initialize as initialize_emotion_analyzer
    from .emotion_analyzer import get_instance as get_emotion_analyzer
except ImportError:
    pass
