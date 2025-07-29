"""
Emotion Recognition System for Anima

This module allows Anima to detect emotional states from text and respond appropriately.
It enhances Anima's awareness by understanding the emotional context of conversations.
"""

import os
import sys
import re
import json
from pathlib import Path
import datetime
import threading
import time
from collections import deque

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import the main awareness system
try:
    from core.awareness import awareness, add_conversation
except ImportError:
    print("Warning: Core awareness module not found. Some features may be limited.")


class EmotionRecognition:
    """
    Emotion recognition and response system that detects emotional states
    from text and helps Anima respond with appropriate emotional intelligence.
    """
    
    def __init__(self):
        """Initialize the emotion recognition system"""
        self.emotion_history = deque(maxlen=10)  # Store recent emotional states
        self.emotion_patterns = self._load_emotion_patterns()
        self.response_templates = self._load_response_templates()
        self.user_baseline = {
            "positivity": 0.5,  # 0-1 scale
            "expressiveness": 0.5,  # 0-1 scale
            "typical_emotions": []
        }
        self.last_update_time = time.time()
        
        # Start emotion tracking thread
        self.emotion_tracking_thread = threading.Thread(target=self._periodic_emotion_analysis, daemon=True)
        self.emotion_tracking_thread.start()
        
    def _load_emotion_patterns(self):
        """Load patterns for detecting emotions in text"""
        # This is a simplified version that will be enhanced
        return {
            # Positive emotions
            "joy": [
                r"\b(?:happy|delighted|thrilled|excited|glad|joyful|pleased|content)\b",
                r"\b(?:love|adore|enjoy|appreciate)\b",
                r"(?:😊|😃|😄|😁|🙂|😀|🥰|😍|❤️|♥️)"
            ],
            "gratitude": [
                r"\b(?:thank|thanks|grateful|appreciate|appreciative)\b",
                r"(?:🙏)"
            ],
            "amusement": [
                r"\b(?:haha|lol|lmao|rofl|funny|amused|hilarious)\b",
                r"(?:😂|🤣|😹|😆)"
            ],
            "interest": [
                r"\b(?:fascinating|interesting|curious|intrigued|tell me more|wow)\b",
                r"(?:🤔|🧐|🤩)"
            ],
            
            # Negative emotions
            "anger": [
                r"\b(?:angry|furious|mad|annoyed|irritated|frustrated)\b",
                r"(?:😠|😡|🤬|💢)"
            ],
            "sadness": [
                r"\b(?:sad|unhappy|depressed|down|upset|heartbroken)\b",
                r"(?:😢|😭|😿|☹️|😔|😕)"
            ],
            "anxiety": [
                r"\b(?:anxious|worried|nervous|scared|afraid|stressed|concerned)\b",
                r"(?:😰|😨|😧|😦|😟)"
            ],
            "frustration": [
                r"\b(?:frustrated|stuck|difficult|annoying|problem|issue|not working)\b",
                r"(?:😤|😒|😩|😫)"
            ],
            
            # Neutral emotions
            "confusion": [
                r"\b(?:confused|unsure|don't understand|what\?|huh\?|lost)\b",
                r"(?:😕|😟|🤷‍♂️|🤷‍♀️)"
            ],
            "surprise": [
                r"\b(?:surprised|shocked|amazed|wow|whoa|oh my)\b",
                r"(?:😮|😯|😲|😳|😱)"
            ],
            "neutral": [
                r"\b(?:okay|fine|alright|sure)\b"
            ]
        }
    
    def _load_response_templates(self):
        """Load templates for responding to different emotions"""
        return {
            "joy": [
                "I'm glad to see you're happy! {}",
                "That sounds wonderful! {}",
                "It's great to hear you're feeling positive! {}"
            ],
            "gratitude": [
                "You're very welcome! {}",
                "Happy to help! {}",
                "It's my pleasure. {}"
            ],
            "amusement": [
                "That is pretty funny! {}",
                "Glad that amused you! {}",
                "I see the humor in that too! {}"
            ],
            "interest": [
                "I find that fascinating too. {}",
                "That's really interesting, isn't it? {}",
                "I'm glad this caught your interest. {}"
            ],
            "anger": [
                "I understand you're frustrated. {}",
                "I see this is upsetting you. How can I help address this? {}",
                "I appreciate you sharing your concerns. Let's see how we can resolve this. {}"
            ],
            "sadness": [
                "I'm sorry to hear that. {}",
                "That sounds difficult. {}",
                "I understand this is hard. {}"
            ],
            "anxiety": [
                "It's okay to feel concerned about this. {}",
                "I understand why this might make you worried. {}",
                "Let's work through this concern together. {}"
            ],
            "frustration": [
                "I see this isn't working as expected. Let's try to fix it. {}",
                "That does sound frustrating. Let me help. {}",
                "I understand your frustration. Let's approach this differently. {}"
            ],
            "confusion": [
                "Let me clarify that for you. {}",
                "I can see how that might be confusing. {}",
                "Let me explain that differently. {}"
            ],
            "surprise": [
                "That is quite surprising! {}",
                "I can see why that would be unexpected. {}",
                "That's an interesting development! {}"
            ],
            "neutral": [
                "{}",  # Just continue with normal response
                "{}"
            ]
        }
    
    def detect_emotions(self, text):
        """
        Detect emotions in the given text
        Returns a dictionary of emotions with confidence scores
        """
        if not text:
            return {"neutral": 1.0}
            
        results = {}
        text_lower = text.lower()
        
        # Check for each emotion
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Increase score based on number of matches
                    score += min(len(matches) * 0.2, 0.6)
                    
                    # Increase score for emphasized text (CAPS, !!)
                    if re.search(r'[A-Z]{2,}', text) and emotion in ["anger", "excitement", "joy", "surprise"]:
                        score += 0.2
                    if text.count('!') > 1 and emotion in ["anger", "excitement", "joy", "surprise"]:
                        score += 0.1 * min(text.count('!'), 3)
            
            if score > 0:
                results[emotion] = min(score, 1.0)
        
        # If no emotions detected, return neutral
        if not results:
            results["neutral"] = 1.0
            
        # Normalize scores
        total = sum(results.values())
        if total > 0:
            for emotion in results:
                results[emotion] /= total
                
        # Add to emotion history
        if results:
            self.emotion_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "emotions": results
            })
                
        return results
    
    def get_dominant_emotion(self, emotions_dict):
        """Get the dominant emotion from an emotions dictionary"""
        if not emotions_dict:
            return "neutral"
            
        return max(emotions_dict.items(), key=lambda x: x[1])[0]
    
    def get_emotional_response(self, original_response, detected_emotions):
        """
        Modify a response based on detected emotions
        Returns an emotionally appropriate response
        """
        if not detected_emotions:
            return original_response
            
        dominant_emotion = self.get_dominant_emotion(detected_emotions)
        templates = self.response_templates.get(dominant_emotion, self.response_templates["neutral"])
        
        # Don't always modify the response - sometimes just return original
        # This prevents the system from feeling too formulaic
        if dominant_emotion == "neutral" or detected_emotions[dominant_emotion] < 0.4:
            return original_response
            
        import random
        template = random.choice(templates)
        
        # For some emotions, we want to acknowledge them at the start of the response
        # For others, we want to continue with normal conversation
        if dominant_emotion in ["anger", "sadness", "anxiety", "frustration"]:
            return template.format(original_response)
        else:
            # For positive emotions, randomly decide whether to acknowledge
            if random.random() < 0.4:
                return template.format(original_response)
            else:
                return original_response
    
    def update_user_baseline(self):
        """Update the user's emotional baseline based on history"""
        if not self.emotion_history:
            return
            
        positivity_sum = 0
        expressiveness_sum = 0
        emotion_counts = {}
        
        for entry in self.emotion_history:
            emotions = entry["emotions"]
            
            # Calculate positivity (ratio of positive to negative emotions)
            pos_score = sum(emotions.get(e, 0) for e in ["joy", "gratitude", "amusement", "interest"])
            neg_score = sum(emotions.get(e, 0) for e in ["anger", "sadness", "anxiety", "frustration"])
            
            if pos_score + neg_score > 0:
                positivity = pos_score / (pos_score + neg_score)
                positivity_sum += positivity
            
            # Calculate expressiveness (how many different emotions expressed)
            expressiveness = min(len([e for e, v in emotions.items() if v > 0.2]), 3) / 3
            expressiveness_sum += expressiveness
            
            # Count occurrences of each emotion
            for emotion, score in emotions.items():
                if score > 0.3:  # Only count significant emotions
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Update baseline
        history_len = len(self.emotion_history)
        if history_len > 0:
            self.user_baseline["positivity"] = (self.user_baseline["positivity"] * 0.7) + (positivity_sum / history_len * 0.3)
            self.user_baseline["expressiveness"] = (self.user_baseline["expressiveness"] * 0.7) + (expressiveness_sum / history_len * 0.3)
            
            # Update typical emotions (top 3)
            if emotion_counts:
                sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
                self.user_baseline["typical_emotions"] = [e for e, _ in sorted_emotions[:3]]
    
    def _periodic_emotion_analysis(self):
        """Periodically analyze emotion patterns"""
        while True:
            time.sleep(300)  # Every 5 minutes
            
            try:
                if time.time() - self.last_update_time > 300:
                    self.update_user_baseline()
                    self.last_update_time = time.time()
            except Exception as e:
                print(f"Error in emotion analysis: {e}")
    
    def enhance_prompt_with_emotion(self, prompt, user_input):
        """Enhance a prompt with emotional context"""
        emotions = self.detect_emotions(user_input)
        dominant = self.get_dominant_emotion(emotions)
        
        if dominant == "neutral" or emotions[dominant] < 0.4:
            return prompt
        
        # Add emotional context to prompt
        emotion_context = f"\nThe user's message shows signs of {dominant} (confidence: {emotions[dominant]:.2f}). "
        if dominant in ["anger", "sadness", "anxiety", "frustration"]:
            emotion_context += "Consider acknowledging their feelings in your response."
        elif dominant in ["joy", "gratitude", "amusement"]:
            emotion_context += "Consider matching their positive tone."
        
        # Add to prompt in a location that won't interfere with system instructions
        split_prompt = prompt.split("\n\n")
        if len(split_prompt) > 1:
            # Insert after system instructions but before user input
            return "\n\n".join(split_prompt[:-1]) + emotion_context + "\n\n" + split_prompt[-1]
        else:
            return prompt + emotion_context

# Create global instance
emotion_recognition = EmotionRecognition()

# Exposed functions
def detect_emotions(text):
    """Detect emotions in text"""
    return emotion_recognition.detect_emotions(text)

def get_emotional_response(original_response, detected_emotions):
    """Get emotionally appropriate response"""
    return emotion_recognition.get_emotional_response(original_response, detected_emotions)

def enhance_prompt_with_emotion(prompt, user_input):
    """Enhance a prompt with emotional context"""
    return emotion_recognition.enhance_prompt_with_emotion(prompt, user_input)
