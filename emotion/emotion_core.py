"""
Emotion Core - Foundation for emotion analysis in Anima
Provides core emotion detection capabilities
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import re
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import NLP system for text analysis
try:
    from nlp.integration import get_instance as get_nlp
    NLP_AVAILABLE = True
except ImportError:
    logger.warning("NLP system not available for emotion analysis")
    NLP_AVAILABLE = False

# Define core emotion types
EMOTION_TYPES = [
    "joy", "sadness", "anger", "fear", 
    "surprise", "disgust", "trust", "anticipation",
    "neutral"
]

# Define emotion intensities
EMOTION_INTENSITIES = ["low", "medium", "high"]

class EmotionDetector:
    """
    Core emotion detection capabilities for text
    Combines rule-based and ML approaches
    """
    
    def __init__(self):
        """Initialize the emotion detector"""
        # Load emotion lexicons and patterns
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.emotion_patterns = self._load_emotion_patterns()
        
        # Get NLP system for advanced analysis
        self.nlp = get_nlp() if NLP_AVAILABLE else None
        
        # Initialize statistics
        self.stats = {
            "texts_analyzed": 0,
            "emotions_detected": 0
        }
        
        logger.info("Emotion detector initialized")
    
    def detect_emotions(self, text: str) -> Dict[str, Any]:
        """
        Detect emotions in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with emotion analysis results
        """
        if not text.strip():
            return self._get_default_emotions()
        
        # Update statistics
        self.stats["texts_analyzed"] += 1
        
        # Combine multiple detection methods
        emotions = {}
        
        # Method 1: Lexicon-based detection
        lexicon_emotions = self._detect_with_lexicon(text)
        
        # Method 2: Pattern-based detection
        pattern_emotions = self._detect_with_patterns(text)
        
        # Method 3: Use NLP sentiment if available
        nlp_emotions = self._detect_with_nlp(text)
        
        # Combine all methods with weights
        emotions = self._combine_emotion_detections([
            (lexicon_emotions, 0.4),
            (pattern_emotions, 0.3),
            (nlp_emotions, 0.3)
        ])
        
        # Determine dominant emotion
        dominant = self._get_dominant_emotion(emotions)
        
        # Format the result
        result = {
            "emotions": emotions,
            "dominant_emotion": dominant["emotion"],
            "dominant_intensity": dominant["intensity"],
            "emotional_state": f"{dominant['intensity']} {dominant['emotion']}",
            "confidence": dominant["confidence"],
            "analysis_methods": ["lexicon", "pattern"]
        }
        
        if self.nlp:
            result["analysis_methods"].append("nlp_sentiment")
        
        # Update statistics
        self.stats["emotions_detected"] += len(emotions)
        
        return result
    
    def detect_emotion_changes(self, 
                               current_text: str, 
                               previous_texts: List[str]) -> Dict[str, Any]:
        """
        Detect changes in emotion over time
        
        Args:
            current_text: Current text input
            previous_texts: List of previous text inputs (most recent first)
            
        Returns:
            Dictionary with emotion change analysis
        """
        # Detect current emotions
        current = self.detect_emotions(current_text)
        
        # If no previous texts, just return current with no change
        if not previous_texts:
            current["change"] = "initial"
            current["change_magnitude"] = 0.0
            return current
        
        # Detect emotions in previous text
        previous = self.detect_emotions(previous_texts[0])
        
        # Compare dominant emotions
        curr_emotion = current["dominant_emotion"]
        prev_emotion = previous["dominant_emotion"]
        
        # Calculate emotional shift
        emotion_shift = self._calculate_emotion_shift(
            prev_emotion, curr_emotion,
            previous["emotions"], current["emotions"]
        )
        
        # Determine change description
        change_desc = "stable"
        if emotion_shift > 0.3:
            change_desc = "significant_shift"
        elif emotion_shift > 0.1:
            change_desc = "moderate_shift"
        elif emotion_shift > 0.0:
            change_desc = "slight_shift"
            
        # Create result with change information
        result = dict(current)
        result["previous_emotion"] = prev_emotion
        result["change"] = change_desc
        result["change_magnitude"] = emotion_shift
        
        return result
    
    def get_emotion_summary(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get emotional summary of multiple texts
        
        Args:
            texts: List of text inputs
            
        Returns:
            Emotional summary
        """
        if not texts:
            return {
                "dominant_emotion": "neutral",
                "emotional_range": [],
                "emotional_stability": 1.0
            }
            
        # Analyze each text
        analyses = [self.detect_emotions(text) for text in texts]
        
        # Count emotion occurrences
        emotion_counts = {}
        for analysis in analyses:
            emotion = analysis["dominant_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Determine dominant emotion across texts
        dominant = max(emotion_counts.items(), key=lambda x: x[1])
        dominant_emotion = dominant[0]
        
        # Calculate emotional range (unique emotions)
        emotional_range = list(emotion_counts.keys())
        
        # Calculate emotional stability (0-1)
        # 1 = perfectly stable (same emotion), 0 = highly unstable (all different)
        stability = dominant[1] / len(analyses) if analyses else 1.0
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotional_range": emotional_range,
            "emotional_stability": stability,
            "emotion_distribution": emotion_counts
        }
    
    def _detect_with_lexicon(self, text: str) -> Dict[str, float]:
        """
        Detect emotions using lexicon-based approach
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping emotions to scores
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Initialize emotion scores
        emotions = {emotion: 0.0 for emotion in EMOTION_TYPES}
        
        # Count emotion words
        for word in words:
            if word in self.emotion_lexicon:
                for emotion, score in self.emotion_lexicon[word].items():
                    emotions[emotion] += score
        
        # Normalize by text length (avoid bias for longer texts)
        if words:
            for emotion in emotions:
                emotions[emotion] /= max(1, len(words) / 5)  # Normalize per 5 words
                emotions[emotion] = min(1.0, emotions[emotion])  # Cap at 1.0
        
        return emotions
    
    def _detect_with_patterns(self, text: str) -> Dict[str, float]:
        """
        Detect emotions using pattern-based approach
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping emotions to scores
        """
        text = text.lower()
        
        # Initialize emotion scores
        emotions = {emotion: 0.0 for emotion in EMOTION_TYPES}
        
        # Check each emotion pattern
        for emotion, patterns in self.emotion_patterns.items():
            for pattern, score in patterns:
                if re.search(pattern, text):
                    emotions[emotion] += score
        
        # Normalize scores
        for emotion in emotions:
            emotions[emotion] = min(1.0, emotions[emotion])  # Cap at 1.0
        
        return emotions
    
    def _detect_with_nlp(self, text: str) -> Dict[str, float]:
        """
        Detect emotions using NLP sentiment
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping emotions to scores
        """
        # Default emotions
        emotions = {emotion: 0.0 for emotion in EMOTION_TYPES}
        
        # If NLP not available, return defaults
        if not self.nlp:
            return emotions
            
        try:
            # Analyze text with NLP
            analysis = self.nlp.analyze_text(text, analysis_types=["sentiment"])
            
            if "sentiment" in analysis:
                sentiment = analysis["sentiment"]
                
                # Map sentiment to emotions
                if sentiment.get("overall_sentiment") == "positive":
                    emotions["joy"] += 0.6
                    emotions["trust"] += 0.4
                elif sentiment.get("overall_sentiment") == "negative":
                    emotions["sadness"] += 0.3
                    emotions["anger"] += 0.3
                    emotions["disgust"] += 0.2
                else:
                    emotions["neutral"] += 0.8
                
                # Map intensity if available
                intensity = sentiment.get("intensity", 0.5)
                for emotion in emotions:
                    emotions[emotion] *= intensity
        except Exception as e:
            logger.error(f"Error in NLP emotion detection: {e}")
        
        return emotions
    
    def _combine_emotion_detections(self, 
                                  detections_with_weights: List[Tuple[Dict[str, float], float]]
                                  ) -> Dict[str, float]:
        """
        Combine multiple emotion detections with weights
        
        Args:
            detections_with_weights: List of (detection, weight) tuples
            
        Returns:
            Combined emotion scores
        """
        combined = {emotion: 0.0 for emotion in EMOTION_TYPES}
        total_weight = 0.0
        
        for detection, weight in detections_with_weights:
            if not detection:  # Skip empty detections
                continue
                
            total_weight += weight
            for emotion, score in detection.items():
                combined[emotion] += score * weight
        
        # Normalize by total weight
        if total_weight > 0:
            for emotion in combined:
                combined[emotion] /= total_weight
        
        # Ensure neutral has some minimal value
        if combined["neutral"] < 0.1:
            combined["neutral"] = 0.1
            
        return combined
    
    def _get_dominant_emotion(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """
        Get dominant emotion from emotion scores
        
        Args:
            emotions: Dictionary mapping emotions to scores
            
        Returns:
            Dictionary with dominant emotion info
        """
        if not emotions:
            return {
                "emotion": "neutral",
                "intensity": "low",
                "score": 0.0,
                "confidence": 0.0
            }
            
        # Find max emotion
        max_emotion = max(emotions.items(), key=lambda x: x[1])
        emotion = max_emotion[0]
        score = max_emotion[1]
        
        # Determine intensity based on score
        intensity = "low"
        if score >= 0.6:
            intensity = "high"
        elif score >= 0.3:
            intensity = "medium"
            
        # Calculate confidence - how much it stands out from the average
        other_scores = [s for e, s in emotions.items() if e != emotion]
        avg_other = sum(other_scores) / len(other_scores) if other_scores else 0
        confidence = score - avg_other
        
        # If score is too low or confidence too low, default to neutral
        if score < 0.2 or confidence < 0.05:
            emotion = "neutral"
            intensity = "low" if score < 0.1 else "medium"
            
        return {
            "emotion": emotion,
            "intensity": intensity,
            "score": score,
            "confidence": confidence
        }
    
    def _calculate_emotion_shift(self, 
                               prev_emotion: str, 
                               curr_emotion: str,
                               prev_scores: Dict[str, float],
                               curr_scores: Dict[str, float]) -> float:
        """
        Calculate emotional shift between two analyses
        
        Args:
            prev_emotion: Previous dominant emotion
            curr_emotion: Current dominant emotion
            prev_scores: Previous emotion scores
            curr_scores: Current emotion scores
            
        Returns:
            Shift magnitude (0-1)
        """
        # Different emotions indicate a shift
        if prev_emotion != curr_emotion:
            # Check how strong the previous emotion was
            prev_strength = prev_scores.get(prev_emotion, 0)
            curr_strength = curr_scores.get(curr_emotion, 0)
            
            # Strong-to-strong shift is more significant
            if prev_strength > 0.5 and curr_strength > 0.5:
                return 0.8
            # Weak-to-strong or strong-to-weak is moderate
            elif prev_strength > 0.5 or curr_strength > 0.5:
                return 0.5
            # Weak-to-weak is minor
            else:
                return 0.2
        
        # Same emotion, check intensity change
        else:
            prev_strength = prev_scores.get(prev_emotion, 0)
            curr_strength = curr_scores.get(curr_emotion, 0)
            
            # Return the absolute difference in strength
            return min(1.0, abs(curr_strength - prev_strength) * 2)
    
    def _get_default_emotions(self) -> Dict[str, Any]:
        """Get default emotion analysis result for empty text"""
        return {
            "emotions": {emotion: 0.0 for emotion in EMOTION_TYPES},
            "dominant_emotion": "neutral",
            "dominant_intensity": "low",
            "emotional_state": "low neutral",
            "confidence": 0.0,
            "analysis_methods": []
        }
    
    def _load_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """
        Load emotion lexicon from file or create default
        
        Returns:
            Dictionary mapping words to emotion scores
        """
        lexicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    "resources", "emotion_lexicon.json")
        
        # If lexicon exists, load it
        if os.path.exists(lexicon_path):
            try:
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading emotion lexicon: {e}")
        
        # Create basic default lexicon
        logger.info("Creating default emotion lexicon")
        
        # Very basic default lexicon - in production, you'd want a more comprehensive one
        default_lexicon = {
            # Joy words
            "happy": {"joy": 0.8, "neutral": 0.1},
            "glad": {"joy": 0.7, "neutral": 0.2},
            "delighted": {"joy": 0.9, "neutral": 0.1},
            "pleased": {"joy": 0.6, "neutral": 0.2},
            "excited": {"joy": 0.8, "anticipation": 0.4, "neutral": 0.1},
            "thrilled": {"joy": 0.9, "surprise": 0.3, "neutral": 0.1},
            
            # Sadness words
            "sad": {"sadness": 0.8, "neutral": 0.1},
            "unhappy": {"sadness": 0.7, "neutral": 0.2},
            "disappointed": {"sadness": 0.6, "neutral": 0.2},
            "depressed": {"sadness": 0.9, "neutral": 0.1},
            "miserable": {"sadness": 0.9, "disgust": 0.2, "neutral": 0.1},
            "heartbroken": {"sadness": 1.0, "neutral": 0.1},
            
            # Anger words
            "angry": {"anger": 0.8, "neutral": 0.1},
            "furious": {"anger": 0.9, "neutral": 0.1},
            "annoyed": {"anger": 0.5, "neutral": 0.3},
            "irritated": {"anger": 0.6, "neutral": 0.2},
            "outraged": {"anger": 0.9, "disgust": 0.3, "neutral": 0.1},
            "hate": {"anger": 0.8, "disgust": 0.5, "neutral": 0.1},
            
            # Fear words
            "afraid": {"fear": 0.8, "neutral": 0.1},
            "scared": {"fear": 0.8, "neutral": 0.1},
            "terrified": {"fear": 0.9, "neutral": 0.1},
            "anxious": {"fear": 0.6, "neutral": 0.2},
            "worried": {"fear": 0.5, "neutral": 0.3},
            "panic": {"fear": 0.9, "neutral": 0.1},
            
            # Surprise words
            "surprised": {"surprise": 0.7, "neutral": 0.2},
            "shocked": {"surprise": 0.8, "fear": 0.3, "neutral": 0.1},
            "amazed": {"surprise": 0.8, "joy": 0.3, "neutral": 0.1},
            "astonished": {"surprise": 0.9, "neutral": 0.1},
            "unexpected": {"surprise": 0.6, "neutral": 0.3},
            "startled": {"surprise": 0.7, "fear": 0.2, "neutral": 0.2},
            
            # Disgust words
            "disgusted": {"disgust": 0.8, "neutral": 0.1},
            "revolted": {"disgust": 0.9, "neutral": 0.1},
            "gross": {"disgust": 0.7, "neutral": 0.2},
            "repulsed": {"disgust": 0.8, "neutral": 0.1},
            "sick": {"disgust": 0.5, "neutral": 0.3},
            "nauseated": {"disgust": 0.7, "neutral": 0.2},
            
            # Trust words
            "trust": {"trust": 0.7, "neutral": 0.2},
            "believe": {"trust": 0.6, "neutral": 0.3},
            "confident": {"trust": 0.7, "joy": 0.3, "neutral": 0.2},
            "faithful": {"trust": 0.8, "neutral": 0.1},
            "assured": {"trust": 0.6, "neutral": 0.3},
            "reliable": {"trust": 0.7, "neutral": 0.2},
            
            # Anticipation words
            "expect": {"anticipation": 0.6, "neutral": 0.3},
            "await": {"anticipation": 0.7, "neutral": 0.2},
            "hope": {"anticipation": 0.6, "joy": 0.3, "neutral": 0.2},
            "looking forward": {"anticipation": 0.8, "joy": 0.4, "neutral": 0.1},
            "eager": {"anticipation": 0.8, "joy": 0.3, "neutral": 0.1}
        }
        
        # Ensure the resources directory exists
        os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
        
        # Save the default lexicon for future use
        try:
            with open(lexicon_path, 'w', encoding='utf-8') as f:
                json.dump(default_lexicon, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default emotion lexicon: {e}")
        
        return default_lexicon
    
    def _load_emotion_patterns(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Load emotion patterns from file or create default
        
        Returns:
            Dictionary mapping emotions to patterns and scores
        """
        patterns_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "resources", "emotion_patterns.json")
        
        # If patterns exist, load them
        if os.path.exists(patterns_path):
            try:
                with open(patterns_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert lists to tuples for patterns
                    return {k: [(p, s) for p, s in v] for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading emotion patterns: {e}")
        
        # Create basic default patterns
        logger.info("Creating default emotion patterns")
        
        # Very basic default patterns - in production, you'd want more sophisticated ones
        default_patterns = {
            "joy": [
                (r"(?:^|\\s)(?:love|adore|enjoy)\\b", 0.7),
                (r"(?:^|\\s)(?:haha|heh|lol)(?:\\s|$)", 0.5),
                (r"\\b(?:wonderful|fantastic|excellent)\\b", 0.6),
                (r"\\b(?:best|great|awesome)\\b", 0.5),
                (r"[!]{2,}", 0.3)
            ],
            "sadness": [
                (r"\\b(?:sad|upset|down|blue)\\b", 0.6),
                (r"\\b(?:miss|missing|missed)\\b", 0.4),
                (r"\\b(?:alone|lonely|abandoned)\\b", 0.7),
                (r"\\b(?:never|nothing|empty)\\b", 0.3),
                (r"\\b(?:sigh|alas)\\b", 0.5)
            ],
            "anger": [
                (r"\\b(?:angry|mad|furious)\\b", 0.7),
                (r"\\b(?:hate|despise|resent)\\b", 0.8),
                (r"\\b(?:stupid|idiot|fool)\\b", 0.6),
                (r"[!]{3,}", 0.4),
                (r"(?:^|\\s)(?:ugh|argh|ugh)(?:\\s|$)", 0.4)
            ],
            "fear": [
                (r"\\b(?:afraid|scared|frightened)\\b", 0.7),
                (r"\\b(?:worry|worried|anxious)\\b", 0.5),
                (r"\\b(?:terrified|horrified|petrified)\\b", 0.8),
                (r"\\b(?:help|dangerous|threat)\\b", 0.4),
                (r"\\b(?:what if|oh no)\\b", 0.3)
            ],
            "surprise": [
                (r"\\b(?:surprised|shocked|amazed)\\b", 0.7),
                (r"\\b(?:wow|whoa|oh)\\b", 0.5),
                (r"\\b(?:unexpected|sudden|strange)\\b", 0.4),
                (r"[!?]{2,}", 0.3),
                (r"\\b(?:really|seriously|no way)\\b", 0.4)
            ],
            "disgust": [
                (r"\\b(?:disgusting|gross|nasty)\\b", 0.7),
                (r"\\b(?:eww|ugh|yuck)\\b", 0.6),
                (r"\\b(?:sick|vomit|nauseous)\\b", 0.5),
                (r"\\b(?:awful|horrible|terrible)\\b", 0.4),
                (r"\\b(?:hate|can't stand)\\b", 0.5)
            ],
            "trust": [
                (r"\\b(?:trust|believe|faith)\\b", 0.7),
                (r"\\b(?:honest|true|reliable)\\b", 0.6),
                (r"\\b(?:sure|certain|confident)\\b", 0.5),
                (r"\\b(?:depend|count on)\\b", 0.6),
                (r"\\b(?:promise|swear|guarantee)\\b", 0.5)
            ],
            "anticipation": [
                (r"\\b(?:expect|anticipate|await)\\b", 0.6),
                (r"\\b(?:hope|wish|look forward)\\b", 0.5),
                (r"\\b(?:soon|about to|going to)\\b", 0.4),
                (r"\\b(?:excited for|can't wait)\\b", 0.7),
                (r"\\b(?:tomorrow|next|coming)\\b", 0.3)
            ],
            "neutral": [
                (r"\\b(?:is|are|was|were)\\b", 0.2),
                (r"\\b(?:the|a|an)\\b", 0.1),
                (r"\\b(?:it|this|that)\\b", 0.1)
            ]
        }
        
        # Ensure the resources directory exists
        os.makedirs(os.path.dirname(patterns_path), exist_ok=True)
        
        # Save the default patterns for future use
        try:
            # Convert tuples to lists for JSON serialization
            json_patterns = {k: [[p, s] for p, s in v] for k, v in default_patterns.items()}
            with open(patterns_path, 'w', encoding='utf-8') as f:
                json.dump(json_patterns, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default emotion patterns: {e}")
        
        return default_patterns


# Create singleton instance
_emotion_detector = None

def initialize() -> EmotionDetector:
    """Initialize the emotion detector singleton"""
    global _emotion_detector
    if _emotion_detector is None:
        logger.info("Initializing emotion detector...")
        try:
            _emotion_detector = EmotionDetector()
            logger.info("Emotion detector initialized")
        except Exception as e:
            logger.error(f"Error initializing emotion detector: {e}")
            return None
    return _emotion_detector

def get_instance() -> Optional[EmotionDetector]:
    """Get the emotion detector singleton instance"""
    global _emotion_detector
    # Auto-initialize if not already done
    if _emotion_detector is None:
        return initialize()
    return _emotion_detector

# Helper functions for easy access
def detect_emotions(text: str) -> Dict[str, Any]:
    """Detect emotions in text"""
    detector = get_instance()
    if detector:
        return detector.detect_emotions(text)
    return {}

def detect_emotion_changes(current_text: str, previous_texts: List[str]) -> Dict[str, Any]:
    """Detect changes in emotion over time"""
    detector = get_instance()
    if detector:
        return detector.detect_emotion_changes(current_text, previous_texts)
    return {}

# Auto-initialize the module
try:
    initialize()
except Exception as e:
    logger.error(f"Error during auto-initialization of emotion detector: {e}")


if __name__ == "__main__":
    # Simple test
    detector = get_instance()
    
    test_texts = [
        "I'm really happy about this new feature!",
        "This makes me so angry, it's completely broken.",
        "I'm a bit worried about how this will work.",
        "Wow! That's amazing, I didn't expect that!",
        "This is disgusting, I can't believe they did that.",
        "I trust your judgment on this matter.",
        "I'm looking forward to seeing what happens next.",
        "The weather is cloudy today with a chance of rain."
    ]
    
    print("=== Emotion Detection Test ===")
    for text in test_texts:
        emotions = detector.detect_emotions(text)
        print(f"\nText: {text}")
        print(f"Dominant emotion: {emotions['emotional_state']} (confidence: {emotions['confidence']:.2f})")
        print("Emotion scores:")
        for emotion, score in sorted(emotions['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]:
            if score > 0.1:
                print(f"  - {emotion}: {score:.2f}")
