"""
Emotion Memory - Tracks emotional context across conversations
Provides historical emotion tracking and emotional profile building
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import time
import threading
from collections import deque, Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import emotion core
try:
    from emotion.emotion_core import get_instance as get_emotion_detector
    EMOTION_CORE_AVAILABLE = True
except ImportError:
    logger.warning("Emotion core not available for emotion memory")
    EMOTION_CORE_AVAILABLE = False

class EmotionMemory:
    """
    Tracks emotional context across conversations
    Builds emotional profiles and detects significant changes
    """
    
    def __init__(self, memory_size: int = 100):
        """
        Initialize the emotion memory
        
        Args:
            memory_size: Maximum number of emotion records to store
        """
        self.memory_size = memory_size
        self.emotion_records = deque(maxlen=memory_size)
        self.emotion_profiles = {}
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        self.save_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "records_added": 0,
            "profiles_updated": 0,
            "significant_changes": 0
        }
        
        # Load existing records and profiles
        self._load_emotion_data()
        
        logger.info("Emotion memory initialized")
    
    def add_emotion_record(self, 
                           text: str, 
                           emotion_analysis: Dict[str, Any] = None,
                           profile_id: str = "default",
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add emotion record to memory
        
        Args:
            text: The text that was analyzed
            emotion_analysis: Emotion analysis results (or None to analyze now)
            profile_id: Identifier for the emotional profile to update
            context: Additional context about the interaction
            
        Returns:
            Added emotion record
        """
        # Get emotion analysis if not provided
        if emotion_analysis is None and EMOTION_CORE_AVAILABLE:
            emotion_detector = get_emotion_detector()
            if emotion_detector:
                emotion_analysis = emotion_detector.detect_emotions(text)
            else:
                emotion_analysis = {
                    "dominant_emotion": "neutral",
                    "dominant_intensity": "low",
                    "confidence": 0.0
                }
        
        # Create record with timestamp
        timestamp = datetime.now().isoformat()
        record = {
            "timestamp": timestamp,
            "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate long texts
            "emotion": emotion_analysis.get("dominant_emotion", "neutral"),
            "intensity": emotion_analysis.get("dominant_intensity", "low"),
            "confidence": emotion_analysis.get("confidence", 0.0),
            "profile_id": profile_id
        }
        
        # Add context if provided
        if context:
            record["context"] = context
            
        # Add to records
        self.emotion_records.append(record)
        self.stats["records_added"] += 1
        
        # Update profile
        self._update_profile(profile_id, emotion_analysis)
        
        # Save periodically
        self._autosave()
        
        return record
    
    def get_recent_emotions(self, count: int = 5, profile_id: str = "default") -> List[Dict[str, Any]]:
        """
        Get recent emotion records
        
        Args:
            count: Number of records to return
            profile_id: Profile to filter by (or None for all)
            
        Returns:
            List of recent emotion records
        """
        if not profile_id:
            return list(self.emotion_records)[-count:]
        
        # Filter by profile
        filtered = [r for r in self.emotion_records if r.get("profile_id") == profile_id]
        return filtered[-count:]
    
    def get_emotion_profile(self, profile_id: str = "default") -> Dict[str, Any]:
        """
        Get emotional profile
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Emotional profile data
        """
        # Return existing profile or create empty one
        if profile_id in self.emotion_profiles:
            return self.emotion_profiles[profile_id]
        
        return {
            "profile_id": profile_id,
            "dominant_emotion": "neutral",
            "emotional_stability": 1.0,
            "emotional_range": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def detect_emotional_shift(self, 
                               text: str, 
                               profile_id: str = "default",
                               threshold: float = 0.3) -> Dict[str, Any]:
        """
        Detect if text represents a significant emotional shift
        
        Args:
            text: Text to analyze
            profile_id: Profile to compare against
            threshold: Threshold for significant shift (0-1)
            
        Returns:
            Shift analysis
        """
        if not EMOTION_CORE_AVAILABLE:
            return {"shift_detected": False, "magnitude": 0.0}
            
        # Get emotion detector
        emotion_detector = get_emotion_detector()
        if not emotion_detector:
            return {"shift_detected": False, "magnitude": 0.0}
            
        # Get current emotion analysis
        current = emotion_detector.detect_emotions(text)
        current_emotion = current.get("dominant_emotion", "neutral")
        current_score = current.get("emotions", {}).get(current_emotion, 0.0)
        
        # Get profile
        profile = self.get_emotion_profile(profile_id)
        profile_emotion = profile.get("dominant_emotion", "neutral")
        
        # Get recent emotions for this profile
        recent = self.get_recent_emotions(3, profile_id)
        
        # If no recent emotions, can't detect shift
        if not recent:
            return {"shift_detected": False, "magnitude": 0.0, "reason": "no_history"}
            
        # Get recent emotion scores
        recent_emotions = [r.get("emotion") for r in recent]
        most_common = Counter(recent_emotions).most_common(1)[0][0]
        
        # Different from most common recent emotion?
        different_from_recent = current_emotion != most_common
        
        # Different from profile dominant emotion?
        different_from_profile = current_emotion != profile_emotion
        
        # Calculate shift magnitude
        magnitude = 0.0
        
        # Higher magnitude if different from both profile and recent
        if different_from_profile and different_from_recent:
            magnitude = 0.7
        # Medium magnitude if different from profile only
        elif different_from_profile:
            magnitude = 0.5
        # Lower magnitude if different from recent only
        elif different_from_recent:
            magnitude = 0.3
            
        # Adjust by confidence
        magnitude *= current.get("confidence", 0.5)
        
        # Detect shift
        shift_detected = magnitude >= threshold
        
        # If shift detected, record it
        if shift_detected:
            self.stats["significant_changes"] += 1
        
        return {
            "shift_detected": shift_detected,
            "magnitude": magnitude,
            "current_emotion": current_emotion,
            "profile_emotion": profile_emotion,
            "recent_emotion": most_common,
            "confidence": current.get("confidence", 0.0)
        }
    
    def summarize_emotional_history(self, profile_id: str = "default") -> Dict[str, Any]:
        """
        Summarize emotional history for a profile
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Emotional history summary
        """
        # Get records for this profile
        records = [r for r in self.emotion_records if r.get("profile_id") == profile_id]
        
        if not records:
            return {
                "profile_id": profile_id,
                "records": 0,
                "summary": "No emotional history recorded"
            }
            
        # Count emotions
        emotions = Counter([r.get("emotion") for r in records])
        intensities = Counter([r.get("intensity") for r in records])
        
        # Calculate stability (consistency of emotions)
        dominant_count = emotions.most_common(1)[0][1]
        stability = dominant_count / len(records) if records else 1.0
        
        # Find emotional range
        emotion_range = list(emotions.keys())
        
        # Calculate average confidence
        avg_confidence = sum(r.get("confidence", 0.0) for r in records) / len(records)
        
        # Determine emotional trend (comparing first and last third)
        third = max(1, len(records) // 3)
        first_third = records[:third]
        last_third = records[-third:]
        
        first_emotions = Counter([r.get("emotion") for r in first_third])
        last_emotions = Counter([r.get("emotion") for r in last_third])
        
        first_dominant = first_emotions.most_common(1)[0][0] if first_emotions else "neutral"
        last_dominant = last_emotions.most_common(1)[0][0] if last_emotions else "neutral"
        
        trend = "stable"
        if first_dominant != last_dominant:
            trend = f"shift from {first_dominant} to {last_dominant}"
        
        return {
            "profile_id": profile_id,
            "records": len(records),
            "dominant_emotion": emotions.most_common(1)[0][0],
            "emotion_distribution": dict(emotions),
            "intensity_distribution": dict(intensities),
            "emotional_stability": stability,
            "emotional_range": emotion_range,
            "average_confidence": avg_confidence,
            "trend": trend,
            "first_timestamp": records[0].get("timestamp") if records else None,
            "last_timestamp": records[-1].get("timestamp") if records else None
        }
    
    def save(self) -> bool:
        """
        Save emotion records and profiles to disk
        
        Returns:
            True if successful, False otherwise
        """
        with self.save_lock:
            try:
                # Ensure directory exists
                data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
                os.makedirs(data_dir, exist_ok=True)
                
                # Save records
                records_path = os.path.join(data_dir, "emotion_records.json")
                with open(records_path, 'w', encoding='utf-8') as f:
                    json.dump(list(self.emotion_records), f, indent=2)
                    
                # Save profiles
                profiles_path = os.path.join(data_dir, "emotion_profiles.json")
                with open(profiles_path, 'w', encoding='utf-8') as f:
                    json.dump(self.emotion_profiles, f, indent=2)
                
                self.last_save_time = time.time()
                return True
            except Exception as e:
                logger.error(f"Error saving emotion memory: {e}")
                return False
    
    def _load_emotion_data(self) -> None:
        """Load emotion records and profiles from disk"""
        try:
            # Check for data directory
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            
            # Load records
            records_path = os.path.join(data_dir, "emotion_records.json")
            if os.path.exists(records_path):
                with open(records_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    self.emotion_records = deque(records, maxlen=self.memory_size)
                    logger.info(f"Loaded {len(records)} emotion records")
                    
            # Load profiles
            profiles_path = os.path.join(data_dir, "emotion_profiles.json")
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r', encoding='utf-8') as f:
                    self.emotion_profiles = json.load(f)
                    logger.info(f"Loaded {len(self.emotion_profiles)} emotion profiles")
        except Exception as e:
            logger.error(f"Error loading emotion memory: {e}")
    
    def _update_profile(self, profile_id: str, emotion_analysis: Dict[str, Any]) -> None:
        """
        Update emotional profile with new emotion data
        
        Args:
            profile_id: Profile identifier
            emotion_analysis: Emotion analysis results
        """
        # Get current profile or create new one
        if profile_id in self.emotion_profiles:
            profile = self.emotion_profiles[profile_id]
        else:
            profile = {
                "profile_id": profile_id,
                "emotions": {},
                "dominant_emotion": "neutral",
                "emotional_stability": 1.0,
                "emotional_range": [],
                "created": datetime.now().isoformat(),
                "record_count": 0
            }
            
        # Update profile
        profile["last_updated"] = datetime.now().isoformat()
        profile["record_count"] = profile.get("record_count", 0) + 1
        
        # Get emotion data
        emotion = emotion_analysis.get("dominant_emotion", "neutral")
        
        # Update emotion counts
        if "emotions" not in profile:
            profile["emotions"] = {}
            
        profile["emotions"][emotion] = profile["emotions"].get(emotion, 0) + 1
        
        # Recalculate dominant emotion
        if profile["emotions"]:
            dominant = max(profile["emotions"].items(), key=lambda x: x[1])
            profile["dominant_emotion"] = dominant[0]
            
            # Calculate stability
            total = sum(profile["emotions"].values())
            profile["emotional_stability"] = dominant[1] / total if total > 0 else 1.0
            
            # Update emotional range
            profile["emotional_range"] = list(profile["emotions"].keys())
        
        # Save updated profile
        self.emotion_profiles[profile_id] = profile
        self.stats["profiles_updated"] += 1
    
    def _autosave(self) -> None:
        """Autosave if it's been long enough since the last save"""
        now = time.time()
        if now - self.last_save_time > self.save_interval:
            threading.Thread(target=self.save).start()


# Create singleton instance
_emotion_memory = None

def initialize() -> EmotionMemory:
    """Initialize the emotion memory singleton"""
    global _emotion_memory
    if _emotion_memory is None:
        logger.info("Initializing emotion memory...")
        try:
            _emotion_memory = EmotionMemory()
            logger.info("Emotion memory initialized")
        except Exception as e:
            logger.error(f"Error initializing emotion memory: {e}")
            return None
    return _emotion_memory

def get_instance() -> Optional[EmotionMemory]:
    """Get the emotion memory singleton instance"""
    global _emotion_memory
    # Auto-initialize if not already done
    if _emotion_memory is None:
        return initialize()
    return _emotion_memory

# Helper functions for easy access
def add_emotion_record(text: str, profile_id: str = "default") -> Dict[str, Any]:
    """Add emotion record to memory"""
    memory = get_instance()
    if memory and EMOTION_CORE_AVAILABLE:
        emotion_detector = get_emotion_detector()
        if emotion_detector:
            emotions = emotion_detector.detect_emotions(text)
            return memory.add_emotion_record(text, emotions, profile_id)
    return {}

def get_emotion_profile(profile_id: str = "default") -> Dict[str, Any]:
    """Get emotional profile"""
    memory = get_instance()
    if memory:
        return memory.get_emotion_profile(profile_id)
    return {}

# Auto-initialize the module
try:
    initialize()
except Exception as e:
    logger.error(f"Error during auto-initialization of emotion memory: {e}")


if __name__ == "__main__":
    # Simple test
    memory = get_instance()
    
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
    
    print("=== Emotion Memory Test ===")
    for text in test_texts:
        record = add_emotion_record(text)
        print(f"Added: {record.get('emotion')} ({record.get('intensity')}) - {text[:30]}...")
        
    profile = get_emotion_profile()
    print("\nEmotion Profile:")
    print(f"Dominant emotion: {profile.get('dominant_emotion')}")
    print(f"Stability: {profile.get('emotional_stability'):.2f}")
    print(f"Range: {', '.join(profile.get('emotional_range', []))}")
    
    summary = memory.summarize_emotional_history()
    print("\nEmotional History Summary:")
    print(f"Records: {summary.get('records')}")
    print(f"Dominant emotion: {summary.get('dominant_emotion')}")
    print(f"Trend: {summary.get('trend')}")
