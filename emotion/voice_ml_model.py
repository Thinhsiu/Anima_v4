"""
ML-based voice emotion detection model for Anima voice bridge

This module provides a more advanced voice emotion detection capability
using machine learning techniques to extract and classify emotions from
audio data.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Import optional dependencies - gracefully degrade if not available
try:
    import librosa
    import soundfile as sf
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

# Set up logger
logger = logging.getLogger(__name__)

# Define emotion categories
EMOTION_CATEGORIES = [
    "neutral", "calm", "happy", "sad", "angry",
    "fearful", "disgust", "surprised", "confused", "excited"
]

class SimpleVoiceEmotionClassifier:
    """Fallback classifier using basic audio features for emotion detection"""
    
    def __init__(self):
        """Initialize the simple classifier with baseline feature extraction"""
        self.feature_extractors = {
            "energy": self._extract_energy,
            "zero_crossing_rate": self._extract_zero_crossing_rate,
            "speech_rate": self._estimate_speech_rate
        }
        logger.warning("Using simple voice emotion classifier - for better results install librosa and transformers")
        
    def _extract_energy(self, audio_array: np.ndarray) -> float:
        """Extract energy/volume features from audio array"""
        # Simple RMS energy calculation
        if len(audio_array) == 0:
            return 0.0
        return np.sqrt(np.mean(np.square(audio_array)))
    
    def _extract_zero_crossing_rate(self, audio_array: np.ndarray) -> float:
        """Calculate zero crossing rate as a proxy for pitch/frequency"""
        if len(audio_array) <= 1:
            return 0.0
        # Count sign changes
        signs = np.sign(audio_array[1:] * audio_array[:-1])
        crossings = np.sum(signs < 0)
        return crossings / (len(audio_array) - 1)
    
    def _estimate_speech_rate(self, audio_array: np.ndarray) -> float:
        """Estimate speech rate from envelope variations"""
        if len(audio_array) <= 1:
            return 0.5  # Default mid-range value
            
        # Get amplitude envelope
        envelope = np.abs(audio_array)
        # Use a simple peak detection
        threshold = 0.5 * np.mean(envelope)
        above_threshold = envelope > threshold
        # Count transitions as an estimate of syllables
        transitions = np.sum(np.abs(np.diff(above_threshold.astype(int))))
        # Normalize by audio length to get rate
        return min(1.0, transitions / (len(audio_array) / 1000))
    
    def convert_audio_bytes_to_array(self, audio_bytes: bytes, sr: int = 16000) -> np.ndarray:
        """Convert raw audio bytes to numpy array using basic techniques"""
        try:
            # Assuming 16-bit PCM mono audio
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            # Normalize
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            return audio_array
        except Exception as e:
            logger.error(f"Error converting audio bytes to array: {e}")
            return np.zeros(1000, dtype=np.float32)  # Return empty array on error
    
    def predict_emotion(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Predict emotion from audio bytes using basic features"""
        # Convert to array
        audio_array = self.convert_audio_bytes_to_array(audio_bytes)
        
        # Extract features
        features = {}
        for name, extractor in self.feature_extractors.items():
            features[name] = extractor(audio_array)
        
        # Simple rule-based emotion classification based on extracted features
        energy = features["energy"]
        zcr = features["zero_crossing_rate"]
        speech_rate = features["speech_rate"]
        
        # Simple rule-based classification
        emotions = {}
        
        # Base neutral value
        emotions["neutral"] = 0.3
        
        # Energy-based emotions
        emotions["angry"] = max(0.0, min(0.9, energy * 0.8))
        emotions["excited"] = max(0.0, min(0.9, energy * speech_rate * 1.2))
        emotions["sad"] = max(0.0, min(0.9, (1.0 - energy) * 0.7))
        emotions["calm"] = max(0.0, min(0.9, (1.0 - energy) * (1.0 - speech_rate) * 0.9))
        
        # Zero-crossing rate based emotions
        emotions["fearful"] = max(0.0, min(0.9, zcr * energy * 0.7))
        emotions["surprised"] = max(0.0, min(0.9, zcr * speech_rate * 0.8))
        
        # Speech rate based emotions
        emotions["happy"] = max(0.0, min(0.9, speech_rate * 0.6 + energy * 0.4))
        emotions["confused"] = max(0.0, min(0.7, (speech_rate * 0.3) + (zcr * 0.3)))
        emotions["disgust"] = max(0.0, min(0.7, 0.3))  # Hard to detect with simple features
        
        # Find dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        # Calculate confidence based on how distinct the dominant emotion is
        emotion_values = list(emotions.values())
        if len(emotion_values) > 1:
            sorted_values = sorted(emotion_values, reverse=True)
            confidence = min(0.8, max(0.3, (sorted_values[0] - sorted_values[1]) / sorted_values[0] * 0.7 + 0.3))
        else:
            confidence = 0.3
            
        return {
            "emotion_detected": True,
            "dominant_emotion": dominant_emotion[0],
            "confidence": confidence,
            "emotion_scores": emotions,
            "voice_metrics": {
                "energy": float(energy),
                "zero_crossing_rate": float(zcr),
                "speech_rate": float(speech_rate)
            }
        }

class AdvancedVoiceEmotionClassifier:
    """Advanced ML-based voice emotion classifier using deep learning models"""
    
    def __init__(self, model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """Initialize the advanced voice emotion classifier with pre-trained models
        
        Args:
            model_name: Name of the HuggingFace model to use for emotion classification
        """
        self.model_name = model_name
        self.sample_rate = 16000  # Default sample rate for most models
        self.feature_extractor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for voice emotion ML: {self.device}")
        self.initialize_model()
        
    def initialize_model(self):
        """Load the pre-trained model and feature extractor"""
        try:
            logger.info(f"Loading voice emotion model: {self.model_name}")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.id2label = self.model.config.id2label
            logger.info(f"Voice emotion model loaded successfully with {len(self.id2label)} emotion classes")
        except Exception as e:
            logger.error(f"Error loading voice emotion model: {e}")
            self.model = None
            self.feature_extractor = None
    
    def convert_audio_bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw audio bytes to a numpy array for processing"""
        try:
            import io
            with io.BytesIO(audio_bytes) as buffer:
                # Load audio using librosa (handles various formats)
                audio_array, sr = librosa.load(buffer, sr=self.sample_rate, mono=True)
                return audio_array
        except Exception as e:
            logger.error(f"Error converting audio bytes to array: {e}")
            return np.zeros(1000, dtype=np.float32)  # Return empty array on error
    
    def predict_emotion(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Predict emotion from audio bytes using the pre-trained model"""
        if self.model is None or self.feature_extractor is None:
            logger.warning("Model not initialized, falling back to simple classifier")
            simple_classifier = SimpleVoiceEmotionClassifier()
            return simple_classifier.predict_emotion(audio_bytes)
            
        try:
            # Convert to array with proper sample rate
            audio_array = self.convert_audio_bytes_to_array(audio_bytes)
            
            # Extract features using the model's feature extractor
            inputs = self.feature_extractor(
                audio_array, 
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to the right device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = F.softmax(logits, dim=-1)[0].cpu().numpy()
            
            # Map indices to emotion labels
            emotions = {}
            for idx, prob in enumerate(probabilities):
                emotion = self.id2label[idx]
                emotions[emotion] = float(prob)
            
            # Get the dominant emotion
            dominant_idx = int(np.argmax(probabilities))
            dominant_emotion = self.id2label[dominant_idx]
            
            # Calculate confidence
            confidence = float(probabilities[dominant_idx])
            
            # Extract additional voice metrics
            energy = float(np.sqrt(np.mean(np.square(audio_array))))
            zcr = float(np.sum(np.abs(np.diff(np.sign(audio_array)))) / (2 * len(audio_array)))
            
            # Estimate speech rate using zero-crossing analysis
            speech_rate = min(1.0, zcr * 3)  # Scale ZCR to a reasonable speech rate proxy
            
            return {
                "emotion_detected": True,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "emotion_scores": emotions,
                "voice_metrics": {
                    "energy": energy,
                    "zero_crossing_rate": zcr,
                    "speech_rate": speech_rate,
                }
            }
        except Exception as e:
            logger.error(f"Error in ML emotion prediction: {e}")
            # Fall back to simple classifier on error
            logger.warning("Falling back to simple classifier")
            simple_classifier = SimpleVoiceEmotionClassifier()
            return simple_classifier.predict_emotion(audio_bytes)

def get_classifier() -> Any:
    """Factory function to get the best available classifier
    
    Returns:
        The most advanced available emotion classifier
    """
    if ADVANCED_FEATURES_AVAILABLE:
        try:
            classifier = AdvancedVoiceEmotionClassifier()
            if classifier.model is not None:
                return classifier
        except Exception as e:
            logger.error(f"Failed to initialize advanced classifier: {e}")
    
    # Fall back to simple classifier
    return SimpleVoiceEmotionClassifier()
