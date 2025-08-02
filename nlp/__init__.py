"""
NLP System for Anima_v4 - Provides advanced natural language processing capabilities
"""

# Import core NLP components with graceful degradation
try:
    from .nlp_core import NLPProcessor
    NLP_CORE_AVAILABLE = True
except ImportError as e:
    print(f"NLP core not available: {e}")
    NLP_CORE_AVAILABLE = False
    NLPProcessor = None

try:
    from .entity_recognition import EntityExtractor
    ENTITY_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"Entity recognition not available: {e}")
    ENTITY_RECOGNITION_AVAILABLE = False
    EntityExtractor = None

try:
    from .sentiment_analysis import SentimentAnalyzer
    SENTIMENT_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Sentiment analysis not available: {e}")
    SENTIMENT_ANALYSIS_AVAILABLE = False
    SentimentAnalyzer = None

try:
    from .text_classification import TextClassifier
    TEXT_CLASSIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Text classification not available (sklearn dependency): {e}")
    TEXT_CLASSIFICATION_AVAILABLE = False
    TextClassifier = None

__all__ = ['NLPProcessor', 'EntityExtractor', 'SentimentAnalyzer', 'TextClassifier']
