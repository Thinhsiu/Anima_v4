"""
NLP System for Anima_v4 - Provides advanced natural language processing capabilities
"""

from .nlp_core import NLPProcessor
from .entity_recognition import EntityExtractor
from .sentiment_analysis import SentimentAnalyzer
from .text_classification import TextClassifier

__all__ = ['NLPProcessor', 'EntityExtractor', 'SentimentAnalyzer', 'TextClassifier']
