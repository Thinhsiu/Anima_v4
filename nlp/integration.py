"""
Integration module to connect the NLP system with the rest of the Anima_v4 framework
"""

import logging
from typing import Dict, Any, List, Optional
import os
import sys
from pathlib import Path

# Setup path to allow imports from parent directory
sys.path.append(str(Path(__file__).parent.parent))

# Import NLP components
from .nlp_core import NLPProcessor
from .entity_recognition import EntityExtractor
from .sentiment_analysis import SentimentAnalyzer
from .text_classification import TextClassifier
from .utils import extract_keywords, summarize_text, compute_readability_scores

# Import Anima components
from core import memory_integration
from llm import openai_llm, local_llm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimalNLP:
    """Main NLP integration class for Anima_v4"""
    
    def __init__(self, model_size: str = 'medium', use_gpu: bool = False):
        """
        Initialize the NLP system
        
        Args:
            model_size: Size of spaCy model to use ('small', 'medium', 'large', or 'trf')
            use_gpu: Whether to use GPU acceleration (requires GPU support)
        """
        # Initialize processors
        self.disable_components = []
        
        if use_gpu:
            try:
                # Try to set up GPU acceleration
                import spacy
                spacy.require_gpu()
                logger.info("GPU acceleration enabled for NLP")
            except Exception as e:
                logger.warning(f"Failed to enable GPU acceleration: {e}")
                use_gpu = False
        
        # Initialize core NLP processor
        self.processor = NLPProcessor(model_size=model_size, disable=self.disable_components)
        
        # Initialize specialized modules
        self.entity_extractor = EntityExtractor(self.processor)
        self.sentiment_analyzer = SentimentAnalyzer(self.processor)
        self.text_classifier = TextClassifier(self.processor)
        
        logger.info(f"Initialized AnimalNLP with {model_size} model")
    
    def analyze_text(self, text: str, analysis_types: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of text
        
        Args:
            text: Text to analyze
            analysis_types: Types of analysis to perform (if None, performs all)
            
        Returns:
            Dictionary with analysis results
        """
        if analysis_types is None:
            analysis_types = ['entities', 'sentiment', 'topics', 'keywords', 'readability']
        
        results = {}
        
        # Process text with spaCy
        doc = self.processor.process(text)
        
        # Extract requested analysis types
        if 'entities' in analysis_types:
            results['entities'] = self.entity_extractor.extract_entities(text)
            results['relationships'] = self.entity_extractor.extract_relationships(text)
        
        if 'sentiment' in analysis_types:
            results['sentiment'] = self.sentiment_analyzer.analyze_sentiment(text, detailed=True)
            results['emotions'] = self.sentiment_analyzer.detect_emotions(text)
            results['subjectivity'] = self.sentiment_analyzer.analyze_subjectivity(text)
        
        if 'topics' in analysis_types:
            results['topics'] = self.text_classifier.classify_with_rules(text, 'topic')
            results['intent'] = self.text_classifier.classify_with_rules(text, 'intent')
            results['content_type'] = self.text_classifier.classify_with_rules(text, 'content_type')
        
        if 'keywords' in analysis_types:
            results['keywords'] = extract_keywords(text, n=10)
            
        if 'readability' in analysis_types:
            results['readability'] = compute_readability_scores(text)
            
        if 'syntax' in analysis_types:
            results['syntax'] = self.processor.analyze_syntax(text)
        
        return results
    
    def enhance_llm_prompt(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Enhance a prompt for LLM with NLP-derived context
        
        Args:
            prompt: Original prompt
            context: Additional context information
            
        Returns:
            Enhanced prompt
        """
        # Analyze the prompt
        analysis = self.analyze_text(prompt, analysis_types=['entities', 'sentiment', 'topics'])
        
        # Extract key entities
        entities = [ent['text'] for ent in analysis.get('entities', [])[:5]]
        
        # Extract dominant sentiment
        sentiment = analysis.get('sentiment', {}).get('sentiment', 'neutral')
        
        # Extract dominant topics
        topics = list(analysis.get('topics', {}).keys())[:3]
        
        # Create context addition
        context_addition = "\nContext: "
        
        if entities:
            context_addition += f"Key entities: {', '.join(entities)}. "
            
        if sentiment != 'neutral':
            context_addition += f"Sentiment: {sentiment}. "
            
        if topics:
            context_addition += f"Topics: {', '.join(topics)}. "
            
        # Add custom context if provided
        if context:
            for key, value in context.items():
                if isinstance(value, str):
                    context_addition += f"{key}: {value}. "
                    
        # Append context to prompt if we have meaningful additions
        if len(context_addition) > 10:
            enhanced_prompt = prompt + context_addition
        else:
            enhanced_prompt = prompt
            
        return enhanced_prompt
    
    def extract_memory_elements(self, user_input: str, system_response: str = None) -> Dict[str, Any]:
        """
        Extract elements suitable for memory storage from conversation
        
        Args:
            user_input: User input text to analyze
            system_response: Optional system response
            
        Returns:
            Dictionary with memory elements
        """
        # Analyze user input
        user_analysis = self.analyze_text(user_input, 
                                   analysis_types=['entities', 'sentiment', 'topics', 'keywords'])
        
        # Extract memory-worthy elements from user input
        memory_elements = {
            'entities': [ent for ent in user_analysis.get('entities', []) 
                        if ent['type'] in ['PERSON', 'ORG', 'GPE', 'DATE', 'EVENT']],
            'topics': user_analysis.get('topics', {}),
            'keywords': user_analysis.get('keywords', []),
            'user_sentiment': user_analysis.get('sentiment', {}).get('overall_sentiment', 'neutral'),
        }
        
        # Analyze system response if provided
        if system_response:
            sys_analysis = self.analyze_text(system_response, 
                                      analysis_types=['entities', 'sentiment'])
            
            # Add system entities and sentiment
            sys_entities = [ent for ent in sys_analysis.get('entities', [])
                          if ent['type'] in ['PERSON', 'ORG', 'GPE', 'DATE', 'EVENT']]
            
            # Merge entities, avoiding duplicates
            existing_texts = {e['text'].lower() for e in memory_elements['entities']}
            for ent in sys_entities:
                if ent['text'].lower() not in existing_texts:
                    memory_elements['entities'].append(ent)
                    existing_texts.add(ent['text'].lower())
            
            memory_elements['system_sentiment'] = sys_analysis.get('sentiment', {}).get('overall_sentiment', 'neutral')
            
            # Analyze conversation as a whole
            full_text = user_input + " " + system_response
        else:
            full_text = user_input
        
        # Extract a summary if the text is long enough
        if len(full_text.split()) > 40:
            memory_elements['summary'] = summarize_text(full_text, max_sentences=2)
        else:
            memory_elements['summary'] = full_text[:100] + ('...' if len(full_text) > 100 else '')
        
        return memory_elements
    
    def enhance_user_input(self, user_input: str, analysis: Dict[str, Any] = None) -> str:
        """
        Enhance user input based on NLP analysis
        
        Args:
            user_input: Original user input
            analysis: Pre-computed analysis or None to compute it
            
        Returns:
            Enhanced user input (or original if no enhancement needed)
        """
        if analysis is None:
            analysis = self.analyze_text(user_input)
        
        # Don't modify short queries - they're likely specific
        if len(user_input.split()) < 5:
            return user_input
            
        entities = analysis.get('entities', [])
        sentiment = analysis.get('sentiment', {}).get('overall_sentiment', 'neutral')
        topics = analysis.get('topics', {})
        
        # No need to enhance if we don't have meaningful insights
        if not entities and sentiment == 'neutral' and not topics:
            return user_input
            
        # For now, just return the original input
        # In the future, this could actually modify the user input to enhance it
        # based on entity recognition, sentiment analysis, etc.
        return user_input
        
    def get_help(self) -> str:
        """
        Get help information about the NLP system
        
        Returns:
            Help text describing NLP capabilities
        """
        return """
# NLP System Capabilities

The NLP (Natural Language Processing) system enhances Anima's understanding of language with:

## Core Capabilities
- **Tokenization**: Breaking down text into words, phrases, symbols
- **Entity Recognition**: Identifying people, places, organizations, dates, etc.
- **Sentiment Analysis**: Detecting emotional tone (positive, negative, neutral)
- **Topic Classification**: Categorizing text by subject matter
- **Word Vectors**: Understanding semantic relationships between words

## Advanced Features
- **Memory Extraction**: Identifying key information worth remembering
- **Relationship Detection**: Finding connections between entities
- **Conversation Analysis**: Understanding dialogue context and flow
- **Prompt Enhancement**: Improving AI prompts with context awareness

All processing happens locally on your machine using spaCy's powerful language models.
"""
        
    def process_conversation(self, user_message: str, system_message: str = None) -> Dict[str, Any]:
        """
        Process a conversation turn, extracting insights
        
        Args:
            user_message: Message from the user
            system_message: Optional response from the system
            
        Returns:
            Dictionary with conversation insights
        """
        insights = {
            'user_message': self.analyze_text(user_message, 
                                           analysis_types=['entities', 'sentiment', 'topics']),
            'memory_elements': self.extract_memory_elements(user_message, system_message)
        }
        
        if system_message:
            insights['system_message'] = self.analyze_text(system_message, 
                                                        analysis_types=['entities', 'sentiment'])
            
            # Extract potential relationships between messages
            user_entities = set(ent['text'].lower() for ent in 
                               insights['user_message'].get('entities', []))
            
            system_entities = set(ent['text'].lower() for ent in 
                                 insights['system_message'].get('entities', []))
            
            # Find common entities
            insights['common_entities'] = list(user_entities.intersection(system_entities))
        
        return insights

# Create singleton instance for easy import
anima_nlp = None

def initialize(model_size: str = 'medium', use_gpu: bool = False) -> AnimalNLP:
    """
    Initialize the NLP system (singleton)
    
    Args:
        model_size: Size of spaCy model to use
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        AnimalNLP instance
    """
    global anima_nlp
    if anima_nlp is None:
        try:
            logger.info("Initializing NLP system...")
            anima_nlp = AnimalNLP(model_size=model_size, use_gpu=use_gpu)
            logger.info("NLP system initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize NLP system: {e}")
            # Return empty placeholder to prevent repeated initialization attempts
            return None
    return anima_nlp

def get_instance() -> Optional[AnimalNLP]:
    """
    Get the current NLP instance
    
    Returns:
        AnimalNLP instance or None if not initialized
    """
    global anima_nlp
    # Auto-initialize if not already done
    if anima_nlp is None:
        logger.info("Auto-initializing NLP system on first use")
        return initialize()
    return anima_nlp

# Auto-initialize the NLP system when this module is imported
try:
    logger.info("Auto-initializing NLP system on module import")
    initialize()
except Exception as e:
    logger.error(f"Error during auto-initialization of NLP system: {e}")
    # Don't propagate the exception to allow graceful degradation
