"""
Core NLP module for Anima_v4 - Provides the main processor for NLP tasks
"""

import spacy
from typing import Dict, List, Any, Optional, Union
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    """Main NLP processor class that handles all NLP operations"""
    
    # Available models with increasing capabilities
    AVAILABLE_MODELS = {
        "small": "en_core_web_sm",  # Fast but less accurate
        "medium": "en_core_web_md", # Good balance
        "large": "en_core_web_lg",  # More accurate but slower
        "trf": "en_core_web_trf"    # Most accurate (transformer-based)
    }
    
    def __init__(self, model_size: str = "medium", disable: List[str] = None):
        """
        Initialize the NLP processor with specified model
        
        Args:
            model_size: Size of the model to use ('small', 'medium', 'large', or 'trf')
            disable: Pipeline components to disable for better performance
        """
        self.model_name = self.AVAILABLE_MODELS.get(model_size, "en_core_web_md")
        
        # Try to load model, download if not available
        try:
            self.nlp = spacy.load(self.model_name, disable=disable)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.info(f"Downloading spaCy model: {self.model_name}")
            os.system(f"python -m spacy download {self.model_name}")
            self.nlp = spacy.load(self.model_name, disable=disable)
        
        # Keep track of custom components
        self.custom_components = {}
    
    def process(self, text: str) -> spacy.tokens.Doc:
        """
        Process text with the full NLP pipeline
        
        Args:
            text: Input text to process
            
        Returns:
            spacy.tokens.Doc object containing the processed text
        """
        return self.nlp(text)
    
    def tokenize(self, text: str, as_strings: bool = False) -> Union[List[str], spacy.tokens.Doc]:
        """
        Tokenize text into individual tokens
        
        Args:
            text: Text to tokenize
            as_strings: If True, returns list of token strings
            
        Returns:
            List of tokens or spaCy Doc
        """
        doc = self.nlp.make_doc(text)
        if as_strings:
            return [token.text for token in doc]
        return doc
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entities with their text, label and position
        """
        doc = self.process(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def analyze_syntax(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze syntactic structure of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with POS tags, dependencies and noun chunks
        """
        doc = self.process(text)
        
        # Get part-of-speech tags
        pos_tags = [{'text': token.text, 'pos': token.pos_, 'explanation': spacy.explain(token.pos_)}
                   for token in doc]
        
        # Get syntactic dependencies
        dependencies = [{'text': token.text, 
                        'dep': token.dep_, 
                        'head': token.head.text,
                        'explanation': spacy.explain(token.dep_)}
                       for token in doc]
        
        # Get noun chunks (noun phrases)
        noun_chunks = [{'text': chunk.text, 
                       'root': chunk.root.text}
                      for chunk in doc.noun_chunks]
        
        return {
            'pos_tags': pos_tags,
            'dependencies': dependencies,
            'noun_chunks': noun_chunks
        }
    
    def get_word_vectors(self, text: str) -> Dict[str, Any]:
        """
        Get word vectors for text
        
        Args:
            text: Text to get vectors for
            
        Returns:
            Dictionary with token vectors and document vector
        """
        # Check if the model has vectors
        if not self.nlp.has_pipe('tok2vec'):
            logger.warning("Current model doesn't support word vectors. Use 'medium' or larger model.")
            return {"error": "Model doesn't support vectors"}
        
        doc = self.process(text)
        
        # Get vectors for individual tokens
        token_vectors = {token.text: {"vector": token.vector.tolist(), "vector_norm": token.vector_norm} 
                        for token in doc if token.has_vector}
        
        # Get document vector (average of token vectors)
        doc_vector = doc.vector.tolist()
        
        return {
            "token_vectors": token_vectors,
            "document_vector": doc_vector
        }
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Process both texts
        doc1 = self.process(text1)
        doc2 = self.process(text2)
        
        # Calculate similarity
        return doc1.similarity(doc2)
    
    def add_custom_component(self, name: str, component: Any, before: Optional[str] = None):
        """
        Add a custom component to the NLP pipeline
        
        Args:
            name: Name of the component
            component: The component to add
            before: Optional name of component to add this before
        """
        self.nlp.add_pipe(component, name=name, before=before)
        self.custom_components[name] = component
        logger.info(f"Added custom component: {name}")
    
    def get_language_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistical information about the language in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language statistics
        """
        doc = self.process(text)
        
        # Count tokens by type
        token_types = {
            'words': len([token for token in doc if not token.is_punct and not token.is_space]),
            'punctuation': len([token for token in doc if token.is_punct]),
            'stopwords': len([token for token in doc if token.is_stop]),
            'numbers': len([token for token in doc if token.like_num]),
            'urls': len([token for token in doc if token.like_url]),
            'emails': len([token for token in doc if token.like_email])
        }
        
        # Count tokens by part of speech
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        
        # Readability metrics (basic)
        sentences = list(doc.sents)
        if len(sentences) > 0:
            avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
        else:
            avg_sentence_length = 0
        
        return {
            'token_count': len(doc),
            'token_types': token_types,
            'pos_counts': pos_counts,
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length
        }
