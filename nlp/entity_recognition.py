"""
Entity recognition module for Anima_v4 - Specialized entity extraction capabilities
"""

import spacy
from typing import Dict, List, Any, Optional, Tuple
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtractor:
    """Advanced entity extraction with custom patterns and rule-based matching"""
    
    def __init__(self, nlp_processor):
        """
        Initialize the entity extractor
        
        Args:
            nlp_processor: An instance of NLPProcessor
        """
        self.nlp = nlp_processor.nlp
        self.processor = nlp_processor
        
        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # Initialize custom entity patterns
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize built-in patterns for common entity types"""
        
        # Pattern for dates beyond what spaCy recognizes
        date_patterns = [
            # MM/DD/YYYY or DD/MM/YYYY
            [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{2,4}"}}],
            # Month name + day + year (e.g., January 1, 2022)
            [{"LOWER": {"IN": ["january", "february", "march", "april", "may", "june",
                              "july", "august", "september", "october", "november", "december"]}},
             {"TEXT": {"REGEX": r"\d{1,2}(st|nd|rd|th)?"}, "OP": "?"},
             {"TEXT": {"REGEX": r"\d{1,2}"}, "OP": "?"},
             {"TEXT": ",", "OP": "?"},
             {"TEXT": {"REGEX": r"\d{4}"}}]
        ]
        self.matcher.add("DATE_EXTENDED", date_patterns)
        
        # Pattern for time expressions
        time_patterns = [
            # HH:MM format (12 or 24 hour)
            [{"TEXT": {"REGEX": r"\d{1,2}:\d{2}"}}],
            # Time with AM/PM
            [{"TEXT": {"REGEX": r"\d{1,2}(:\d{2})?"}, "OP": "+"},
             {"LOWER": {"IN": ["am", "pm", "a.m.", "p.m."]}}]
        ]
        self.matcher.add("TIME", time_patterns)
        
        # Pattern for contact information
        contact_patterns = [
            # Phone numbers
            [{"TEXT": {"REGEX": r"(\+\d{1,3}[\s-])?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}"}}],
            # Emails (extending spaCy's detection)
            [{"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]
        ]
        self.matcher.add("CONTACT_INFO", contact_patterns)
    
    def add_custom_patterns(self, entity_type: str, patterns: List[List[Dict]]):
        """
        Add custom patterns for entity matching
        
        Args:
            entity_type: Name for the entity type
            patterns: List of pattern lists compatible with spaCy's Matcher
        """
        self.matcher.add(entity_type, patterns)
        logger.info(f"Added custom pattern for entity type: {entity_type}")
    
    def add_phrase_patterns(self, entity_type: str, phrases: List[str]):
        """
        Add exact phrase patterns for entity matching
        
        Args:
            entity_type: Name for the entity type
            phrases: List of phrases to match
        """
        phrase_patterns = [self.nlp(phrase) for phrase in phrases]
        self.phrase_matcher.add(entity_type, None, *phrase_patterns)
        logger.info(f"Added {len(phrases)} phrases for entity type: {entity_type}")
    
    def extract_entities(self, text: str, include_standard: bool = True) -> List[Dict[str, Any]]:
        """
        Extract entities from text using both spaCy's NER and custom patterns
        
        Args:
            text: Text to extract entities from
            include_standard: Whether to include standard spaCy entities
            
        Returns:
            List of entities with their text, type, and position
        """
        doc = self.nlp(text)
        entities = []
        
        # First add standard entities if requested
        if include_standard:
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start_char': ent.start_char,
                    'end_char': ent.end_char,
                    'source': 'spacy'
                })
        
        # Add custom pattern matches
        matcher_matches = self.matcher(doc)
        for match_id, start, end in matcher_matches:
            # Get the matched span
            entity_text = doc[start:end].text
            entity_type = self.nlp.vocab.strings[match_id]
            
            # Only add if it doesn't overlap with an existing entity
            overlap = False
            span = Span(doc, start, end, label=entity_type)
            
            for ent in entities:
                # Check for overlapping spans
                e_start, e_end = ent['start_char'], ent['end_char']
                s_start, s_end = span.start_char, span.end_char
                
                if (s_start <= e_end and s_end >= e_start):
                    overlap = True
                    break
            
            if not overlap:
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'description': f"Custom entity: {entity_type}",
                    'start_char': span.start_char,
                    'end_char': span.end_char,
                    'source': 'custom_pattern'
                })
        
        # Add phrase matches
        phrase_matches = self.phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            entity_text = doc[start:end].text
            entity_type = self.nlp.vocab.strings[match_id]
            
            # Only add if it doesn't overlap with an existing entity
            overlap = False
            span = Span(doc, start, end, label=entity_type)
            
            for ent in entities:
                e_start, e_end = ent['start_char'], ent['end_char']
                s_start, s_end = span.start_char, span.end_char
                
                if (s_start <= e_end and s_end >= e_start):
                    overlap = True
                    break
            
            if not overlap:
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'description': f"Phrase entity: {entity_type}",
                    'start_char': span.start_char,
                    'end_char': span.end_char,
                    'source': 'phrase_pattern'
                })
                
        # Sort entities by their position in text
        entities.sort(key=lambda x: x['start_char'])
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract potential relationships between entities
        
        Args:
            text: Text to extract relationships from
            
        Returns:
            List of potential entity relationships
        """
        doc = self.nlp(text)
        relationships = []
        
        for entity in doc.ents:
            # For each named entity, look for potential relationships
            if entity.label_ in ('PERSON', 'ORG', 'GPE'):
                # Find verbs that might connect this entity to others
                for token in doc:
                    if token.pos_ == 'VERB':
                        # Look for subject-verb-object relationships
                        subj = None
                        obj = None
                        
                        # Check if this entity is the subject
                        if any(t.idx >= entity.start_char and t.idx < entity.end_char for t in token.head.children if t.dep_ in ('nsubj', 'nsubjpass')):
                            subj = entity
                            
                            # Look for objects of this verb
                            for potential_obj in doc.ents:
                                if potential_obj != entity:
                                    if any(t.idx >= potential_obj.start_char and t.idx < potential_obj.end_char for t in token.children if t.dep_ in ('dobj', 'pobj')):
                                        obj = potential_obj
                                        break
                        
                        # Check if this entity is the object
                        elif any(t.idx >= entity.start_char and t.idx < entity.end_char for t in token.children if t.dep_ in ('dobj', 'pobj')):
                            obj = entity
                            
                            # Look for subjects of this verb
                            for potential_subj in doc.ents:
                                if potential_subj != entity:
                                    if any(t.idx >= potential_subj.start_char and t.idx < potential_subj.end_char for t in token.head.children if t.dep_ in ('nsubj', 'nsubjpass')):
                                        subj = potential_subj
                                        break
                        
                        # If we found both subject and object
                        if subj and obj:
                            relationships.append({
                                'subject': {
                                    'text': subj.text,
                                    'type': subj.label_
                                },
                                'predicate': token.text,
                                'object': {
                                    'text': obj.text,
                                    'type': obj.label_
                                }
                            })
        
        return relationships
