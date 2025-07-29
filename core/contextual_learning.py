"""
Contextual Learning System for Anima

This module allows Anima to identify and remember explanations and teachings from conversations.
It detects when the user is explaining concepts and automatically stores that knowledge.
"""

import os
import sys
import re
import json
from pathlib import Path
import datetime
import time
import threading
import hashlib
import nltk

# Try to import nltk components, download if needed
try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
except ImportError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import the main awareness system
try:
    from core.awareness import awareness, add_conversation
except ImportError:
    print("Warning: Core awareness module not found. Some features may be limited.")


class ContextualLearning:
    """
    Contextual Learning system that detects explanations and teachings
    in conversation and stores them for future reference.
    """
    
    def __init__(self):
        """Initialize the contextual learning system"""
        self.memory_path = Path(parent_dir) / "memories" / "learned_concepts"
        self.memory_path.mkdir(exist_ok=True, parents=True)
        
        self.explanation_patterns = [
            r"(?:is|are|means|refers to|defined as)[\w\s]*",
            r"(?:called|known as|termed)[\w\s]*",
            r"(?:explanation|definition|concept)[\w\s]*",
            r"(?:let me explain|to clarify|to put it simply)[\w\s]*",
            r"(?:for example|e\.g\.|i\.e\.|such as)[\w\s]*"
        ]
        
        self.topics = self._load_topics()
        self.learned_concepts = self._load_concepts()
        
        # Start periodic saving thread
        self.save_thread = threading.Thread(target=self._periodic_save, daemon=True)
        self.save_thread.start()
    
    def _load_topics(self):
        """Load existing topics"""
        topics_file = self.memory_path / "topics.json"
        if topics_file.exists():
            try:
                with open(topics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _load_concepts(self):
        """Load existing learned concepts"""
        concepts = {}
        concepts_dir = self.memory_path / "concepts"
        concepts_dir.mkdir(exist_ok=True)
        
        for file in concepts_dir.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    concept_data = json.load(f)
                    concepts[concept_data.get("concept_id")] = concept_data
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading concept file {file}: {e}")
        
        return concepts
    
    def _save_topics(self):
        """Save topics to disk"""
        topics_file = self.memory_path / "topics.json"
        with open(topics_file, 'w', encoding='utf-8') as f:
            json.dump(self.topics, f, indent=2)
    
    def _save_concept(self, concept_data):
        """Save a single concept to disk"""
        concept_id = concept_data["concept_id"]
        file_path = self.memory_path / "concepts" / f"{concept_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(concept_data, f, indent=2)
    
    def _periodic_save(self):
        """Periodically save data to disk"""
        while True:
            time.sleep(300)  # Every 5 minutes
            try:
                self._save_topics()
                print("Contextual learning data saved")
            except Exception as e:
                print(f"Error saving contextual learning data: {e}")
    
    def detect_explanation(self, text):
        """
        Detect if the text contains an explanation
        Returns (is_explanation, concept, explanation)
        """
        # Check for common explanation patterns
        has_pattern = any(re.search(pattern, text.lower()) for pattern in self.explanation_patterns)
        
        # If no pattern found, likely not an explanation
        if not has_pattern:
            return False, None, None
        
        # Tokenize and tag parts of speech
        sentences = sent_tokenize(text)
        if not sentences:
            return False, None, None
            
        # Check first sentence for definition structure
        tagged_words = pos_tag(word_tokenize(sentences[0]))
        
        # Look for noun followed by verb patterns typical in definitions
        nouns = []
        for i, (word, tag) in enumerate(tagged_words):
            if tag.startswith('N'):  # Noun
                nouns.append(word)
                # Check if followed by linking verb
                if i + 1 < len(tagged_words) and tagged_words[i+1][1].startswith('VB'):
                    if tagged_words[i+1][0].lower() in ['is', 'are', 'means', 'refers']:
                        # Found potential concept definition
                        concept = word
                        explanation = text
                        return True, concept, explanation
        
        # Check for "X is called/known as Y" pattern
        for i, (word, tag) in enumerate(tagged_words):
            if i + 2 < len(tagged_words):
                if (tagged_words[i+1][0].lower() in ['is', 'are'] and
                    tagged_words[i+2][0].lower() in ['called', 'known']):
                    concept = word
                    explanation = text
                    return True, concept, explanation
        
        # If we get here, it might be an explanation but we couldn't extract a clear concept
        return True, None, text
    
    def extract_key_concepts(self, text):
        """Extract key concepts from text using POS tagging"""
        concepts = []
        
        # Tokenize and tag
        tagged_words = pos_tag(word_tokenize(text))
        
        # Extract nouns and noun phrases
        current_phrase = []
        for word, tag in tagged_words:
            if tag.startswith('N'):  # Noun
                current_phrase.append(word)
            elif current_phrase:
                if len(current_phrase) > 0:
                    concepts.append(' '.join(current_phrase))
                current_phrase = []
        
        # Add the last phrase if there is one
        if current_phrase:
            concepts.append(' '.join(current_phrase))
            
        # Filter out common words and short concepts
        filtered_concepts = [c for c in concepts if len(c) > 3]
        return filtered_concepts
    
    def learn_from_text(self, text, conversation_id=None):
        """
        Process text to learn any new concepts
        Returns True if something was learned
        """
        if not text or len(text) < 20:  # Too short to be an explanation
            return False
            
        is_explanation, concept, explanation = self.detect_explanation(text)
        
        if not is_explanation:
            return False
            
        # If we couldn't extract a specific concept, try to extract key concepts
        if not concept:
            potential_concepts = self.extract_key_concepts(text)
            if potential_concepts:
                concept = potential_concepts[0]  # Use the first extracted concept
            else:
                # Create a generic concept ID based on content
                concept = f"concept_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        
        # Store the concept
        concept_id = f"{concept.lower().replace(' ', '_')}_{hashlib.md5(explanation.encode()).hexdigest()[:8]}"
        
        # Check if this is new information or just a repeat
        if concept_id in self.learned_concepts:
            # Update existing concept with new information if substantially different
            existing = self.learned_concepts[concept_id]
            if explanation != existing["explanation"]:
                existing["explanation"] = explanation
                existing["last_updated"] = datetime.datetime.now().isoformat()
                existing["update_count"] = existing.get("update_count", 0) + 1
                self._save_concept(existing)
                return True
            return False
        
        # Extract potential topics
        topics = []
        
        # Try to extract topics from the conversation context
        try:
            if hasattr(awareness, "context_memory"):
                recent_exchanges = awareness.context_memory.get_recent_exchanges(3)
                for exchange in recent_exchanges:
                    if "topics" in exchange:
                        topics.extend(exchange["topics"])
        except Exception:
            pass
            
        # Create new concept entry
        concept_entry = {
            "concept_id": concept_id,
            "concept_name": concept,
            "explanation": explanation,
            "first_learned": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
            "topics": topics,
            "conversation_id": conversation_id,
            "update_count": 0
        }
        
        # Store concept
        self.learned_concepts[concept_id] = concept_entry
        self._save_concept(concept_entry)
        
        # Update topics
        for topic in topics:
            if topic in self.topics:
                if concept_id not in self.topics[topic]:
                    self.topics[topic].append(concept_id)
            else:
                self.topics[topic] = [concept_id]
        
        return True
    
    def get_relevant_concepts(self, text, max_concepts=3):
        """Get concepts relevant to the given text"""
        if not text or not self.learned_concepts:
            return []
            
        # Extract key terms from the query
        key_terms = self.extract_key_concepts(text)
        if not key_terms:
            return []
            
        # Score concepts based on relevance
        concept_scores = {}
        for concept_id, concept_data in self.learned_concepts.items():
            score = 0
            concept_name = concept_data["concept_name"]
            explanation = concept_data["explanation"]
            
            # Score based on direct concept name match
            for term in key_terms:
                if term.lower() in concept_name.lower():
                    score += 3
                    
                # Score based on explanation content
                if term.lower() in explanation.lower():
                    score += 1
            
            if score > 0:
                concept_scores[concept_id] = score
                
        # Get top scoring concepts
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.learned_concepts[concept_id] for concept_id, _ in sorted_concepts[:max_concepts]]
    
    def format_concepts_for_prompt(self, concepts):
        """Format concepts for inclusion in prompt"""
        if not concepts:
            return ""
            
        formatted = "\n\nRelevant learned concepts:\n"
        for i, concept in enumerate(concepts, 1):
            formatted += f"{i}. {concept['concept_name']}: {concept['explanation'][:150]}...\n"
            
        return formatted
    
    def enhance_prompt_with_concepts(self, prompt, user_input):
        """Enhance prompt with relevant concepts"""
        relevant_concepts = self.get_relevant_concepts(user_input)
        if not relevant_concepts:
            return prompt
            
        concepts_text = self.format_concepts_for_prompt(relevant_concepts)
        
        # Add to prompt in a location that won't interfere with system instructions
        split_prompt = prompt.split("\n\n")
        if len(split_prompt) > 1:
            # Insert after system instructions but before user input
            return "\n\n".join(split_prompt[:-1]) + concepts_text + "\n\n" + split_prompt[-1]
        else:
            return prompt + concepts_text


# Create global instance
contextual_learning = ContextualLearning()

# Exposed functions
def learn_from_text(text, conversation_id=None):
    """Learn concepts from text"""
    return contextual_learning.learn_from_text(text, conversation_id)

def get_relevant_concepts(text, max_concepts=3):
    """Get concepts relevant to text"""
    return contextual_learning.get_relevant_concepts(text, max_concepts)

def enhance_prompt_with_concepts(prompt, user_input):
    """Enhance prompt with relevant concepts"""
    return contextual_learning.enhance_prompt_with_concepts(prompt, user_input)

def process_response_for_learning(user_input, ai_response, conversation_id=None):
    """Process AI's response to learn from any explanations it gives"""
    if not ai_response:
        return
        
    # AI responses often contain explanations
    return contextual_learning.learn_from_text(ai_response, conversation_id)
