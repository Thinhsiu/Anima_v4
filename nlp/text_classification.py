"""
Text classification module for Anima_v4 - Advanced text categorization capabilities
"""

import spacy
from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassifier:
    """Advanced text classification with multiple algorithms and custom categories"""
    
    def __init__(self, nlp_processor):
        """
        Initialize the text classifier
        
        Args:
            nlp_processor: An instance of NLPProcessor
        """
        self.nlp = nlp_processor.nlp
        self.processor = nlp_processor
        
        # Dictionary to store classifiers
        self.classifiers = {}
        
        # Dictionary to store rule-based classifiers
        self.rule_classifiers = {}
        
        # Initialize pre-built classifiers
        self._initialize_default_classifiers()
    
    def _initialize_default_classifiers(self):
        """Initialize default rule-based classifiers for common tasks"""
        
        # Topic detection - simple keyword-based
        topics = {
            'technology': ['computer', 'software', 'hardware', 'internet', 'digital', 'tech', 
                          'app', 'data', 'algorithm', 'programming', 'code', 'device', 'ai',
                          'artificial intelligence', 'machine learning', 'neural', 'cyber'],
            'business': ['company', 'market', 'finance', 'investment', 'stock', 'profit', 
                        'revenue', 'startup', 'entrepreneur', 'business', 'corporate', 
                        'management', 'strategy', 'leadership', 'ceo', 'executive'],
            'health': ['health', 'medical', 'doctor', 'patient', 'hospital', 'disease', 
                      'treatment', 'medicine', 'therapy', 'wellness', 'fitness', 'exercise',
                      'diet', 'nutrition', 'symptom', 'diagnosis', 'healthcare'],
            'politics': ['government', 'politics', 'policy', 'election', 'president', 'senator',
                        'congress', 'campaign', 'vote', 'democrat', 'republican', 'law',
                        'legislation', 'political', 'party', 'candidate', 'ballot'],
            'entertainment': ['movie', 'film', 'music', 'song', 'album', 'artist', 'actor',
                             'actress', 'celebrity', 'entertainment', 'tv', 'television', 
                             'show', 'series', 'performance', 'concert', 'streaming'],
            'sports': ['game', 'team', 'player', 'score', 'win', 'lose', 'championship',
                      'tournament', 'match', 'sports', 'football', 'basketball', 'baseball',
                      'soccer', 'tennis', 'golf', 'athlete', 'coach', 'league'],
            'science': ['research', 'science', 'scientist', 'study', 'experiment', 'theory',
                       'discovery', 'physics', 'chemistry', 'biology', 'laboratory', 'evidence',
                       'hypothesis', 'scientific', 'academic', 'journal', 'publication']
        }
        
        def topic_classifier(text):
            doc = self.nlp(text.lower())
            
            # Count keyword matches for each topic
            scores = {topic: 0 for topic in topics}
            
            for token in doc:
                word = token.text.lower()
                lemma = token.lemma_.lower()
                
                for topic, keywords in topics.items():
                    if word in keywords or lemma in keywords:
                        scores[topic] += 1
                    
                    # Also check for phrases
                    for keyword in keywords:
                        if ' ' in keyword and keyword in text.lower():
                            scores[topic] += 2  # Higher weight for phrases
            
            # Return all scores above a threshold
            results = {topic: score for topic, score in scores.items() if score > 0}
            
            # If no topics found, return 'general'
            if not results:
                return {'general': 1.0}
                
            # Normalize scores
            total = sum(results.values())
            results = {k: v/total for k, v in results.items()}
            
            return results
        
        # Register the topic classifier
        self.rule_classifiers['topic'] = topic_classifier
        
        # Intent detection - simple rule-based
        intents = {
            'question': ['who', 'what', 'when', 'where', 'why', 'how', '?'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings'],
            'farewell': ['goodbye', 'bye', 'see you', 'talk to you later', 'have a good day', 'farewell'],
            'agreement': ['yes', 'sure', 'of course', 'absolutely', 'agree', 'correct', 'right', 'indeed'],
            'disagreement': ['no', 'not', 'disagree', 'incorrect', 'wrong', "don't agree", 'nope'],
            'request': ['please', 'could you', 'would you', 'can you', 'help', 'assist', 'show me', 'tell me'],
            'complaint': ['complaint', 'issue', 'problem', 'unhappy', 'disappointed', 'frustrated', 'annoyed', 'wrong']
        }
        
        def intent_classifier(text):
            text_lower = text.lower()
            doc = self.nlp(text_lower)
            
            # Check for question structure
            has_question_mark = '?' in text
            has_question_word = any(token.text.lower() in ['who', 'what', 'when', 'where', 'why', 'how'] 
                                   for token in doc)
            
            # Initialize scores
            scores = {intent: 0 for intent in intents}
            
            # Special case for questions
            if has_question_mark or has_question_word:
                scores['question'] += 5
            
            # Check for keywords
            for token in doc:
                word = token.text.lower()
                
                for intent, keywords in intents.items():
                    if word in keywords:
                        scores[intent] += 1
            
            # Check for phrases
            for intent, keywords in intents.items():
                for keyword in keywords:
                    if ' ' in keyword and keyword in text_lower:
                        scores[intent] += 2
            
            # Filter and normalize scores
            results = {intent: score for intent, score in scores.items() if score > 0}
            
            # If no intents found, return 'other'
            if not results:
                return {'other': 1.0}
                
            # Normalize scores
            total = sum(results.values())
            results = {k: v/total for k, v in results.items()}
            
            return results
            
        # Register the intent classifier
        self.rule_classifiers['intent'] = intent_classifier
        
        # Content type classifier
        content_types = {
            'factual': ['fact', 'study', 'research', 'according', 'evidence', 'data', 'statistics', 'report'],
            'opinion': ['think', 'believe', 'feel', 'opinion', 'perspective', 'view', 'consider', 'suggest'],
            'promotional': ['buy', 'sale', 'discount', 'offer', 'free', 'limited', 'exclusive', 'deal', 'promotion'],
            'personal': ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'],
            'instructional': ['how to', 'guide', 'steps', 'instructions', 'tutorial', 'learn', 'follow', 'process']
        }
        
        def content_type_classifier(text):
            text_lower = text.lower()
            doc = self.nlp(text_lower)
            
            # Initialize scores
            scores = {content_type: 0 for content_type in content_types}
            
            # Count factual indicators (numbers, statistics, citations)
            num_count = len([token for token in doc if token.like_num])
            if num_count > 2:
                scores['factual'] += num_count
                
            # Count personal pronouns
            personal_pronoun_count = len([token for token in doc 
                                         if token.text.lower() in ['i', 'me', 'my', 'mine', 'myself']])
            if personal_pronoun_count > 1:
                scores['personal'] += personal_pronoun_count * 2
                
            # Check for keywords
            for token in doc:
                word = token.text.lower()
                lemma = token.lemma_.lower()
                
                for content_type, keywords in content_types.items():
                    if word in keywords or lemma in keywords:
                        scores[content_type] += 1
            
            # Check for phrases
            for content_type, keywords in content_types.items():
                for keyword in keywords:
                    if ' ' in keyword and keyword in text_lower:
                        scores[content_type] += 2
            
            # Filter and normalize scores
            results = {content_type: score for content_type, score in scores.items() if score > 0}
            
            # If no content types found, determine based on length and structure
            if not results:
                if len(text.split()) > 50:  # Longer texts tend to be more factual
                    return {'factual': 0.7, 'other': 0.3}
                else:
                    return {'other': 1.0}
                
            # Normalize scores
            total = sum(results.values())
            results = {k: v/total for k, v in results.items()}
            
            return results
            
        # Register the content type classifier
        self.rule_classifiers['content_type'] = content_type_classifier
    
    def classify_with_rules(self, text: str, classifier_type: str) -> Dict[str, float]:
        """
        Classify text using rule-based classifiers
        
        Args:
            text: Text to classify
            classifier_type: Type of classifier to use ('topic', 'intent', etc.)
            
        Returns:
            Dictionary with category scores
        """
        if classifier_type not in self.rule_classifiers:
            logger.warning(f"Rule-based classifier '{classifier_type}' not found")
            return {'error': 1.0}
        
        classifier = self.rule_classifiers[classifier_type]
        return classifier(text)
    
    def create_classifier(self, name: str, categories: List[str], model_type: str = 'naive_bayes'):
        """
        Create a new machine learning classifier
        
        Args:
            name: Name for the classifier
            categories: List of categories for classification
            model_type: Type of model ('naive_bayes', 'logistic_regression', 'svm')
        """
        # Create feature extraction pipeline
        if model_type == 'naive_bayes':
            classifier = MultinomialNB()
        elif model_type == 'logistic_regression':
            classifier = LogisticRegression(max_iter=1000)
        elif model_type == 'svm':
            classifier = LinearSVC()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline with TF-IDF features
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('classifier', classifier)
        ])
        
        # Store classifier information
        self.classifiers[name] = {
            'pipeline': pipeline,
            'categories': categories,
            'trained': False,
            'model_type': model_type,
            'training_data': {category: [] for category in categories}
        }
        
        logger.info(f"Created {model_type} classifier '{name}' with categories: {categories}")
        
    def add_training_example(self, classifier_name: str, text: str, category: str):
        """
        Add a training example to a classifier
        
        Args:
            classifier_name: Name of the classifier
            text: Text example
            category: Category label
        """
        if classifier_name not in self.classifiers:
            logger.warning(f"Classifier '{classifier_name}' not found")
            return False
        
        classifier_info = self.classifiers[classifier_name]
        
        # Check if category is valid
        if category not in classifier_info['categories']:
            logger.warning(f"Category '{category}' not found in classifier '{classifier_name}'")
            return False
        
        # Add the example
        classifier_info['training_data'][category].append(text)
        logger.info(f"Added training example to '{classifier_name}' for category '{category}'")
        
        # Mark as untrained since new data was added
        classifier_info['trained'] = False
        return True
    
    def train_classifier(self, classifier_name: str) -> bool:
        """
        Train a classifier with its collected examples
        
        Args:
            classifier_name: Name of the classifier
            
        Returns:
            True if training was successful
        """
        if classifier_name not in self.classifiers:
            logger.warning(f"Classifier '{classifier_name}' not found")
            return False
        
        classifier_info = self.classifiers[classifier_name]
        training_data = classifier_info['training_data']
        
        # Check if we have enough data
        total_examples = sum(len(examples) for examples in training_data.values())
        if total_examples < 2:
            logger.warning(f"Not enough training examples for '{classifier_name}'")
            return False
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for category, examples in training_data.items():
            for text in examples:
                X_train.append(text)
                y_train.append(category)
        
        # Train the classifier
        try:
            classifier_info['pipeline'].fit(X_train, y_train)
            classifier_info['trained'] = True
            logger.info(f"Successfully trained classifier '{classifier_name}' with {total_examples} examples")
            return True
        except Exception as e:
            logger.error(f"Failed to train classifier '{classifier_name}': {str(e)}")
            return False
    
    def classify(self, classifier_name: str, text: str) -> Dict[str, float]:
        """
        Classify text using a trained classifier
        
        Args:
            classifier_name: Name of the classifier
            text: Text to classify
            
        Returns:
            Dictionary with category probabilities
        """
        if classifier_name not in self.classifiers:
            logger.warning(f"Classifier '{classifier_name}' not found")
            return {'error': 'Classifier not found'}
        
        classifier_info = self.classifiers[classifier_name]
        
        # Check if classifier is trained
        if not classifier_info['trained']:
            logger.warning(f"Classifier '{classifier_name}' is not trained")
            return {'error': 'Classifier not trained'}
        
        # Classify the text
        pipeline = classifier_info['pipeline']
        categories = classifier_info['categories']
        
        try:
            # Get prediction
            if hasattr(pipeline, 'predict_proba'):
                # For models that support probability estimates
                probas = pipeline.predict_proba([text])[0]
                return {category: float(proba) for category, proba in zip(categories, probas)}
            else:
                # For models that only support class predictions (like SVM)
                prediction = pipeline.predict([text])[0]
                return {category: 1.0 if category == prediction else 0.0 for category in categories}
        except Exception as e:
            logger.error(f"Failed to classify text: {str(e)}")
            return {'error': str(e)}
    
    def save_classifier(self, classifier_name: str, directory: str) -> bool:
        """
        Save a classifier to disk
        
        Args:
            classifier_name: Name of the classifier
            directory: Directory to save to
            
        Returns:
            True if saving was successful
        """
        if classifier_name not in self.classifiers:
            logger.warning(f"Classifier '{classifier_name}' not found")
            return False
        
        classifier_info = self.classifiers[classifier_name]
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Save the classifier
        try:
            file_path = os.path.join(directory, f"{classifier_name}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(classifier_info, f)
            logger.info(f"Saved classifier '{classifier_name}' to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save classifier '{classifier_name}': {str(e)}")
            return False
    
    def load_classifier(self, file_path: str) -> str:
        """
        Load a classifier from disk
        
        Args:
            file_path: Path to the classifier file
            
        Returns:
            Name of the loaded classifier
        """
        try:
            with open(file_path, 'rb') as f:
                classifier_info = pickle.load(f)
            
            # Extract classifier name from file path
            classifier_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Store the loaded classifier
            self.classifiers[classifier_name] = classifier_info
            
            logger.info(f"Loaded classifier '{classifier_name}' from {file_path}")
            return classifier_name
        except Exception as e:
            logger.error(f"Failed to load classifier from {file_path}: {str(e)}")
            return ""
