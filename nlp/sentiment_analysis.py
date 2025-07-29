"""
Sentiment analysis module for Anima_v4 - Advanced sentiment and emotion detection
"""

import spacy
from typing import Dict, List, Any, Optional, Union
import re
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

class SentimentAnalyzer:
    """Advanced sentiment and emotion analysis"""
    
    def __init__(self, nlp_processor):
        """
        Initialize the sentiment analyzer
        
        Args:
            nlp_processor: An instance of NLPProcessor
        """
        self.nlp = nlp_processor.nlp
        self.processor = nlp_processor
        
        # Load sentiment lexicons
        self._load_lexicons()
        
        # Emotion categories and their associated terms
        self._emotion_lexicon = {
            'joy': ['happy', 'delighted', 'pleased', 'glad', 'joyful', 'enjoy', 'cheerful', 
                   'content', 'satisfied', 'thrilled', 'elated', 'jubilant', 'blissful'],
            'sadness': ['sad', 'unhappy', 'miserable', 'depressed', 'gloomy', 'heartbroken', 
                       'melancholy', 'grief', 'sorrow', 'dejected', 'dismal', 'downcast'],
            'anger': ['angry', 'furious', 'outraged', 'enraged', 'hostile', 'irritated', 
                     'annoyed', 'frustrated', 'mad', 'incensed', 'infuriated', 'irate'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'fearful', 'anxious', 
                    'worried', 'nervous', 'panicked', 'alarmed', 'dread', 'horrified'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 
                        'startled', 'bewildered', 'dumbfounded', 'flabbergasted'],
            'disgust': ['disgusted', 'revolted', 'nauseated', 'repulsed', 'loathing', 
                       'abhorrence', 'aversion', 'repelled', 'sickened']
        }
    
    def _load_lexicons(self):
        """Load sentiment lexicons from files or create basic ones"""
        # Create basic positive and negative word lists
        # In a production system, you would load comprehensive lexicons from files
        self.positive_words = {
            'good': 3, 'great': 4, 'excellent': 5, 'awesome': 5, 'nice': 3,
            'wonderful': 4, 'fantastic': 5, 'amazing': 5, 'love': 4, 'best': 5,
            'beautiful': 4, 'perfect': 5, 'happy': 4, 'glad': 3, 'pleased': 3,
            'enjoy': 3, 'liked': 3, 'helpful': 3, 'positive': 3, 'recommend': 4
        }
        
        self.negative_words = {
            'bad': -3, 'terrible': -5, 'awful': -4, 'horrible': -5, 'poor': -3,
            'disappointing': -3, 'worst': -5, 'hate': -4, 'dislike': -3, 'negative': -3,
            'ugly': -3, 'annoying': -3, 'frustrating': -4, 'useless': -4, 'angry': -3,
            'sad': -3, 'failed': -3, 'broken': -3, 'problem': -2, 'expensive': -2
        }
        
        # Intensifiers modify sentiment scores
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.5, 'so': 1.5, 'too': 1.3,
            'completely': 1.8, 'absolutely': 2.0, 'totally': 1.8, 'utterly': 1.8,
            'highly': 1.5, 'incredibly': 1.8, 'exceptionally': 1.8
        }
        
        # Negators flip sentiment polarity
        self.negators = [
            'not', "n't", 'never', 'no', 'neither', 'nor', 'hardly', 'barely',
            'scarcely', 'seldom', 'rarely'
        ]
    
    def analyze_sentiment(self, text: str, detailed: bool = False) -> Dict[str, Any]:
        """
        Analyze the sentiment of text
        
        Args:
            text: Text to analyze
            detailed: Whether to return detailed breakdown
            
        Returns:
            Dictionary with sentiment scores and analysis
        """
        doc = self.nlp(text)
        
        # Track sentiment at sentence level
        sentences = list(doc.sents)
        sentence_sentiments = []
        
        # Track overall scores
        overall_score = 0
        word_count = 0
        positive_words = []
        negative_words = []
        
        for sent in sentences:
            sent_score = 0
            sent_word_count = 0
            
            # Process each token in the sentence
            skip_tokens = 0  # Used to skip words after processing multi-word expressions
            for i, token in enumerate(sent):
                if skip_tokens > 0:
                    skip_tokens -= 1
                    continue
                
                # Skip punctuation and stop words
                if token.is_punct or token.is_space:
                    continue
                
                # Check for negation in a window before the current token
                negated = any(sent[max(0, i-3):i].text.lower() in self.negators for i in range(max(0, i-3), i+1))
                
                # Check for intensifiers
                intensifier = 1.0
                for j in range(max(0, i-3), i):
                    if j < len(sent) and sent[j].text.lower() in self.intensifiers:
                        intensifier = self.intensifiers[sent[j].text.lower()]
                        break
                
                # Look for the word or lemma in our sentiment lexicons
                word = token.text.lower()
                lemma = token.lemma_.lower()
                
                score = 0
                # Check positive lexicon
                if word in self.positive_words:
                    score = self.positive_words[word]
                elif lemma in self.positive_words:
                    score = self.positive_words[lemma]
                
                # Check negative lexicon
                elif word in self.negative_words:
                    score = self.negative_words[word]
                elif lemma in self.negative_words:
                    score = self.negative_words[lemma]
                
                # Apply negation and intensifiers
                if score != 0:
                    if negated:
                        score = -score
                    score *= intensifier
                    
                    # Add to appropriate list for reporting
                    if score > 0:
                        positive_words.append({'word': token.text, 'score': score})
                    elif score < 0:
                        negative_words.append({'word': token.text, 'score': score})
                    
                    sent_score += score
                    sent_word_count += 1
            
            # Add sentence sentiment to list
            if sent_word_count > 0:
                avg_sent_score = sent_score / sent_word_count
                sentence_sentiments.append({
                    'text': sent.text,
                    'score': avg_sent_score,
                    'sentiment': self._categorize_sentiment(avg_sent_score)
                })
                
                overall_score += sent_score
                word_count += sent_word_count
        
        # Calculate overall sentiment
        if word_count > 0:
            avg_score = overall_score / word_count
            sentiment = self._categorize_sentiment(avg_score)
        else:
            avg_score = 0
            sentiment = 'neutral'
        
        # Build result
        result = {
            'sentiment': sentiment,
            'score': avg_score,
            'sentence_count': len(sentences)
        }
        
        if detailed:
            result['sentences'] = sentence_sentiments
            result['positive_words'] = sorted(positive_words, key=lambda x: -x['score'])[:10]  # Top 10
            result['negative_words'] = sorted(negative_words, key=lambda x: x['score'])[:10]   # Top 10
        
        return result
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize numerical sentiment score into a text label"""
        if score > 0.6:
            return 'very positive'
        elif score > 0.2:
            return 'positive'
        elif score > -0.2:
            return 'neutral'
        elif score > -0.6:
            return 'negative'
        else:
            return 'very negative'
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion scores
        """
        doc = self.nlp(text.lower())
        
        # Initialize emotion counters
        emotions = {emotion: 0 for emotion in self._emotion_lexicon}
        
        # Count emotion words
        for token in doc:
            word = token.text.lower()
            lemma = token.lemma_.lower()
            
            for emotion, terms in self._emotion_lexicon.items():
                if word in terms or lemma in terms:
                    emotions[emotion] += 1
                
                # Also check for partial matches for emotion terms
                for term in terms:
                    if len(term) > 4 and (term in word or term in lemma):
                        emotions[emotion] += 0.5
        
        # Normalize scores (if any emotions detected)
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        
        return emotions
    
    def analyze_subjectivity(self, text: str) -> Dict[str, Any]:
        """
        Analyze the subjectivity of text (objective vs subjective)
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with subjectivity score and analysis
        """
        doc = self.nlp(text)
        
        # Factors that indicate subjectivity
        personal_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
        opinion_verbs = ['think', 'believe', 'feel', 'consider', 'assume', 'suggest', 'guess', 
                        'suppose', 'suspect', 'reckon', 'presume', 'doubt']
        subjective_adverbs = ['clearly', 'obviously', 'certainly', 'undoubtedly', 'definitely', 
                             'arguably', 'surely', 'frankly', 'honestly', 'apparently']
        
        # Count subjective elements
        pronoun_count = 0
        opinion_verb_count = 0
        adverb_count = 0
        sentiment_word_count = 0
        
        for token in doc:
            # Check for personal pronouns
            if token.text.lower() in personal_pronouns:
                pronoun_count += 1
            
            # Check for opinion verbs
            if token.lemma_.lower() in opinion_verbs:
                opinion_verb_count += 1
            
            # Check for subjective adverbs
            if token.text.lower() in subjective_adverbs:
                adverb_count += 1
            
            # Check for sentiment-bearing words
            if token.text.lower() in self.positive_words or token.text.lower() in self.negative_words:
                sentiment_word_count += 1
            elif token.lemma_.lower() in self.positive_words or token.lemma_.lower() in self.negative_words:
                sentiment_word_count += 1
        
        # Calculate weighted subjectivity score
        total_words = len([token for token in doc if not token.is_punct and not token.is_space])
        if total_words == 0:
            return {"score": 0, "assessment": "neutral"}
        
        # Weights for different factors
        weights = {
            'pronoun': 0.3,
            'opinion_verb': 0.25,
            'adverb': 0.15,
            'sentiment': 0.3
        }
        
        # Normalized scores
        scores = {
            'pronoun': min(pronoun_count / max(total_words/10, 1), 1),
            'opinion_verb': min(opinion_verb_count / max(total_words/20, 1), 1),
            'adverb': min(adverb_count / max(total_words/20, 1), 1),
            'sentiment': min(sentiment_word_count / max(total_words/5, 1), 1)
        }
        
        # Weighted average
        subjectivity = sum(scores[k] * weights[k] for k in weights)
        
        # Categorize
        if subjectivity < 0.2:
            assessment = "highly objective"
        elif subjectivity < 0.4:
            assessment = "somewhat objective"
        elif subjectivity < 0.6:
            assessment = "balanced"
        elif subjectivity < 0.8:
            assessment = "somewhat subjective"
        else:
            assessment = "highly subjective"
        
        return {
            "score": subjectivity,
            "assessment": assessment,
            "factors": {
                "personal_pronouns": pronoun_count,
                "opinion_verbs": opinion_verb_count,
                "subjective_adverbs": adverb_count,
                "sentiment_words": sentiment_word_count
            }
        }
