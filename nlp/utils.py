"""
Utility functions for NLP processing in Anima_v4
"""

import re
import string
from typing import List, Dict, Any, Set, Optional, Tuple
import logging
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text by removing excess whitespace and normalizing
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text

def extract_keywords(text: str, n: int = 10, min_length: int = 3, exclude_stopwords: bool = True) -> List[Dict[str, Any]]:
    """
    Extract keywords from text using frequency analysis
    
    Args:
        text: Text to extract keywords from
        n: Number of keywords to extract
        min_length: Minimum length of keywords
        exclude_stopwords: Whether to exclude stopwords
        
    Returns:
        List of keywords with scores
    """
    # Common English stopwords
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when',
                'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those', 'then',
                'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of',
                'while', 'during', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
                'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
                'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                'only', 'own', 'same', 'too', 'very', 'can', 'will', 'should', 'now', 'with',
                'be', 'have', 'has', 'had', 'do', 'does', 'did', 'am', 'are', 'was', 'were',
                'been', 'being', 'having', 'doing', 'at', 'by', 'it', 'its'}
    
    # Clean text and convert to lowercase
    text = clean_text(text.lower())
    
    # Tokenize text by splitting on non-alphanumeric characters
    words = re.findall(r'\b[a-z0-9]+\b', text)
    
    # Filter words
    if exclude_stopwords:
        words = [word for word in words if word not in stopwords and len(word) >= min_length]
    else:
        words = [word for word in words if len(word) >= min_length]
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Get top N keywords
    keywords = [{'word': word, 'count': count, 'score': count / len(words)}
               for word, count in word_counts.most_common(n)]
    
    return keywords

def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Generate a simple extractive summary of text
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Summarized text
    """
    # Clean text
    text = clean_text(text)
    
    # Split into sentences (simple split on .!?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If there are fewer sentences than requested, return the whole text
    if len(sentences) <= max_sentences:
        return text
    
    # Calculate sentence scores based on word frequency
    word_freq = Counter(re.findall(r'\b\w+\b', text.lower()))
    
    # Score sentences
    sentence_scores = []
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 3:
            continue
            
        # Calculate score based on word frequency
        score = sum(word_freq[word.lower()] for word in re.findall(r'\b\w+\b', sentence))
        # Normalize by sentence length
        score = score / len(re.findall(r'\b\w+\b', sentence))
        
        sentence_scores.append((score, sentence))
    
    # Sort sentences by score and take top max_sentences
    summary_sentences = [s for _, s in sorted(sentence_scores, reverse=True)[:max_sentences]]
    
    # Reorder sentences to maintain original order
    ordered_summary = []
    for sentence in sentences:
        if sentence in summary_sentences:
            ordered_summary.append(sentence)
            if len(ordered_summary) >= max_sentences:
                break
    
    return ' '.join(ordered_summary)

def compute_readability_scores(text: str) -> Dict[str, Any]:
    """
    Compute various readability metrics
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with readability scores
    """
    # Clean text
    text = clean_text(text)
    
    # Count sentences, words and syllables
    sentences = re.split(r'(?<=[.!?])\s+', text)
    words = re.findall(r'\b\w+\b', text)
    
    # Count syllables (simplified method)
    def count_syllables(word):
        word = word.lower()
        
        # Exception cases
        exceptions = {
            "the": 1, "every": 2, "just": 1, "once": 1, 
            "something": 2, "nothing": 2, "anywhere": 3
        }
        if word in exceptions:
            return exceptions[word]
            
        # Remove ending 'e', 'es', 'ed'
        if word.endswith('e'):
            word = word[:-1]
        elif word.endswith(('es', 'ed')):
            word = word[:-2]
        
        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))
        
        # Ensure at least one syllable
        return max(1, count)
    
    num_sentences = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(word) for word in words)
    num_complex_words = sum(1 for word in words if count_syllables(word) >= 3)
    
    # Calculate readability scores
    if num_sentences == 0 or num_words == 0:
        return {"error": "Text too short for readability analysis"}
    
    # Flesch Reading Ease
    flesch = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    
    # Flesch-Kincaid Grade Level
    fk_grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    
    # Gunning Fog
    fog = 0.4 * ((num_words / num_sentences) + 100 * (num_complex_words / num_words))
    
    # Interpretation of Flesch Reading Ease
    if flesch >= 90:
        interpretation = "Very Easy - 5th Grade"
    elif flesch >= 80:
        interpretation = "Easy - 6th Grade"
    elif flesch >= 70:
        interpretation = "Fairly Easy - 7th Grade"
    elif flesch >= 60:
        interpretation = "Standard - 8th-9th Grade"
    elif flesch >= 50:
        interpretation = "Fairly Difficult - 10th-12th Grade"
    elif flesch >= 30:
        interpretation = "Difficult - College"
    else:
        interpretation = "Very Difficult - College Graduate"
    
    return {
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fk_grade,
        "gunning_fog_index": fog,
        "interpretation": interpretation,
        "stats": {
            "num_sentences": num_sentences,
            "num_words": num_words,
            "num_syllables": num_syllables,
            "num_complex_words": num_complex_words,
            "avg_words_per_sentence": num_words / num_sentences,
            "avg_syllables_per_word": num_syllables / num_words
        }
    }

def detect_language(text: str) -> str:
    """
    Detect the language of text using character frequency analysis
    (simplified version for common languages)
    
    Args:
        text: Text to analyze
        
    Returns:
        ISO language code of detected language
    """
    # Language character frequency profiles (simplified)
    language_profiles = {
        'en': {'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 'n': 6.7, 's': 6.3},  # English
        'es': {'e': 13.7, 'a': 11.5, 'o': 8.7, 's': 7.2, 'r': 6.9, 'n': 6.7, 'i': 6.3},  # Spanish
        'fr': {'e': 14.7, 'a': 8.0, 's': 7.9, 'i': 7.5, 't': 7.0, 'n': 7.0, 'r': 6.5},  # French
        'de': {'e': 17.4, 'n': 10.0, 'i': 7.6, 's': 7.3, 'r': 7.0, 't': 6.1, 'a': 6.5},  # German
    }
    
    # Clean and lowercase text
    text = ''.join(c for c in text.lower() if c.isalpha())
    
    # Count character frequencies
    total_chars = len(text)
    if total_chars < 10:  # Too short to detect reliably
        return "unknown"
        
    char_counts = Counter(text)
    
    # Calculate frequency percentages
    char_freqs = {char: (count/total_chars) * 100 for char, count in char_counts.items()}
    
    # Compare with language profiles
    best_match = "en"  # Default to English
    best_score = float('inf')
    
    for lang, profile in language_profiles.items():
        # Calculate difference score (lower is better)
        score = 0
        for char, expected_freq in profile.items():
            actual_freq = char_freqs.get(char, 0)
            score += abs(expected_freq - actual_freq)
        
        # Update best match if this language is closer
        if score < best_score:
            best_score = score
            best_match = lang
    
    return best_match
