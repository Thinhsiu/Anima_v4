"""
Demonstration script for Anima_v4 NLP system

This script shows how to use the new NLP capabilities with your existing project.
Run this script to see the various NLP features in action.
"""

import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Import NLP system
from nlp import NLPProcessor, EntityExtractor, SentimentAnalyzer, TextClassifier
from nlp.integration import initialize as initialize_nlp, get_instance
from nlp.utils import extract_keywords, summarize_text, compute_readability_scores

# Import existing LLM functionality for integration examples
try:
    from llm import openai_llm, local_llm
    llm_available = True
except ImportError:
    logger.warning("LLM modules not available, some examples will be skipped")
    llm_available = False

def demo_nlp_core():
    """Demonstrate core NLP capabilities"""
    print("\n" + "="*80)
    print("CORE NLP CAPABILITIES DEMO")
    print("="*80)
    
    # Initialize the NLP processor
    print("\nInitializing NLP processor...")
    processor = NLPProcessor(model_size="medium")
    
    # Sample text for analysis
    text = """
    Apple Inc. is planning to open a new headquarters in Austin, Texas next year. 
    The company's CEO, Tim Cook, announced this on Thursday during a press conference.
    The new facility will employ over 5,000 people and cost approximately $1 billion.
    "We're thrilled to be expanding our operations in Austin," said Cook.
    The construction is expected to be completed by December 2026.
    """
    
    print(f"\nSample text for analysis:\n{text}\n")
    
    # Basic processing
    print("Processing text...")
    doc = processor.process(text)
    
    # Tokenization
    print("\nTokenization:")
    tokens = processor.tokenize(text, as_strings=True)
    print(f"Found {len(tokens)} tokens. First 10: {tokens[:10]}")
    
    # Entity extraction
    print("\nNamed Entity Recognition:")
    entities = processor.extract_entities(text)
    for entity in entities:
        print(f"  • {entity['text']} - {entity['label']} ({entity['description']})")
    
    # Syntactic analysis
    print("\nSyntactic Analysis:")
    syntax = processor.analyze_syntax(text)
    
    print("\nPart-of-speech tags (first 5):")
    for pos in syntax['pos_tags'][:5]:
        print(f"  • {pos['text']} - {pos['pos']} ({pos['explanation']})")
    
    print("\nDependencies (first 5):")
    for dep in syntax['dependencies'][:5]:
        print(f"  • {dep['text']} - {dep['dep']} to '{dep['head']}' ({dep['explanation']})")
    
    print("\nNoun chunks:")
    for chunk in syntax['noun_chunks']:
        print(f"  • {chunk['text']} (root: {chunk['root']})")

def demo_entity_extraction():
    """Demonstrate advanced entity extraction"""
    print("\n" + "="*80)
    print("ENTITY EXTRACTION DEMO")
    print("="*80)
    
    # Initialize processor and entity extractor
    processor = NLPProcessor(model_size="medium")
    entity_extractor = EntityExtractor(processor)
    
    # Sample text with various entity types
    text = """
    John Smith met with Sarah Johnson on January 15th, 2025, at 3:30 PM in New York City.
    They discussed the merger between Apple Inc. and TechCorp, which is valued at $50 billion.
    You can contact John at john.smith@example.com or call him at (555) 123-4567.
    The meeting was held at the Grand Hotel on 42nd Street.
    """
    
    print(f"\nSample text for entity extraction:\n{text}\n")
    
    # Extract entities using custom patterns
    print("Extracting entities with standard and custom patterns:")
    entities = entity_extractor.extract_entities(text)
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        entity_type = entity['type']
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity['text'])
    
    # Display grouped entities
    for entity_type, entity_texts in entity_groups.items():
        print(f"\n{entity_type}:")
        for text in entity_texts:
            print(f"  • {text}")
    
    # Extract potential relationships
    print("\nExtracting potential relationships:")
    relationships = entity_extractor.extract_relationships(text)
    
    for relation in relationships:
        print(f"  • {relation['subject']['text']} ({relation['subject']['type']}) "
              f"--[{relation['predicate']}]--> "
              f"{relation['object']['text']} ({relation['object']['type']})")

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities"""
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS DEMO")
    print("="*80)
    
    # Initialize processor and sentiment analyzer
    processor = NLPProcessor(model_size="medium")
    sentiment_analyzer = SentimentAnalyzer(processor)
    
    # Sample texts with different sentiments
    texts = [
        "I absolutely love this product! It's fantastic and works perfectly.",
        "This movie was terrible. I hated every minute of it and regret watching it.",
        "The service was okay. Nothing special but it got the job done.",
        "I'm incredibly frustrated with how poorly designed this app is.",
        "I'm excited about the upcoming changes but also a bit anxious about them."
    ]
    
    print("\nAnalyzing sentiment in various texts:")
    
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: {text}")
        
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze_sentiment(text)
        print(f"Sentiment: {sentiment['sentiment']} (score: {sentiment['score']:.2f})")
        
        # Detect emotions
        emotions = sentiment_analyzer.detect_emotions(text)
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
        print("Top emotions: " + ", ".join([f"{emotion} ({score:.2f})" for emotion, score in top_emotions if score > 0]))
        
        # Analyze subjectivity
        subjectivity = sentiment_analyzer.analyze_subjectivity(text)
        print(f"Subjectivity: {subjectivity['assessment']} (score: {subjectivity['score']:.2f})")

def demo_text_classification():
    """Demonstrate text classification capabilities"""
    print("\n" + "="*80)
    print("TEXT CLASSIFICATION DEMO")
    print("="*80)
    
    # Initialize processor and text classifier
    processor = NLPProcessor(model_size="medium")
    classifier = TextClassifier(processor)
    
    # Sample texts for classification
    texts = [
        "How do I reset my password for my account?",
        "I'd like to return the item I purchased yesterday.",
        "The latest iPhone has impressive battery life and camera quality.",
        "Mix 2 cups of flour with 1 egg and stir until combined.",
        "The stock market showed significant gains today, with tech companies leading."
    ]
    
    print("\nClassifying texts using rule-based classifiers:")
    
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: {text}")
        
        # Topic classification
        topics = classifier.classify_with_rules(text, 'topic')
        print("Topics: " + ", ".join([f"{topic} ({score:.2f})" for topic, score in topics.items()]))
        
        # Intent classification
        intents = classifier.classify_with_rules(text, 'intent')
        print("Intents: " + ", ".join([f"{intent} ({score:.2f})" for intent, score in intents.items()]))
        
        # Content type classification
        content_types = classifier.classify_with_rules(text, 'content_type')
        print("Content types: " + ", ".join([f"{ctype} ({score:.2f})" for ctype, score in content_types.items()]))

def demo_nlp_utils():
    """Demonstrate NLP utility functions"""
    print("\n" + "="*80)
    print("NLP UTILITIES DEMO")
    print("="*80)
    
    # Sample text for utilities
    text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence
    concerned with the interactions between computers and human language, in particular how to program computers
    to process and analyze large amounts of natural language data. The goal is a computer capable of
    "understanding" the contents of documents, including the contextual nuances of the language within them.
    The technology can then accurately extract information and insights contained in the documents as well as
    categorize and organize the documents themselves.
    
    Challenges in natural language processing frequently involve speech recognition, natural language understanding,
    and natural language generation. Modern NLP approaches based on machine learning are highly effective at many
    NLP tasks, though they have difficulty with common-sense reasoning, which is still an active area of NLP research.
    """
    
    print(f"\nSample text for utilities:\n{text}\n")
    
    # Extract keywords
    print("Extracting keywords:")
    keywords = extract_keywords(text, n=8)
    for keyword in keywords:
        print(f"  • {keyword['word']} (score: {keyword['score']:.3f}, count: {keyword['count']})")
    
    # Text summarization
    print("\nText summarization:")
    summary = summarize_text(text, max_sentences=2)
    print(summary)
    
    # Readability analysis
    print("\nReadability analysis:")
    readability = compute_readability_scores(text)
    print(f"  • Flesch Reading Ease: {readability['flesch_reading_ease']:.1f}")
    print(f"  • Flesch-Kincaid Grade Level: {readability['flesch_kincaid_grade']:.1f}")
    print(f"  • Gunning Fog Index: {readability['gunning_fog_index']:.1f}")
    print(f"  • Interpretation: {readability['interpretation']}")
    print(f"  • Average words per sentence: {readability['stats']['avg_words_per_sentence']:.1f}")
    print(f"  • Average syllables per word: {readability['stats']['avg_syllables_per_word']:.2f}")

def demo_llm_integration():
    """Demonstrate integration with existing LLM functionality"""
    if not llm_available:
        print("\nSkipping LLM integration demo (LLM modules not available)")
        return
    
    print("\n" + "="*80)
    print("LLM INTEGRATION DEMO")
    print("="*80)
    
    # Initialize the NLP system
    anima_nlp = initialize_nlp(model_size="medium")
    
    # Sample user message
    user_message = "Can you tell me about the climate change impact on coral reefs?"
    print(f"\nOriginal user message: {user_message}")
    
    # Enhance the prompt with NLP context
    enhanced_prompt = anima_nlp.enhance_llm_prompt(user_message)
    print(f"\nEnhanced prompt: {enhanced_prompt}")
    
    # Extract memory elements from user input
    memory_elements = anima_nlp.extract_memory_elements(user_message)
    print("\nExtracted memory elements:")
    for key, value in memory_elements.items():
        print(f"  • {key}: {value}")
    
    # Note: Actual LLM calls are commented out to avoid making API calls during demo
    print("\nNote: Actual LLM API calls are skipped in this demo")
    
    # Process conversation turn
    system_response = "Climate change is causing significant damage to coral reefs worldwide through ocean warming and acidification. Rising sea temperatures lead to coral bleaching, while increased CO2 levels make it harder for corals to build their calcium carbonate skeletons. Scientists predict that 70-90% of coral reefs could disappear in the next 20 years if current trends continue."
    
    print("\nSimulating a conversation turn analysis:")
    conversation_insights = anima_nlp.process_conversation(user_message, system_response)
    
    print("\nConversation insights:")
    print("  • User sentiment: " + conversation_insights['user_message']['sentiment']['sentiment'])
    print("  • System sentiment: " + conversation_insights['system_message']['sentiment']['sentiment'])
    if 'common_entities' in conversation_insights and conversation_insights['common_entities']:
        print("  • Common entities: " + ", ".join(conversation_insights['common_entities']))

def main():
    print("\n" + "="*80)
    print("ANIMA_V4 NLP SYSTEM DEMONSTRATION".center(80))
    print("="*80)
    print("\nThis script demonstrates the capabilities of the new NLP system")
    print("implemented for Anima_v4. It will guide you through various features.")
    
    # Prompt to install dependencies if necessary
    print("\nBefore proceeding, please ensure you have installed all required dependencies:")
    print("pip install -r requirements.txt")
    print("\nAlso, make sure to download the spaCy language model:")
    print("python -m spacy download en_core_web_md")
    
    # Pause for user to read
    input("\nPress Enter to start the demonstration...")
    
    # Run individual demos
    try:
        demo_nlp_core()
        input("\nPress Enter to continue to entity extraction demo...")
        
        demo_entity_extraction()
        input("\nPress Enter to continue to sentiment analysis demo...")
        
        demo_sentiment_analysis()
        input("\nPress Enter to continue to text classification demo...")
        
        demo_text_classification()
        input("\nPress Enter to continue to NLP utilities demo...")
        
        demo_nlp_utils()
        input("\nPress Enter to continue to LLM integration demo...")
        
        demo_llm_integration()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE".center(80))
        print("="*80)
        print("\nYou have now seen the main features of the Anima_v4 NLP system.")
        print("To use these capabilities in your project, import the modules from the nlp package.")
        print("\nExample usage:")
        print("  from nlp.integration import initialize_nlp")
        print("  nlp = initialize_nlp()")
        print("  results = nlp.analyze_text('Your text here')")
        print("\nRefer to the docstrings in each module for detailed usage instructions.")
    
    except Exception as e:
        logger.error(f"Demonstration error: {str(e)}", exc_info=True)
        print(f"\nAn error occurred during the demonstration: {str(e)}")
        print("You may need to install missing dependencies or download the required spaCy model.")

if __name__ == "__main__":
    main()
