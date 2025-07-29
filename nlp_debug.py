"""
Debug script to fix NLP initialization issues
"""

import sys
import os
import logging
import traceback

# Set up verbose logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NLP-DEBUG")

def check_spacy_models():
    """Check if required spaCy models are installed"""
    logger.info("Checking spaCy installation...")
    try:
        import spacy
        logger.info(f"spaCy version: {spacy.__version__}")
        
        # Check available models
        logger.info("Checking installed models:")
        for name in ["en_core_web_sm", "en_core_web_md"]:
            try:
                spacy.load(name)
                logger.info(f"✓ {name} - INSTALLED")
            except OSError:
                logger.warning(f"✗ {name} - NOT INSTALLED")
                logger.info(f"Installing {name}...")
                os.system(f"python -m spacy download {name}")
                try:
                    spacy.load(name)
                    logger.info(f"✓ {name} - INSTALLED SUCCESSFULLY")
                except OSError as e:
                    logger.error(f"Failed to install {name}: {e}")
        return True
    except ImportError as e:
        logger.error(f"spaCy import error: {e}")
        return False

def fix_nlp_integration():
    """Fix the NLP integration issues"""
    logger.info("Attempting to fix NLP integration...")
    try:
        # Try importing the NLP system
        from nlp.integration import initialize as initialize_nlp
        logger.info("Initializing NLP system...")
        nlp_instance = initialize_nlp(model_size="small")  # Use small model for faster init
        
        if nlp_instance:
            logger.info("NLP system initialized successfully!")
            test_text = "Apple is looking at buying U.K. startup for $1 billion"
            logger.info(f"Testing with text: '{test_text}'")
            result = nlp_instance.analyze_text(test_text)
            logger.info(f"Analysis result: {result}")
            return True
        else:
            logger.error("NLP system initialization returned None")
            return False
    except Exception as e:
        logger.error(f"Error in NLP integration: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    logger.info("Starting NLP debug...")
    
    # First check spaCy models
    models_ok = check_spacy_models()
    if not models_ok:
        logger.error("Failed to verify spaCy models")
        return
    
    # Try fixing the integration
    integration_ok = fix_nlp_integration()
    if integration_ok:
        logger.info("NLP system is now working correctly!")
        logger.info("You can now run Anima with the NLP system enabled.")
    else:
        logger.error("Failed to fix NLP integration issues")

if __name__ == "__main__":
    main()
