import os
import sys
import json
import uuid
import base64
import datetime
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import openai
from PIL import Image
from io import BytesIO

# Constants
VISION_DIR = os.path.join(parent_dir, "vision")
MEMORIES_DIR = os.path.join(VISION_DIR, "memories")
CACHE_DIR = os.path.join(VISION_DIR, "cache")

# Create directories if they don't exist
os.makedirs(MEMORIES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def encode_image_to_base64(image_path):
    """
    Encode an image to base64 for API requests
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, prompt=None):
    """
    Analyze an image using OpenAI's vision model
    
    Args:
        image_path (str): Path to the image file
        prompt (str, optional): Specific question or instruction about the image
        
    Returns:
        dict: Analysis results
    """
    try:
        # Encode the image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Create a unique cache key for this image+prompt combination
        cache_key = f"{os.path.basename(image_path)}_{hash(prompt or '')}.json"
        cache_path = os.path.join(CACHE_DIR, cache_key)
        
        # Check if we have a cached result
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Prepare the message content
        message_content = [
            {
                "type": "text", 
                "text": prompt or "What's in this image? Describe it in detail."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
        
        # Get the API key
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_4_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key found in environment variables")
            
        # Set the API key
        openai.api_key = api_key
            
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes images."},
                {"role": "user", "content": message_content}
            ],
            max_tokens=500
        )
        
        # Extract the analysis
        try:
            # Try new format first
            analysis = response.choices[0].message.content.strip()
        except AttributeError:
            # Fall back to dictionary access
            analysis = response['choices'][0]['message']['content'].strip()
        
        result = {
            "image_path": image_path,
            "prompt": prompt,
            "analysis": analysis,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Cache the result
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "image_path": image_path,
            "timestamp": datetime.datetime.now().isoformat()
        }

def save_image_memory(image_path, analysis=None, memory_type="shared", tags=None):
    """
    Save an image and its analysis as a memory
    
    Args:
        image_path (str): Path to the image file
        analysis (dict, optional): Analysis data
        memory_type (str): Type of memory (shared, personal, etc.)
        tags (list, optional): Tags to categorize the memory
        
    Returns:
        str: ID of the saved memory
    """
    try:
        # Generate a unique ID for the memory
        memory_id = str(uuid.uuid4())
        
        # Create a directory for this memory
        memory_dir = os.path.join(MEMORIES_DIR, memory_id)
        os.makedirs(memory_dir, exist_ok=True)
        
        # Copy the image to the memory directory
        image = Image.open(image_path)
        
        # Save in original format
        image_filename = os.path.basename(image_path)
        image_save_path = os.path.join(memory_dir, image_filename)
        image.save(image_save_path)
        
        # Also save a thumbnail for efficiency
        thumbnail_size = (300, 300)
        thumbnail = image.copy()
        thumbnail.thumbnail(thumbnail_size)
        thumbnail_path = os.path.join(memory_dir, "thumbnail.jpg")
        thumbnail.save(thumbnail_path, "JPEG")
        
        # If no analysis provided, generate one
        if not analysis:
            analysis = analyze_image(image_path)
        
        # Create metadata file
        metadata = {
            "id": memory_id,
            "original_path": image_path,
            "saved_path": image_save_path,
            "thumbnail_path": thumbnail_path,
            "analysis": analysis,
            "memory_type": memory_type,
            "tags": tags or [],
            "timestamp": datetime.datetime.now().isoformat(),
            "file_size": os.path.getsize(image_path),
            "dimensions": image.size
        }
        
        # Save metadata
        metadata_path = os.path.join(memory_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        return memory_id
        
    except Exception as e:
        print(f"Error saving image memory: {e}")
        return None

def get_memory(memory_id=None, limit=10, tags=None):
    """
    Retrieve image memories
    
    Args:
        memory_id (str, optional): Specific memory ID to retrieve
        limit (int): Maximum number of memories to return
        tags (list, optional): Filter by tags
        
    Returns:
        list: List of memory metadata
    """
    results = []
    
    try:
        # If specific memory requested
        if memory_id:
            memory_dir = os.path.join(MEMORIES_DIR, memory_id)
            if os.path.exists(memory_dir):
                metadata_path = os.path.join(memory_dir, "metadata.json")
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return [json.load(f)]
            return []
            
        # Get all memories
        memory_dirs = [d for d in os.listdir(MEMORIES_DIR) if os.path.isdir(os.path.join(MEMORIES_DIR, d))]
        
        for memory_dir in memory_dirs[:limit]:
            try:
                metadata_path = os.path.join(MEMORIES_DIR, memory_dir, "metadata.json")
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # Filter by tags if specified
                if tags:
                    if not isinstance(tags, list):
                        tags = [tags]
                    if any(tag in metadata.get("tags", []) for tag in tags):
                        results.append(metadata)
                else:
                    results.append(metadata)
            except Exception as e:
                print(f"Error reading memory {memory_dir}: {e}")
                
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return results[:limit]
        
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return []

def get_image_description(image_path, brief=False):
    """
    Get a description of an image
    
    Args:
        image_path (str): Path to the image
        brief (bool): Whether to get a brief description
        
    Returns:
        str: Description of the image
    """
    prompt = "Give a very brief description of what's in this image." if brief else "Describe this image in detail."
    result = analyze_image(image_path, prompt)
    
    if "error" in result:
        return f"Unable to analyze image: {result['error']}"
        
    return result["analysis"]

# Test function
def test_vision(image_path):
    """Test the vision functionality with an image"""
    print(f"Testing vision with image: {image_path}")
    
    analysis = analyze_image(image_path)
    print("\nAnalysis result:")
    print(json.dumps(analysis, indent=2))
    
    memory_id = save_image_memory(image_path, analysis)
    print(f"\nSaved image memory with ID: {memory_id}")
    
    memories = get_memory(memory_id)
    print("\nRetrieved memory:")
    print(json.dumps(memories, indent=2))
    
    return True

if __name__ == "__main__":
    # Test with a sample image if provided as argument
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        if os.path.exists(test_image):
            test_vision(test_image)
        else:
            print(f"Image not found: {test_image}")
    else:
        print("Please provide an image path as argument to test the vision module.")
