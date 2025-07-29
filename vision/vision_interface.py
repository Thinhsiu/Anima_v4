import os
import sys
import time
import datetime
from pathlib import Path
import shutil
import json

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from vision.vision_processor import analyze_image, save_image_memory, get_memory

# Constants
UPLOAD_DIR = os.path.join(parent_dir, "vision", "uploads")
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_image_file(file_path, user_query=None, save_memory=True):
    """
    Process an image file and optionally save it as a memory
    
    Args:
        file_path (str): Path to the image file
        user_query (str, optional): User's question about the image
        save_memory (bool): Whether to save the image as a memory
        
    Returns:
        dict: Analysis results
    """
    # Verify file exists and is an image
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        return {"error": f"Unsupported file format: {file_ext}. Supported formats: {', '.join(SUPPORTED_FORMATS)}"}
    
    # Process the image with vision model
    analysis = analyze_image(file_path, user_query)
    
    # Save as memory if requested
    if save_memory:
        memory_id = save_image_memory(file_path, analysis)
        analysis["memory_id"] = memory_id
    
    return analysis

def process_uploaded_file(file_data, filename, user_query=None):
    """
    Process an uploaded file
    
    Args:
        file_data (bytes): Raw file data
        filename (str): Original filename
        user_query (str, optional): User's question about the image
        
    Returns:
        dict: Analysis results
    """
    # Generate a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(file_data)
    
    # Process the image
    result = process_image_file(file_path, user_query)
    
    # Add file info to result
    result["file_path"] = file_path
    result["original_filename"] = filename
    
    return result

def handle_clipboard_image():
    """
    Handle images from clipboard - saves to temp file and processes
    
    Returns:
        dict: Analysis results or error
    """
    try:
        from PIL import ImageGrab
        import tempfile
        
        # Attempt to grab image from clipboard
        img = ImageGrab.grabclipboard()
        
        if img is None:
            return {"error": "No image found in clipboard"}
        
        # Save to temporary file
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        temp_file = os.path.join(UPLOAD_DIR, f"clipboard_{timestamp}.png")
        img.save(temp_file, "PNG")
        
        # Process the image
        result = process_image_file(temp_file)
        result["source"] = "clipboard"
        result["file_path"] = temp_file
        
        return result
    except ImportError:
        return {"error": "PIL or ImageGrab not available. Install with: pip install pillow"}
    except Exception as e:
        return {"error": f"Error processing clipboard image: {str(e)}"}

def watch_directory(directory_path, polling_interval=2.0):
    """
    Watch a directory for new images and process them
    
    Args:
        directory_path (str): Path to watch for new images
        polling_interval (float): How often to check for new files (seconds)
        
    Returns:
        Generator that yields analysis results for new images
    """
    # Keep track of processed files
    processed_files = set()
    
    # Make sure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        
    print(f"Watching directory for new images: {directory_path}")
    print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
    
    while True:
        # Get all image files in the directory
        all_files = []
        for ext in SUPPORTED_FORMATS:
            all_files.extend(Path(directory_path).glob(f"*{ext}"))
            all_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
        
        # Process new files
        for file_path in all_files:
            file_path_str = str(file_path)
            if file_path_str not in processed_files:
                # Process the new file
                result = process_image_file(file_path_str)
                processed_files.add(file_path_str)
                yield result
        
        # Wait before checking again
        time.sleep(polling_interval)

def drag_drop_handler(file_path, user_query=None):
    """
    Handle drag and drop of an image file
    
    Args:
        file_path (str): Path to the dragged image file
        user_query (str, optional): User's question about the image
        
    Returns:
        dict: Analysis results
    """
    # Copy file to uploads directory
    filename = os.path.basename(file_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    dest_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    shutil.copy2(file_path, dest_path)
    
    # Process the image
    result = process_image_file(dest_path, user_query)
    result["source"] = "drag_drop"
    result["original_path"] = file_path
    
    return result

def list_recent_memories(limit=10, include_analysis=False):
    """
    List recent image memories
    
    Args:
        limit (int): Maximum number of memories to return
        include_analysis (bool): Whether to include the full analysis
        
    Returns:
        list: Recent memories
    """
    memories = get_memory(limit=limit)
    
    if not include_analysis:
        # Remove detailed analysis to keep response shorter
        for memory in memories:
            if "analysis" in memory:
                if isinstance(memory["analysis"], dict) and "analysis" in memory["analysis"]:
                    memory["analysis"] = {"brief": memory["analysis"]["analysis"][:100] + "..."}
                else:
                    memory["analysis"] = {"brief": str(memory["analysis"])[:100] + "..."}
    
    return memories

if __name__ == "__main__":
    # Simple command line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Anima's vision capabilities")
    parser.add_argument("--image", help="Path to image file to process")
    parser.add_argument("--query", help="Question about the image")
    parser.add_argument("--clipboard", action="store_true", help="Process image from clipboard")
    parser.add_argument("--watch", help="Watch directory for new images")
    
    args = parser.parse_args()
    
    if args.image:
        result = process_image_file(args.image, args.query)
        print(json.dumps(result, indent=2))
    elif args.clipboard:
        result = handle_clipboard_image()
        print(json.dumps(result, indent=2))
    elif args.watch:
        for result in watch_directory(args.watch):
            print("\nNew image detected:")
            print(json.dumps(result, indent=2))
    else:
        parser.print_help()
