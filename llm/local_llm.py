import subprocess
import re

def sanitize_text_for_console(text):
    """Sanitize text to ensure it can be displayed in Windows console.
    Removes or replaces problematic Unicode characters."""
    if not text:
        return ""
        
    # Replace common emoji and special symbols with text alternatives
    replacements = {
        '🧠': '[brain]',
        '🤖': '[robot]',
        '💬': '[speech]',
        '🎉': '[party]',
        '🚀': '[rocket]',
        # Add more common emoji replacements as needed
    }
    
    # Replace known emoji with text versions
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
        
    # Strip remaining non-ASCII characters that might cause issues
    text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
    # Clean up any double spaces from replacements
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def query_local_llm(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt,
            capture_output=True,
            text=True
        )
        # Sanitize output to fix encoding issues
        sanitized_output = sanitize_text_for_console(result.stdout.strip())
        return sanitized_output
    except Exception as e:
        error_msg = str(e)
        # Sanitize error message too
        clean_error = sanitize_text_for_console(error_msg)
        return f"[Local LLM error: {clean_error}]"
