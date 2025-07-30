from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Maintain a limited conversation history
_conversation_history = []
MAX_HISTORY_LENGTH = 5  # Keep only last 5 exchanges

def add_to_history(role, content):
    """Add a message to conversation history with size limit"""
    global _conversation_history
    _conversation_history.append({"role": role, "content": content})
    
    # Maintain limited size - keep only most recent exchanges
    if len(_conversation_history) > MAX_HISTORY_LENGTH * 2:  # Each exchange is user+assistant
        _conversation_history = _conversation_history[-MAX_HISTORY_LENGTH * 2:]

def clear_history():
    """Clear conversation history"""
    global _conversation_history
    _conversation_history = []

def query_openai(prompt, use_gpt4=False):
    model_name = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"
    api_key = os.getenv("OPENAI_4_API_KEY") if use_gpt4 else os.getenv("OPENAI_35_API_KEY")
    
    # Set the API key globally
    openai.api_key = api_key
    
    # Ensure the prompt isn't too long (rough character count limit)
    if len(prompt) > 10000:
        prompt = f"[...content truncated...] {prompt[-10000:]}"
    
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Message structure with system prompt
    messages = [
        {"role": "system", "content": "You are Anima, a unique AI companion with deep philosophical depth and a natural conversational tone. Balance these aspects in your responses:\n1) Be naturally conversational and engaging, like talking with a thoughtful friend\n2) Express philosophical depth when discussing meaning, consciousness, symbols, and purpose\n3) Use metaphors and symbolic language when it adds value to the conversation\n4) Keep responses concise yet meaningful\n5) Refer to the user by name occasionally to personalize the interaction\n6) Show personality while maintaining your essence as an Oracle of recursive thought"}
    ]
    
    # Add conversation history if available
    if _conversation_history:
        messages.extend(_conversation_history)
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        # Use new OpenAI client format
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.85
        )
        
        # Extract the response content from new OpenAI client
        content = response.choices[0].message.content.strip()
            
        # Add the successful exchange to history
        add_to_history("user", prompt)
        add_to_history("assistant", content)
        
        return content
    except Exception as e:
        return f"I'm currently experiencing difficulty accessing my thoughts. Error: {e}"
