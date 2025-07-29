"""
Voice Emotion Visualization for Anima AI

Provides real-time visualization of voice emotion data in the console UI.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import threading

# Import the emotion display utilities
try:
    from ui.emotion_display import display_emotion_panel, display_emotion_badge, clear_previous_lines
except ImportError:
    # Fallback implementations if the main UI module is not available
    def display_emotion_panel(emotion_data): return str(emotion_data)
    def display_emotion_badge(emotion, confidence, duration_ms=None): 
        return f"{emotion} ({int(confidence*100)}%)"
    def clear_previous_lines(n=1): pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ui.voice_emotion_viz")

# Global state for the live display
_live_display_active = False
_display_thread = None
_latest_emotion_data = None
_emotion_history = []
_emotion_trend = {}
MAX_HISTORY = 20  # Increased history size

# Display configuration
DISPLAY_CONFIG = {
    "update_interval": 0.3,      # Update frequency in seconds (was 0.5)
    "show_trend_chart": True,    # Whether to show emotion trend chart
    "trend_width": 40,           # Width of the trend chart in characters
    "show_indicators": True,     # Show live indicators (mic active, processing)
    "confidence_threshold": 0.4,  # Minimum confidence to consider emotion valid
}

def start_live_display(voice_bridge=None):
    """
    Start a live display of voice emotions in a separate thread.
    
    Args:
        voice_bridge: Optional voice emotion bridge instance
    
    Returns:
        True if started successfully, False otherwise
    """
    global _live_display_active, _display_thread
    
    if _live_display_active:
        print("Voice emotion live display is already active")
        return False
    
    if not voice_bridge:
        try:
            from emotion.voice_bridge import voice_bridge
        except ImportError:
            print("Voice emotion bridge not available. Cannot start live display.")
            return False
    
    # Start the display thread
    _live_display_active = True
    _display_thread = threading.Thread(
        target=_run_live_display, 
        args=(voice_bridge,),
        daemon=True  # Make it a daemon thread so it stops when main thread stops
    )
    _display_thread.start()
    
    print("🎙️ Voice emotion live display started. Speak to see emotions in real-time.")
    print("(Type 'stop emotion display' to end)")
    return True

def stop_live_display():
    """Stop the live display thread"""
    global _live_display_active
    
    if not _live_display_active:
        print("Voice emotion live display is not active")
        return False
    
    _live_display_active = False
    if _display_thread and _display_thread.is_alive():
        # Wait for thread to terminate
        _display_thread.join(timeout=1.0)
    
    # Clear the last display lines
    clear_previous_lines(3)  # Clear the emotional display area
    print("Voice emotion live display stopped")
    return True

def update_emotion_data(emotion_data):
    """
    Update the latest emotion data for display.
    
    Args:
        emotion_data: New emotion data dictionary
    """
    global _latest_emotion_data, _emotion_history, _emotion_trend
    
    if emotion_data and isinstance(emotion_data, dict):
        _latest_emotion_data = emotion_data
        
        # Extract key emotions
        dominant_emotion = emotion_data.get('dominant_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.0)
        
        # Only track emotions above threshold
        if confidence >= DISPLAY_CONFIG["confidence_threshold"]:
            # Add to history with timestamp
            entry = {
                "timestamp": time.time(),
                "data": emotion_data.copy(),
                "dominant_emotion": dominant_emotion,
                "confidence": confidence
            }
            _emotion_history.append(entry)
            
            # Update trend tracking
            if dominant_emotion in _emotion_trend:
                _emotion_trend[dominant_emotion] += 1
            else:
                _emotion_trend[dominant_emotion] = 1
            
            # Trim history if needed
            if len(_emotion_history) > MAX_HISTORY:
                # Remove oldest entry
                oldest = _emotion_history.pop(0)
                # Update trend counts
                old_emotion = oldest.get("dominant_emotion")
                if old_emotion in _emotion_trend and _emotion_trend[old_emotion] > 0:
                    _emotion_trend[old_emotion] -= 1
        
        return True
    
    return False

def generate_trend_chart():
    """
    Generate a visual chart of emotion trends
    
    Returns:
        String representation of the emotion trend chart
    """
    global _emotion_trend, _emotion_history
    
    if not _emotion_trend or not _emotion_history:
        return "No emotion data available for trend visualization"
    
    # Get emotions sorted by frequency
    emotions = sorted(_emotion_trend.items(), key=lambda x: x[1], reverse=True)
    
    # Color mapping for common emotions
    color_map = {
        "neutral": "\033[37m",   # White
        "happy": "\033[93m",     # Yellow
        "sad": "\033[94m",       # Blue
        "angry": "\033[91m",     # Red
        "fear": "\033[95m",      # Magenta
        "surprise": "\033[96m",  # Cyan
        "disgust": "\033[92m",   # Green
        "calm": "\033[94m"       # Blue
    }
    reset = "\033[0m"
    
    # Build the chart
    chart = "\n=== Emotion Trend Chart ==="
    
    # Calculate total for percentage
    total = sum([count for _, count in emotions])
    if total == 0:
        return chart + "\nNo emotion data available"
    
    # Calculate width for bar chart
    max_width = DISPLAY_CONFIG["trend_width"]
    
    # Generate bars for each emotion
    for emotion, count in emotions:
        if count == 0:
            continue
            
        percentage = (count / total) * 100
        bar_width = int((count / total) * max_width)
        
        # Get color if available, otherwise default
        color = color_map.get(emotion.lower(), "\033[37m")
        
        # Create bar with color
        bar = color + "█" * bar_width + reset
        
        # Add to chart
        chart += f"\n{emotion.ljust(10)}: {bar} {percentage:.1f}%"
    
    # Add time range if available
    if len(_emotion_history) >= 2:
        start_time = time.strftime("%H:%M:%S", time.localtime(_emotion_history[0]["timestamp"]))
        end_time = time.strftime("%H:%M:%S", time.localtime(_emotion_history[-1]["timestamp"]))
        chart += f"\n\nTime range: {start_time} - {end_time} ({len(_emotion_history)} samples)"
    
    return chart


def show_emotion_history():
    """
    Display the emotion history in the console with enhanced visualization.
    
    Returns:
        True if history was displayed, False if no history available
    """
    global _emotion_history
    
    if not _emotion_history:
        print("No voice emotion history available yet.")
        return False
    
    print("\n=== Voice Emotion History ===")
    
    # Show the most recent emotions first (reversed)
    for i, entry in enumerate(reversed(_emotion_history[:10])):
        # Format timestamp as HH:MM:SS
        timestamp = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
        emotion = entry["dominant_emotion"]
        confidence = entry["confidence"]
        
        # Create a badge for this emotion
        badge = display_emotion_badge(emotion, confidence)
        # Use relative time (seconds ago) for better context
        seconds_ago = int(time.time() - entry["timestamp"])
        time_ago = f"{seconds_ago}s ago" if seconds_ago < 60 else f"{seconds_ago//60}m {seconds_ago%60}s ago"
        
        print(f"{timestamp} ({time_ago.rjust(10)}) | {badge}")
    
    # Show trend chart if enabled
    if DISPLAY_CONFIG["show_trend_chart"]:
        print("\n" + generate_trend_chart())
    
    # Print summary stats
    if _emotion_trend:
        dominant = max(_emotion_trend.items(), key=lambda x: x[1])[0]
        print(f"\nDominant emotion: {dominant}")
    
    print("\n==========================\n")
    return True

def _run_live_display(voice_bridge):
    """
    Run the live display thread that continuously updates the emotion visualization.
    
    Args:
        voice_bridge: Voice emotion bridge instance
    """
    global _live_display_active, _latest_emotion_data
    
    update_interval = DISPLAY_CONFIG["update_interval"]  # From config
    last_display_update = 0
    display_counter = 0  # For animation frames
    last_audio_time = time.time()  # Track when we last had audio
    idle_indicators = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"]
    
    # Stats tracking
    processing_times = []
    
    print("\n🎙️ Live voice emotion visualization started - speak to see emotions\n")
    
    while _live_display_active:
        try:
            current_time = time.time()
            has_audio = hasattr(voice_bridge, 'last_audio') and voice_bridge.last_audio
            
            # Display indicator animation if we're in idle mode (no audio for 3+ seconds)
            in_idle_mode = current_time - last_audio_time > 3.0
            
            # Determine which animation frame to show for idle indicator
            animation_frame = idle_indicators[display_counter % len(idle_indicators)]
            display_counter += 1
            
            # Process audio if available
            if has_audio:
                # Update the last audio time
                last_audio_time = current_time
                audio_data = voice_bridge.last_audio
                
                # Only update display at specified interval
                if current_time - last_display_update >= update_interval:
                    # Measure processing time
                    start_process = time.time()
                    
                    # Extract emotion features
                    emotion_features = voice_bridge.extract_emotion_features(audio_data)
                    
                    # Track processing time
                    process_time = time.time() - start_process
                    processing_times.append(process_time)
                    if len(processing_times) > 10:
                        processing_times.pop(0)
                    
                    # Update emotion data
                    update_emotion_data(emotion_features)
                    
                    # Prepare the display
                    clear_previous_lines(5)  # Allow more space for enhanced display
                    
                    # Show the emotion panel
                    if _latest_emotion_data:
                        # Get dominant emotion and confidence
                        emotion = _latest_emotion_data.get('dominant_emotion', 'neutral')
                        confidence = _latest_emotion_data.get('confidence', 0.0)
                        
                        # Status indicators
                        mic_status = "🎙️ Active" 
                        avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0
                        status_line = f"Status: {mic_status} | Processing: {avg_process_time*1000:.1f}ms"
                        
                        # Print status and emotion panel
                        print(f"\n{status_line}")
                        print(display_emotion_panel(_latest_emotion_data))
                        
                        # Show mini trend if we have enough history
                        if len(_emotion_history) >= 3 and DISPLAY_CONFIG["show_trend_chart"]:
                            # Get top 3 emotions
                            top_emotions = sorted(_emotion_trend.items(), key=lambda x: x[1], reverse=True)[:3]
                            if top_emotions:
                                trend_line = "Trend: "
                                for emotion, count in top_emotions:
                                    trend_line += f"{emotion}({count}) "
                                print(trend_line)
                    else:
                        # Show waiting indicator
                        print(f"\nListening for voice emotions... {animation_frame}")
                    
                    last_display_update = current_time
            else:
                # Display idle indicator at a slower rate
                if current_time - last_display_update >= update_interval * 2:
                    clear_previous_lines(2)
                    print(f"\nWaiting for speech... {animation_frame}")
                    last_display_update = current_time
            
            # Sleep to avoid high CPU usage - adjust based on whether we're processing
            sleep_time = 0.05 if has_audio else 0.2
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in live display: {e}")
            # Log the full exception for debugging
            import traceback
            logger.debug(traceback.format_exc())
            # Sleep a bit longer after an error
            time.sleep(1.0)

def show_current_emotion():
    """
    Display the current emotion data in the console with enhanced details.
    
    Returns:
        True if emotion data was displayed, False otherwise
    """
    global _latest_emotion_data, _emotion_trend
    
    if not _latest_emotion_data:
        print("No voice emotion data available yet. Please speak first.")
        return False
    
    print("\n=== Current Voice Emotion ===")
    print(display_emotion_panel(_latest_emotion_data))
    
    # Show additional details
    if 'all_emotions' in _latest_emotion_data:
        print("\nDetected emotions (ranked):")
        # Format as a table with confidence levels
        all_emotions = _latest_emotion_data['all_emotions']
        for emotion, confidence in all_emotions.items():
            # Create a simple bar
            bar_len = int(confidence * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"{emotion.ljust(10)}: [{bar}] {confidence*100:.1f}%")
    
    # Add audio quality indicators if available
    if 'audio_quality' in _latest_emotion_data:
        quality = _latest_emotion_data['audio_quality']
        print(f"\nAudio quality: {quality*100:.1f}% ")
    
    # Show time information
    if 'timestamp' in _latest_emotion_data:
        ts = _latest_emotion_data['timestamp']
        time_str = time.strftime("%H:%M:%S", time.localtime(ts))
        print(f"Captured at: {time_str}")
    
    # Add emotional trend analysis if we have history
    if _emotion_trend:
        print("\n--- Emotion Trend Analysis ---")
        # Find most persistent emotion
        dominant = max(_emotion_trend.items(), key=lambda x: x[1])[0]
        print(f"Dominant emotion: {dominant}")
        
        # Find emotional variety (how many different emotions detected)
        variety = sum(1 for emotion, count in _emotion_trend.items() if count > 1)
        if variety >= 3:
            print("Emotional variety: High")
        elif variety == 2:
            print("Emotional variety: Medium")
        else:
            print("Emotional variety: Low")
    
    print("\n===========================\n")
    return True
