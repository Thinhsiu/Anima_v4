"""
Flask application for Anima Voice Emotion Visualization
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
import logging
import traceback
from datetime import datetime
import random

# Add parent directory to path so we can import from the project
sys.path.append(str(Path(__file__).parent.parent))

# Flask imports
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'anima-voice-emotion-secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("anima-web")

# Import emotion system components
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from emotion.voice_bridge import get_bridge_instance, get_last_voice_emotion
    from ui.voice_emotion_viz import get_emotion_history
    HAS_VOICE_EMOTION = True
    logger.info("Voice emotion components imported successfully")
except ImportError as e:
    logger.error(f"Error importing voice emotion components: {e}")
    traceback.print_exc()
    HAS_VOICE_EMOTION = False

# Global state for voice emotion monitoring
voice_bridge = None
emotion_monitor_active = False
emotion_monitor_thread = None
last_emotion_update = time.time()

# Emotion data storage
emotion_data = {
    "current": {
        "emotion": "neutral",
        "confidence": 0.5,
        "timestamp": datetime.now().isoformat()
    },
    "history": [],
    "distribution": {
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "fearful": 0,
        "disgusted": 0, 
        "neutral": 0,
        "surprised": 0
    }
}

# Connect to the voice emotion system
def connect_voice_emotion():
    """Connect to the Anima voice emotion bridge"""
    global voice_bridge
    
    if not HAS_VOICE_EMOTION:
        logger.warning("Voice emotion components not available")
        return False
    
    try:
        voice_bridge = get_bridge_instance()
        logger.info("Successfully connected to voice emotion bridge")
        return True
    except Exception as e:
        logger.error(f"Error connecting to voice emotion bridge: {e}")
        traceback.print_exc()
        return False

# Update emotion distribution stats
def update_emotion_distribution(emotion_name):
    """Update the emotion distribution counts"""
    if emotion_name in emotion_data["distribution"]:
        emotion_data["distribution"][emotion_name] += 1

# Start monitoring for emotions
def start_emotion_monitoring():
    """Start the emotion monitoring thread"""
    global emotion_monitor_active, emotion_monitor_thread
    
    if emotion_monitor_active:
        logger.info("Emotion monitoring already active")
        return True
        
    # Connect to emotion bridge if needed
    if not voice_bridge and not connect_voice_emotion():
        logger.error("Cannot start monitoring: Failed to connect to voice bridge")
        return False
    
    try:
        # Set monitoring flag
        emotion_monitor_active = True
        
        # Start monitoring thread
        emotion_monitor_thread = threading.Thread(
            target=emotion_monitor_loop,
            daemon=True
        )
        emotion_monitor_thread.start()
        logger.info("Started voice emotion monitoring thread")
        return True
    except Exception as e:
        logger.error(f"Error starting emotion monitoring: {e}")
        traceback.print_exc()
        return False

# Stop monitoring for emotions
def stop_emotion_monitoring():
    """Stop the emotion monitoring thread"""
    global emotion_monitor_active
    
    emotion_monitor_active = False
    logger.info("Stopping emotion monitoring")
    return True

# Main emotion monitoring loop
def emotion_monitor_loop():
    """Background thread that monitors for emotion updates"""
    global emotion_monitor_active, emotion_data, last_emotion_update
    
    logger.info("Emotion monitor thread started")
    
    # Initialize history from existing data if available
    try:
        if HAS_VOICE_EMOTION:
            history = get_emotion_history()
            if history:
                emotion_data["history"] = history[-20:]  # Keep last 20 items
                logger.info(f"Loaded {len(history)} emotion history items")
    except Exception as e:
        logger.error(f"Error getting emotion history: {e}")
        traceback.print_exc()
    
    # Main monitoring loop
    while emotion_monitor_active:
        try:
            if HAS_VOICE_EMOTION and voice_bridge:
                # Get latest emotion data from voice bridge
                current_emotion = get_last_voice_emotion()
                
                if current_emotion and 'emotion' in current_emotion:
                    # Only update if enough time has passed (limit update rate)
                    if time.time() - last_emotion_update > 0.5:  # Update at most twice per second
                        emotion_name = current_emotion.get('emotion', 'neutral')
                        confidence = current_emotion.get('confidence', 0.5)
                        
                        # Create formatted emotion update
                        new_emotion = {
                            "emotion": emotion_name,
                            "confidence": confidence,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Update state
                        emotion_data["current"] = new_emotion
                        emotion_data["history"].append(new_emotion)
                        update_emotion_distribution(emotion_name)
                        
                        # Limit history size
                        if len(emotion_data["history"]) > 100:
                            emotion_data["history"] = emotion_data["history"][-100:]
                            
                        # Emit emotion update via websocket
                        socketio.emit('emotion_update', new_emotion)
                        last_emotion_update = time.time()
            else:
                # If no voice emotion system, simulate data for testing UI
                time.sleep(2)
                emotions = ["happy", "sad", "angry", "fearful", "disgusted", "neutral", "surprised"]
                emotion_name = random.choice(emotions)
                confidence = random.uniform(0.5, 0.95)
                
                new_emotion = {
                    "emotion": emotion_name,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
                
                emotion_data["current"] = new_emotion
                emotion_data["history"].append(new_emotion)
                update_emotion_distribution(emotion_name)
                
                if len(emotion_data["history"]) > 100:
                    emotion_data["history"] = emotion_data["history"][-100:]
                    
                socketio.emit('emotion_update', new_emotion)
                last_emotion_update = time.time()
                
            time.sleep(0.1)  # Small sleep to prevent CPU hogging
                
        except Exception as e:
            logger.error(f"Error in emotion monitor thread: {e}")
            traceback.print_exc()
            time.sleep(1)
    
    logger.info("Emotion monitor thread stopped")

# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Return the current system status"""
    return jsonify({
        'has_voice_emotion': HAS_VOICE_EMOTION,
        'monitoring_active': emotion_monitor_active,
        'bridge_connected': voice_bridge is not None,
        'history_count': len(emotion_data['history'])
    })

@app.route('/api/emotions/history')
def get_emotion_history_api():
    """Return the emotion history"""
    return jsonify(emotion_data['history'])

@app.route('/api/emotions/latest')
def get_latest_emotion():
    """Return the latest emotion data"""
    return jsonify(emotion_data['current'])

@app.route('/api/emotions/distribution')
def get_emotion_distribution_api():
    """Return the emotion distribution data"""
    return jsonify(emotion_data['distribution'])

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {
        'has_voice_emotion': HAS_VOICE_EMOTION,
        'monitoring_active': emotion_monitor_active,
        'bridge_connected': voice_bridge is not None
    })
    # Send initial data to new client
    emit('emotion_update', emotion_data['current'])
    emit('emotion_history', emotion_data['history'][-20:])  # Send last 20 items
    emit('emotion_distribution', emotion_data['distribution'])

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start voice emotion monitoring"""
    success = start_emotion_monitoring()
    emit('monitoring_status', {
        'active': emotion_monitor_active,
        'success': success
    })
    return success

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop voice emotion monitoring"""
    success = stop_emotion_monitoring()
    emit('monitoring_status', {
        'active': emotion_monitor_active,
        'success': success
    })
    return success

# Main entry point
if __name__ == '__main__':
    # Make sure template directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Connect to the voice emotion bridge
    connect_voice_emotion()
    
    # Start monitoring automatically
    if HAS_VOICE_EMOTION:
        start_emotion_monitoring()
        logger.info("Started voice emotion monitoring automatically")
    
    # Start the web server
    port = 5000
    logger.info(f"Starting Anima Voice Emotion Web UI on http://localhost:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
