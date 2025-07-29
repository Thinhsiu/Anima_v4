# Anima Voice System Documentation

## Overview
This guide documents the voice synthesis system in the Anima project, including recent fixes and optimizations.

## Recent Fixes

### 1. PyTorch 2.6 Compatibility Issue
**Problem:** PyTorch 2.6 changed the default value of `weights_only` in `torch.load()` from `False` to `True` for security reasons, resulting in unpickling errors.

**Solution:** Added necessary classes to the safe globals list using PyTorch's `add_safe_globals()` function:
```python
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
```

### 2. Performance Optimizations

#### GPU Acceleration
The system now automatically detects if CUDA is available and uses GPU acceleration if possible:
```python
use_gpu = torch.cuda.is_available()
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=use_gpu)
```

#### Speech Caching
Implemented a caching system to store generated speech files, significantly improving performance for repeated phrases:
```python
# Check if we have this text cached
cache_path = get_cache_path(text)
if use_cache and os.path.exists(cache_path):
    output_path = cache_path
else:
    # Generate speech and save to cache
    # ...
```

## Usage Examples

### Basic Usage
```python
from tts.synthesize import speak

# Simple speech synthesis
speak("Hello world")
```

### Advanced Options
```python
# With caching disabled (always regenerate speech)
speak("This will not be cached", use_cache=False)

# Note: Speed control requires additional libraries like pydub
# speak("This would be faster speech", speed=1.5)
```

## Future Improvements
1. Implement proper speed control using pydub or another audio manipulation library
2. Add voice selection options for multiple character voices
3. Improve error handling for better reliability

## Troubleshooting
If you encounter any issues with the voice system:
1. Check that all required packages are installed (`pip install -r requirements.txt`)
2. Ensure the voice file exists at `tts/voices/anima_voice.wav`
3. If using GPU acceleration, verify CUDA is properly installed
4. For performance issues, check disk space for the cache directory
