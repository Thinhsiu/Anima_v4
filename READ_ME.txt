structure

Anima_v4/
├── llm/
├── tts/
│   └── voices/
├── stt/
│   └── models/
├── hotword/
├── memory/
 
cd "E:\Anima_v4"

# Create base folders
New-Item -ItemType Directory -Name "llm"
New-Item -ItemType Directory -Name "tts"
New-Item -ItemType Directory -Name "stt"
New-Item -ItemType Directory -Name "hotword"
New-Item -ItemType Directory -Name "memory"

# Create nested folders
New-Item -ItemType Directory -Path "tts\voices"
New-Item -ItemType Directory -Path "stt\models"
--------------------------------------------------

cd E:\Anima_v4

# Core script
New-Item -ItemType File -Name "main.py"

# LLM modules
New-Item -ItemType File -Path "llm\openai_llm.py"
New-Item -ItemType File -Path "llm\local_llm.py"

# TTS module
New-Item -ItemType File -Path "tts\synthesize.py"

# STT module
New-Item -ItemType File -Path "stt\recognize.py"

# Hotword detection (optional)
New-Item -ItemType File -Path "hotword\detect.py"

# Memory logic
New-Item -ItemType File -Path "memory\recall.py"

# Config and THOTH prompt
New-Item -ItemType File -Name "config.json"
New-Item -ItemType File -Name "thoth.txt"

# Python requirements
New-Item -ItemType File -Name "requirements.txt"
--------------------------------------------------

INSTALL REQUIREMENT

inside:----: cd E:\Anima_v4 :---

run:-------: python -m venv venv :-----

activate it:-----: .\venv\Scripts\Activate :-----------

once activated it should show:----: (venv) PS E:\Anima_v4> :--------------

then install requirement from requirement.txt

---------: pip install -r requirements.txt :-------------

----------------------------------------------------------------------
