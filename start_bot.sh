#!/bin/bash

# 1. Ensure Ollama is running in the background
if ! pgrep -x "ollama" > /dev/null
then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5 # Give it a moment to initialize
fi

# 2. Ensure your uncensored model is pulled/ready
echo "Verifying local model: dolphin-llama3..."
ollama pull dolphin-llama3

# 3. Set environment overrides to ensure unfiltered mode
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=dolphin-llama3
export MEDIA_SAFETY_MODE=none

# 4. Launch the bot
echo "Launching Kiba Bot in unfiltered mode..."
python3 bot.py