"""
Configuration file for SFCExtraction project.

This file contains sensitive configuration data that should not be committed to version control.
Copy this file and update the values according to your environment.
"""

# Database Configuration
DATABASE_URL = 'postgresql://username:password@localhost:5432/database_name'

# You can add other configuration settings here as needed:
# TWITCH_CLIENT_ID = 'your_twitch_client_id'
# TWITCH_CLIENT_SECRET = 'your_twitch_client_secret'
# DEFAULT_DOWNLOAD_PATH = '/mnt/d/Projects/twitch' 
# LLM Configuration
# Leave OPENAI_API_KEY empty to use local Ollama
# OPENAI_API_KEY = 'your_openai_api_key_here'

# Ollama Configuration (used when OPENAI_API_KEY is not set)
OLLAMA_MODEL = 'qwen2.5:7b'  # Can be: llama3.2:3b, llama3.2:1b, etc. depends on what you want to use on your local machine
OLLAMA_BASE_URL = 'http://localhost:11434'  # Default Ollama server
