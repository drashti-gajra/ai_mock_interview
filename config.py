import os

# Ollama local server (Mistral)
LLM_MODEL = "mistral"
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_API_KEY = "ollama"  # Ollama doesn't need a real key but the SDK requires one
TEMPERATURE = 0.7
MAX_TOKENS = 8192
