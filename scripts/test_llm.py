#!/usr/bin/env python3
"""Test local Ollama gemma4:31b."""
import sys, json
sys.path.insert(0, '.')
from scripts.llm_router import llm_call

print("Testing local gemma4:31b via Ollama...")
result = llm_call('news_analysis', 'Say hello in one word.', model='qwen')
print(json.dumps(result, indent=2, default=str))
