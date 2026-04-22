#!/usr/bin/env python3
import urllib.request
import json

# Check Ollama
try:
    resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    data = json.loads(resp.read())
    print("OLLAMA models:")
    for m in data.get("models", []):
        print(f"  - {m['name']}")
except Exception as e:
    print(f"Ollama: {e}")

print()

# Check LMStudio
try:
    resp = urllib.request.urlopen("http://localhost:1234/v1/models", timeout=5)
    data = json.loads(resp.read())
    print("LMSTUDIO models:")
    for m in data.get("data", []):
        print(f"  - {m['id']}")
except Exception as e:
    print(f"LMStudio: {e}")
