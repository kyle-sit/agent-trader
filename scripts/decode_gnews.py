#!/usr/bin/env python3
"""Decode Google News article URLs to find the real article URL."""
import base64
import re

url_encoded = 'CBMiVEFVX3lxTE43VTBSLTEwZVlycV9SYVNDOUEtb2FRekFMMnM4WTlQSHdxcFhZc3NDcWU1ZlFnanlpVE9fbl9hNEhEeU8teHRIZ2RNZjQ3N2kzckY3bg'

# Add padding
padded = url_encoded + '=' * (4 - len(url_encoded) % 4)

decoded = base64.urlsafe_b64decode(padded)
print('Raw bytes:', repr(decoded))
print()

# Try to find URL pattern in decoded bytes
text = decoded.decode('utf-8', errors='ignore')
print('UTF-8:', text)
print()

# Look for http patterns
urls = re.findall(r'https?://[^\s\x00-\x1f]+', text)
print('Found URLs:', urls)

# Also try: the part after first few bytes might be the URL
# Google News encodes with a protobuf-like format
# Try skipping first few bytes
for skip in range(1, 10):
    try:
        t = decoded[skip:].decode('utf-8', errors='strict')
        if 'http' in t or '.com' in t or 'www.' in t:
            print(f'Skip {skip}: {t}')
    except:
        pass
