#!/usr/bin/env python3
from googlenewsdecoder import new_decoderv1

test_url = "https://news.google.com/rss/articles/CBMiVEFVX3lxTE43VTBSLTEwZVlycV9SYVNDOUEtb2FRekFMMnM4WTlQSHdxcFhZc3NDcWU1ZlFnanlpVE9fbl9hNEhEeU8teHRIZ2RNZjQ3N2kzckY3bg?oc=5"

try:
    decoded = new_decoderv1(test_url, interval=5)
    print(f"Status: {decoded.get('status')}")
    print(f"Decoded URL: {decoded.get('decoded_url')}")
except Exception as e:
    print(f"Error: {e}")
    
    # Try v2
    from googlenewsdecoder import new_decoderv2
    try:
        decoded = new_decoderv2(test_url, interval=5)
        print(f"V2 Status: {decoded.get('status')}")
        print(f"V2 Decoded URL: {decoded.get('decoded_url')}")
    except Exception as e2:
        print(f"V2 Error: {e2}")
