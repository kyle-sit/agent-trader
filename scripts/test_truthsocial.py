#!/usr/bin/env python3
"""Test Truth Social API access for Trump's posts."""
import urllib.request
import json

# Trump's Truth Social account ID
TRUMP_ID = "107780257626128497"
BASE = "https://truthsocial.com/api/v1"

# Test account lookup
try:
    req = urllib.request.Request(f"{BASE}/accounts/lookup?acct=realDonaldTrump")
    req.add_header("User-Agent", "Mozilla/5.0")
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
    print(f"✅ Account found: @{data.get('username')} (ID: {data.get('id')})")
    print(f"   Followers: {data.get('followers_count')}")
except Exception as e:
    print(f"❌ Account lookup failed: {e}")

# Test fetching posts
try:
    req = urllib.request.Request(f"{BASE}/accounts/{TRUMP_ID}/statuses?limit=3")
    req.add_header("User-Agent", "Mozilla/5.0")
    resp = urllib.request.urlopen(req, timeout=10)
    posts = json.loads(resp.read())
    print(f"\n✅ Got {len(posts)} posts:")
    for p in posts:
        # Strip HTML tags
        import re
        text = re.sub('<[^<]+?>', '', p.get('content', ''))
        print(f"  📝 {text[:120]}")
        print(f"     {p.get('created_at', '')}")
        print()
except Exception as e:
    print(f"❌ Statuses fetch failed: {e}")
