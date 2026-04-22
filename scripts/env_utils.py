#!/usr/bin/env python3
"""Shared environment loading helpers for market-intel scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ENV = PROJECT_ROOT / ".env"
HERMES_ENV = Path.home() / ".hermes" / ".env"


def load_market_intel_env() -> list[Path]:
    """Load env vars from likely locations.

    Priority:
    1. Existing process environment (never overridden)
    2. Project-local ~/market-intel/.env
    3. Shared ~/.hermes/.env
    """
    loaded: list[Path] = []
    for env_path in (PROJECT_ENV, HERMES_ENV):
        if env_path.exists():
            load_dotenv(env_path, override=False)
            loaded.append(env_path)
    return loaded


def configure_alpaca_env(base_url: str = "https://paper-api.alpaca.markets") -> dict:
    """Load env files and normalize Alpaca vars into os.environ."""
    loaded = load_market_intel_env()
    os.environ.setdefault("APCA_API_BASE_URL", base_url)

    key_id = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")

    os.environ["APCA_API_KEY_ID"] = key_id
    os.environ["APCA_API_SECRET_KEY"] = secret

    missing = [
        name for name, value in {
            "APCA_API_KEY_ID": key_id,
            "APCA_API_SECRET_KEY": secret,
        }.items()
        if not value
    ]

    return {
        "loaded_files": [str(p) for p in loaded],
        "missing": missing,
        "base_url": os.environ.get("APCA_API_BASE_URL", base_url),
    }


def warn_missing_credentials(missing: list[str], context: str = "Alpaca") -> None:
    if not missing:
        return
    print(
        f"⚠️  {context} credentials missing: {', '.join(missing)}. "
        f"Checked process env, {PROJECT_ENV}, and {HERMES_ENV}.",
        file=sys.stderr,
    )
