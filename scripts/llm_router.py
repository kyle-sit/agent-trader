#!/usr/bin/env python3
"""
Centralized LLM Router — Model routing, per-call toggles, priority tiers.

Every LLM call in the pipeline goes through this module.
Supports: local Ollama, Anthropic Claude, and GPT via Codex OAuth.

Configuration:
  - Each call site has a name (e.g., "news_analysis", "ta_interpretation")
  - Each call site has a tier: "critical" or "secondary"
  - You choose which model handles each tier
  - Each call site can be individually enabled/disabled

Usage:
    from llm_router import llm_call, get_config, update_config

    # Simple call — uses configured model for this call site
    result = llm_call("news_analysis", prompt)

    # Override model for one call
    result = llm_call("ta_interpretation", prompt, model="claude")
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import httpx
from dotenv import load_dotenv
load_dotenv(Path.home() / ".hermes" / ".env")

ANTHROPIC_KEY = os.getenv("ANTHROPIC_TOKEN", "") or os.getenv("ANTHROPIC_API_KEY", "")


def _load_codex_runtime_credentials() -> dict:
    """Load Codex OAuth runtime credentials from Hermes's auth stores.

    Prefer Hermes's own auth store, but tolerate both the legacy provider-shaped
    file and the newer credential_pool format currently on disk.
    """
    hermes_repo = Path.home() / ".hermes" / "hermes-agent"
    if hermes_repo.exists():
        hermes_repo_str = str(hermes_repo)
        if hermes_repo_str not in sys.path:
            sys.path.insert(0, hermes_repo_str)

    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials
        try:
            return resolve_codex_runtime_credentials()
        except Exception:
            pass
    except Exception:
        pass

    auth_path = Path.home() / ".hermes" / "auth.json"
    if auth_path.exists():
        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))

            providers = payload.get("providers", {}) if isinstance(payload, dict) else {}
            provider_entry = providers.get("openai-codex", {}) if isinstance(providers, dict) else {}
            provider_tokens = provider_entry.get("tokens", {}) if isinstance(provider_entry, dict) else {}
            access_token = str(provider_tokens.get("access_token", "") or "").strip()
            if access_token:
                return {
                    "provider": "openai-codex",
                    "base_url": provider_entry.get("base_url", "https://chatgpt.com/backend-api/codex"),
                    "api_key": access_token,
                    "source": "hermes-auth-store.providers",
                    "last_refresh": provider_entry.get("last_refresh"),
                    "auth_mode": provider_entry.get("auth_mode", "chatgpt"),
                }

            pool = payload.get("credential_pool", {}) if isinstance(payload, dict) else {}
            codex_entries = pool.get("openai-codex", []) if isinstance(pool, dict) else []
            if isinstance(codex_entries, list):
                for entry in codex_entries:
                    if not isinstance(entry, dict):
                        continue
                    access_token = str(entry.get("access_token", "") or "").strip()
                    if access_token:
                        return {
                            "provider": "openai-codex",
                            "base_url": entry.get("base_url", "https://chatgpt.com/backend-api/codex"),
                            "api_key": access_token,
                            "source": "hermes-auth-store.credential_pool",
                            "last_refresh": entry.get("last_refresh"),
                            "auth_mode": "chatgpt",
                        }
        except Exception as e:
            raise RuntimeError(f"Failed reading Hermes auth store: {e}")

    codex_auth_path = Path.home() / ".codex" / "auth.json"
    if codex_auth_path.exists():
        try:
            payload = json.loads(codex_auth_path.read_text(encoding="utf-8"))
            tokens = payload.get("tokens", {}) if isinstance(payload, dict) else {}
            access_token = str(tokens.get("access_token", "") or "").strip()
            if access_token:
                return {
                    "provider": "openai-codex",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "api_key": access_token,
                    "source": "codex-cli-auth-store",
                    "last_refresh": payload.get("last_refresh"),
                    "auth_mode": payload.get("auth_mode", "chatgpt"),
                }
        except Exception as e:
            raise RuntimeError(f"Failed reading Codex CLI auth store: {e}")

    raise RuntimeError("No Codex OAuth credentials found in ~/.hermes/auth.json or ~/.codex/auth.json")


def _read_codex_default_model() -> str:
    """Read the user's Codex default model from ~/.codex/config.toml if present."""
    config_path = Path.home() / ".codex" / "config.toml"
    if not config_path.exists():
        return "gpt-5.2-codex"

    try:
        import tomllib
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        model = payload.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    except Exception:
        pass

    return "gpt-5.2-codex"

# ── Config File ─────────────────────────────────────────────

CONFIG_FILE = Path.home() / "market-intel" / "data" / "llm_config.json"

DEFAULT_CONFIG = {
    "disable_fallbacks": False,

    # Model settings
    "models": {
        "qwen": {
            "type": "ollama",
            "url": "http://localhost:11434",
            "model": "gemma4:31b",
            "timeout": 300,
        },
        "claude": {
            "type": "anthropic",
            "model": "claude-opus-4-20250514",
            "fallback_models": ["claude-sonnet-4-20250514"],
            "timeout": 120,
        },
        "gpt": {
            "type": "openai-codex",
            "model": _read_codex_default_model(),
            "timeout": 180,
        },
    },

    # Tier → model routing
    "tier_routing": {
        "critical": "gpt",        # News analysis, trade decisions
        "secondary": "gpt",       # Summaries, commentary, interpretation
    },

    # Per-call-site configuration
    "call_sites": {
        "news_analysis": {
            "enabled": True,
            "tier": "critical",
            "description": "Analyze news headlines for market impact",
        },
        "article_summarization": {
            "enabled": True,
            "tier": "secondary",
            "description": "Summarize full article bodies before batch analysis",
        },
        "ta_interpretation": {
            "enabled": True,
            "tier": "secondary",
            "description": "Interpret TA indicator confluence and chart patterns",
        },
        "correlation_reasoning": {
            "enabled": True,
            "tier": "secondary",
            "description": "Assess news/TA alignment with deeper reasoning",
        },
        "trade_validation": {
            "enabled": True,
            "tier": "critical",
            "description": "Validate trade decision before execution",
        },
        "exit_analysis": {
            "enabled": True,
            "tier": "secondary",
            "description": "Check if news catalyst has been invalidated for exits",
        },
        "risk_commentary": {
            "enabled": True,
            "tier": "secondary",
            "description": "Contextual portfolio risk analysis and rebalance suggestions",
        },
        "learning_synthesis": {
            "enabled": True,
            "tier": "secondary",
            "description": "Synthesize performance stats into strategy adjustments",
        },
    },
}


def load_config() -> dict:
    """Load LLM config, merging with defaults for any missing keys."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            saved = json.load(f)
        # Deep merge with defaults
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        for key in saved:
            if isinstance(saved[key], dict) and key in merged:
                merged[key].update(saved[key])
            else:
                merged[key] = saved[key]
        return merged
    return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(config: dict):
    """Save LLM config."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_config() -> dict:
    return load_config()


def update_config(**kwargs):
    """Update specific config values. Examples:
    update_config(tier_routing={"critical": "qwen", "secondary": "qwen"})
    update_config(call_sites={"ta_interpretation": {"enabled": False}})
    """
    config = load_config()
    for key, value in kwargs.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key].update(value)
        else:
            config[key] = value
    save_config(config)
    return config


# ── LLM Backends ────────────────────────────────────────────

def _call_ollama(prompt: str, model_config: dict) -> str:
    """Call local Ollama."""
    url = model_config.get("url", "http://localhost:11434")
    model = model_config.get("model", "qwen3.5:27b")
    timeout = model_config.get("timeout", 300)

    req_data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{url}/api/chat",
        data=req_data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    result = json.loads(resp.read())
    return result.get("message", {}).get("content", "")


def _call_anthropic(prompt: str, model_config: dict) -> str:
    """Call Anthropic Claude."""
    if not ANTHROPIC_KEY:
        raise ValueError("No Anthropic API key")

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    models = [model_config.get("model", "claude-sonnet-4-20250514")]
    models.extend(model_config.get("fallback_models", []))

    for model in models:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                time.sleep(2)
                continue
            elif "not_found" in str(e).lower() or "404" in str(e):
                continue
            raise

    raise RuntimeError("All Anthropic models failed")


def _extract_responses_text(payload: dict) -> str:
    """Extract assistant text from an OpenAI Responses-style payload."""
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if part.get("type") in ("output_text", "text"):
                text = part.get("text", "")
                if text:
                    text_parts.append(text)

    if text_parts:
        return "".join(text_parts)

    raise RuntimeError("Codex response did not contain output text")


def _call_openai_codex(prompt: str, model_config: dict) -> str:
    """Call GPT via Hermes's Codex OAuth runtime credentials."""
    creds = _load_codex_runtime_credentials()
    model = model_config.get("model") or _read_codex_default_model()
    timeout = model_config.get("timeout", 180)

    headers = {
        "Authorization": f"Bearer {creds['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "instructions": "You are a market intelligence analysis assistant. Follow the user's instructions exactly and respond plainly.",
        "input": [{"role": "user", "content": prompt}],
        "store": False,
        "stream": True,
    }

    text_parts = []
    completed_payload = None
    with httpx.stream(
        "POST",
        f"{creds['base_url'].rstrip('/')}/responses",
        headers=headers,
        json=payload,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            if not data.strip() or data.strip() == "[DONE]":
                continue
            event = json.loads(data)
            event_type = event.get("type", "")
            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    text_parts.append(delta)
            elif event_type == "response.output_text.done":
                text = event.get("text", "")
                if text and not text_parts:
                    text_parts.append(text)
            elif event_type == "response.completed":
                completed_payload = event.get("response", {})

    if text_parts:
        return "".join(text_parts)
    if completed_payload:
        return _extract_responses_text(completed_payload)
    raise RuntimeError("Codex streamed response did not contain output text")


def _get_fallback_candidates(primary: str, config: dict) -> list[str]:
    """Return ordered fallback model names after a primary backend fails."""
    if config.get("disable_fallbacks", False):
        return []

    preference = {
        "gpt": ["qwen", "claude"],
        "claude": ["gpt", "qwen"],
        "qwen": ["gpt", "claude"],
    }
    candidates = preference.get(primary, [])
    return [name for name in candidates if name in config.get("models", {})]


def _dispatch_model_call(prompt: str, model: str, model_config: dict) -> str:
    """Route a prompt to the configured backend."""
    model_type = model_config.get("type", "ollama")
    if model_type == "ollama":
        return _call_ollama(prompt, model_config)
    if model_type == "anthropic":
        return _call_anthropic(prompt, model_config)
    if model_type == "openai-codex":
        return _call_openai_codex(prompt, model_config)
    raise RuntimeError(f"Unknown model type: {model_type}")


# ── Main Call Interface ─────────────────────────────────────

def llm_call(call_site: str, prompt: str, model: str = None) -> dict:
    """
    Make an LLM call through the router.

    Args:
        call_site: Name of the call site (e.g., "news_analysis")
        prompt: The prompt to send
        model: Override model ("qwen" or "claude"). If None, uses tier routing.

    Returns:
        {"success": bool, "text": str, "model_used": str, "call_site": str}
    """
    config = load_config()

    # Check if call site is enabled
    site_config = config.get("call_sites", {}).get(call_site, {})
    if not site_config.get("enabled", True):
        return {
            "success": False,
            "text": "",
            "model_used": "disabled",
            "call_site": call_site,
            "reason": f"Call site '{call_site}' is disabled",
        }

    # Determine which model to use
    if model is None:
        tier = site_config.get("tier", "secondary")
        model = config.get("tier_routing", {}).get(tier, "qwen")

    model_config = config.get("models", {}).get(model, {})

    # Try primary model, then ordered fallbacks if it fails
    try:
        text = _dispatch_model_call(prompt, model, model_config)

        return {
            "success": True,
            "text": text,
            "model_used": model,
            "call_site": call_site,
        }

    except Exception as primary_error:
        fallback_errors = []
        for fallback in _get_fallback_candidates(model, config):
            fallback_config = config.get("models", {}).get(fallback, {})
            try:
                text = _dispatch_model_call(prompt, fallback, fallback_config)
                return {
                    "success": True,
                    "text": text,
                    "model_used": f"{fallback} (fallback from {model})",
                    "call_site": call_site,
                }
            except Exception as fallback_error:
                fallback_errors.append(f"{fallback}: {fallback_error}")

        fallback_msg = "; ".join(fallback_errors) if fallback_errors else "No fallback models configured"
        reason = f"Primary ({model}): {primary_error}"
        if not config.get("disable_fallbacks", False):
            reason = f"{reason}; Fallbacks: {fallback_msg}"
        return {
            "success": False,
            "text": "",
            "model_used": "none",
            "call_site": call_site,
            "reason": reason,
        }


def extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response text."""
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


# ── CLI ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Router — Model routing and config")
    parser.add_argument("--config", action="store_true", help="Show current config")
    parser.add_argument("--set-critical", type=str, choices=["qwen", "claude", "gpt"], help="Set model for critical tier")
    parser.add_argument("--set-secondary", type=str, choices=["qwen", "claude", "gpt"], help="Set model for secondary tier")
    parser.add_argument("--set-all", type=str, choices=["qwen", "claude", "gpt"], help="Set model for all tiers")
    parser.add_argument("--enable", type=str, help="Enable a call site")
    parser.add_argument("--disable", type=str, help="Disable a call site")
    parser.add_argument("--enable-all", action="store_true", help="Enable all call sites")
    parser.add_argument("--disable-all", action="store_true", help="Disable all call sites")
    parser.add_argument("--test", type=str, choices=["qwen", "claude", "gpt"], help="Test a model connection")
    args = parser.parse_args()

    if args.config:
        config = load_config()
        print(f"\n{'=' * 60}")
        print(f"  LLM ROUTER CONFIG")
        print(f"{'=' * 60}")
        print(f"\n  Tier Routing:")
        for tier, model in config.get("tier_routing", {}).items():
            print(f"    {tier:12s} → {model}")
        print(f"\n  Call Sites:")
        for name, site in config.get("call_sites", {}).items():
            status = "✅" if site.get("enabled") else "❌"
            tier = site.get("tier", "?")
            model = config.get("tier_routing", {}).get(tier, "?")
            print(f"    {status} {name:25s} [{tier:10s}] → {model:8s} | {site.get('description', '')}")
        print(f"\n  Models:")
        for name, model in config.get("models", {}).items():
            print(f"    {name:8s}: {model.get('type')} / {model.get('model', model.get('url', ''))}")
        return

    if args.set_critical:
        update_config(tier_routing={"critical": args.set_critical})
        print(f"✅ Critical tier → {args.set_critical}")

    if args.set_secondary:
        update_config(tier_routing={"secondary": args.set_secondary})
        print(f"✅ Secondary tier → {args.set_secondary}")

    if args.set_all:
        update_config(tier_routing={"critical": args.set_all, "secondary": args.set_all})
        print(f"✅ All tiers → {args.set_all}")

    if args.enable:
        config = load_config()
        if args.enable in config["call_sites"]:
            config["call_sites"][args.enable]["enabled"] = True
            save_config(config)
            print(f"✅ Enabled: {args.enable}")

    if args.disable:
        config = load_config()
        if args.disable in config["call_sites"]:
            config["call_sites"][args.disable]["enabled"] = False
            save_config(config)
            print(f"❌ Disabled: {args.disable}")

    if args.enable_all:
        config = load_config()
        for site in config["call_sites"]:
            config["call_sites"][site]["enabled"] = True
        save_config(config)
        print("✅ All call sites enabled")

    if args.disable_all:
        config = load_config()
        for site in config["call_sites"]:
            config["call_sites"][site]["enabled"] = False
        save_config(config)
        print("❌ All call sites disabled")

    if args.test:
        print(f"Testing {args.test}...")
        result = llm_call("test", "Reply with exactly: OK", model=args.test)
        if result["success"]:
            print(f"✅ {args.test}: {result['text'][:50]}")
        else:
            print(f"❌ {args.test}: {result.get('reason', 'Failed')}")


if __name__ == "__main__":
    main()
