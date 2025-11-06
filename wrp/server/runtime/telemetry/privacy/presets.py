# wrp/server/runtime/telemetry/privacy/presets.py
from __future__ import annotations
from typing import Dict
from .policy import TelemetryResourcePolicy, RedactRules, vis

# Opinionated, author-friendly presets. Call, then tweak with .with_overrides()

def open_default(*, hide_usage: bool = False) -> TelemetryResourcePolicy:
    """
    **Default for OSS**: everything PUBLIC (usage/models visible) but apply global scrubs
    even in public mode (passwords/tokens/jwts/keys masked; creds-in-URLs/email scrubbed).
    """
    return TelemetryResourcePolicy.defaults(mask_usage=hide_usage)

def hosted_private(*, hide_usage: bool = True) -> TelemetryResourcePolicy:
    """
    Like open_default, but make agent/llm/tool *private* (no payload shown to clients).
    """
    return TelemetryResourcePolicy.defaults(mask_usage=hide_usage).set_all_internals_private()

def open_no_redaction() -> TelemetryResourcePolicy:
    """
    Everything PUBLIC, and **no** scrubs/masks at all. For local dev only.
    WARNING: Do not use on a hosted server.
    """
    base = TelemetryResourcePolicy.defaults(mask_usage=False)
    # No global rules and don't apply globals in public mode
    return base.model_copy(update={"global_rules": None, "apply_global_in_public": False})

def strict_redacted(*, hide_usage: bool = True) -> TelemetryResourcePolicy:
    """
    **Strict redaction preset (best-effort, no guarantees).**
    - Applies aggressive global rules when a part is REDACTED
    - You **must** validate against your own payloads; regex scrubs are heuristic.
    - If your data is sensitive, prefer VISIBILITY=`private` instead of trusting scrubs.
    """
    base = TelemetryResourcePolicy.defaults(mask_usage=hide_usage)
    global_rules = RedactRules(
        drop=[
            "**.authorization", "**.auth", "**.cookies", "**.set_cookie",
            "**.proxyAuthorization", "**.x-api-key", "**.x-apiKey",
        ],
        mask=[
            "**.password", "**.secret", "**.token", "**.api_key", "**.jwt", "**.refresh_token",
        ],
        scrub=["email", "url_creds", "api_key_like", "jwt", "bearer", "pem_private_key"],
        hide_usage=hide_usage,
    )
    return base.with_overrides(global_rules=global_rules)

# Handy helper: register org-specific scrubs in one place
def with_org_scrubs(pol: TelemetryResourcePolicy, patterns: Dict[str, str]) -> TelemetryResourcePolicy:
    """
    Add/override scrub regexes by name: {"my_secret": r"..."}; then reference them in rules.scrub
    """
    return pol.with_overrides(custom_scrub_patterns=patterns)