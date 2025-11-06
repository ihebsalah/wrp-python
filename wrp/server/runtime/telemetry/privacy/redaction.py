# wrp/server/runtime/telemetry/privacy/redaction.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple, Pattern
from copy import deepcopy
from ..payloads.types import SpanPayloadEnvelope  # type: ignore[unused-import]
from weakref import WeakKeyDictionary
from .policy import TelemetryResourcePolicy, RedactRules

# --- scrub regex library ---
_PATTERNS: Dict[str, Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "url_creds": re.compile(r"https?://[^/\s:@]+:[^@\s/]+@"),
    "api_key_like": re.compile(r"\b(sk|rk|ak)[_-][A-Za-z0-9]{8,}\b", re.IGNORECASE),
    # a few more pragmatic defaults
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    "bearer": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-\._~\+\/]+=*\b"),
    # Catch most common PEM/ASCII-armored private key blocks:
    # - PKCS#8: (ENCRYPTED )?PRIVATE KEY
    # - PKCS#1 RSA/EC/DSA: RSA/EC/DSA PRIVATE KEY
    # - OpenSSH: OPENSSH PRIVATE KEY
    # - PGP: PGP PRIVATE KEY BLOCK
    "pem_private_key": re.compile(
        r"-----BEGIN (?:ENCRYPTED )?(?:PRIVATE KEY|RSA PRIVATE KEY|EC PRIVATE KEY|DSA PRIVATE KEY)"
        r"-----[\s\S]+?-----END (?:ENCRYPTED )?(?:PRIVATE KEY|RSA PRIVATE KEY|EC PRIVATE KEY|DSA PRIVATE KEY)-----"
        r"|-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]+?-----END OPENSSH PRIVATE KEY-----"
        r"|-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]+?-----END PGP PRIVATE KEY BLOCK-----"
    ),
}

# A cache for compiled custom regex patterns, keyed by the policy object itself.
# Using WeakKeyDictionary to avoid memory leaks if policy objects are discarded.
_COMPILED_CUSTOM_CACHE: "WeakKeyDictionary[TelemetryResourcePolicy, Dict[str, Pattern[str]]]" = WeakKeyDictionary()


def _compile_custom_patterns(custom: Dict[str, str]) -> Dict[str, Pattern[str]]:
    """Compile a dictionary of custom regex patterns, ignoring invalid ones."""
    compiled: Dict[str, Pattern[str]] = {}
    for name, rx in custom.items():
        try:
            compiled[name] = re.compile(rx)
        except re.error:
            # ignore bad patterns to avoid breaking serving path
            continue
    return compiled


def scrub_string(s: str, names: List[str], registry: Dict[str, Pattern[str]]) -> str:
    """Scrub a string using a list of named regex patterns from a registry."""
    out = s
    for n in names:
        pat = registry.get(n)
        if not pat:
            continue
        out = pat.sub(lambda m: "***", out)
    return out


# --- JSONPath-lite helpers (dot paths, "*" for any, "**" for any-depth under a node) ---


def _match_key(seg: str, key: str) -> bool:
    """Check if a path segment matches a given key."""
    return seg == key or seg == "*" or seg == "**"


def _apply_path(obj: Any, path: List[str], fn):
    """
    Apply fn(node, key) where node[key] is target for last path segment.
    Supports lists (numeric or "*") and dicts; "**" descends all dict children recursively.
    """
    if not path:
        return
    seg, *rest = path

    if seg == "**":
        # descend all dict/list children and also attempt rest at this level
        _apply_path(obj, rest, fn)
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                _apply_path(v, ["**"] + rest, fn)
        elif isinstance(obj, list):
            for v in list(obj):
                _apply_path(v, ["**"] + rest, fn)
        return

    if isinstance(obj, dict):
        if not rest:
            # leaf node
            if seg in obj or seg in ("*",):
                for k in list(obj.keys()):
                    if _match_key(seg, k):
                        fn(obj, k)
            return
        # mid-path
        for k, v in list(obj.items()):
            if _match_key(seg, k):
                _apply_path(v, rest, fn)
    elif isinstance(obj, list):
        # arrays: seg may be "*" or an index
        if not rest:
            # leaf node
            if seg == "*":
                for i in range(len(obj)):
                    fn(obj, i)
            else:
                try:
                    idx = int(seg)
                    if 0 <= idx < len(obj):
                        fn(obj, idx)
                except ValueError:
                    # treat as wildcard if not int
                    for i in range(len(obj)):
                        fn(obj, i)
            return
        # mid-path
        if seg == "*":
            for v in obj:
                _apply_path(v, rest, fn)
        else:
            try:
                idx = int(seg)
                if 0 <= idx < len(obj):
                    _apply_path(obj[idx], rest, fn)
            except ValueError:
                # treat as wildcard if not int
                for v in obj:
                    _apply_path(v, rest, fn)


def _drop_at(obj: Any, key):
    """Remove a key from a dict or nullify an element in a list."""
    if isinstance(obj, dict):
        obj.pop(key, None)
    elif isinstance(obj, list) and isinstance(key, int):
        if 0 <= key < len(obj):
            obj[key] = None  # preserve length


def _mask_at(obj: Any, key):
    """Mask a value in a dict or list with '***'."""
    if isinstance(obj, dict):
        obj[key] = "***"
    elif isinstance(obj, list) and isinstance(key, int):
        obj[key] = "***"


def _scrub_all_strings(obj: Any, names: List[str], registry: Dict[str, Pattern[str]]) -> Any:
    """Recursively scrub all string values in a nested object."""
    if isinstance(obj, str):
        return scrub_string(obj, names, registry)
    if isinstance(obj, dict):
        return {k: _scrub_all_strings(v, names, registry) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_all_strings(v, names, registry) for v in obj]
    return obj


def apply_rules(data: Any, rules: RedactRules, *, registry: Dict[str, Pattern[str]]) -> Tuple[Any, bool]:
    """Apply a set of redaction rules to data. Return (masked_data, did_redact)."""
    did = False
    out = deepcopy(data)
    for p in rules.drop:
        changed = False

        def _mark_drop(node, key):
            nonlocal changed
            _drop_at(node, key)
            changed = True

        _apply_path(out, p.split("."), _mark_drop)
        did = did or changed
    for p in rules.mask:
        changed = False

        def _mark_mask(node, key):
            nonlocal changed
            _mask_at(node, key)
            changed = True

        _apply_path(out, p.split("."), _mark_mask)
        did = did or changed
    if rules.scrub:
        out2 = _scrub_all_strings(out, rules.scrub, registry)
        if out2 != out:
            did = True
        out = out2
    if rules.hide_usage and isinstance(out, dict):
        for k in ("usage", "usage_total", "usage_llms"):
            if k in out and out[k] is not None:
                out[k] = None
                did = True
    return out, did


def sanitize_envelope_dict(env: Dict[str, Any], policy: TelemetryResourcePolicy) -> Dict[str, Any]:
    """Return a client-facing sanitized copy of a SpanPayloadEnvelope dict."""
    env = deepcopy(env)
    cap = env.get("capture", {})
    new_cap: Dict[str, Any] = {}
    any_redacted = False
    envelope_span_kind = env.get("span_kind")  # e.g. "run" | "agent" | "llm" | "tool"
    # merge built-ins with author custom patterns
    registry = dict(_PATTERNS)
    if policy.custom_scrub_patterns:
        compiled = _COMPILED_CUSTOM_CACHE.get(policy)
        if compiled is None:
            compiled = _compile_custom_patterns(policy.custom_scrub_patterns)
            _COMPILED_CUSTOM_CACHE[policy] = compiled
        registry.update(compiled)

    for part_name in ("start", "end", "point"):
        part = cap.get(part_name)
        if not part:
            continue
        data = part.get("data")
        if not isinstance(data, dict):
            new_cap[part_name] = part
            continue
        kind = data.get("kind")  # preferred when present in payload
        if not isinstance(kind, str) and isinstance(envelope_span_kind, str):
            # Fallback to envelope-derived kind if payload omitted it
            kind = f"{envelope_span_kind}.{part_name}"
        if not isinstance(kind, str):
            new_cap[part_name] = part
            continue
        mode = policy.mode_for_kind(kind)

        # Extra masks when the author toggles `mask_model_in_spans`.
        # Applies to payload kinds that can carry model/model_settings.
        # - llm.start: has `model` and `model_settings`
        # - agent.start: only `model_settings` (model name lives on the span event)
        # - handoff: contains snapshots of agents with model info
        def _apply_model_masks(d: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
            if not policy.mask_model_in_spans:
                return d, False
            # apply to llm/agent starts and also to handoff snapshots
            if kind in ("llm.start", "agent.start"):
                extra = RedactRules(mask=["model", "model_settings", "model_settings.*"])
                return apply_rules(d, extra, registry=registry)
            if kind == "handoff":
                extra = RedactRules(
                    mask=[
                        "from_agent.model",
                        "from_agent.model_settings",
                        "from_agent.model_settings.*",
                        "to_agent.model",
                        "to_agent.model_settings",
                        "to_agent.model_settings.*",
                    ]
                )
                return apply_rules(d, extra, registry=registry)
            return d, False

        if mode == "private":
            # do not include this part at all
            any_redacted = True
            continue
        if mode == "public":
            # optionally apply global_rules even in public mode for basic safety, also honoring mask_usage as a global toggle
            if policy.apply_global_in_public:
                gr = policy.global_rules
                if policy.mask_usage:
                    gr = gr.model_copy(update={"hide_usage": True}) if gr else RedactRules(hide_usage=True)
                if gr:
                    redacted_data, did = apply_rules(data, gr, registry=registry)
                    redacted_data, did2 = _apply_model_masks(redacted_data)
                    any_redacted = any_redacted or did or did2
                    new_cap[part_name] = dict(part, data=redacted_data)
                else:
                    # public + no globals; still honor model mask toggle
                    redacted_data, did3 = _apply_model_masks(data)
                    if did3:
                        any_redacted = True
                        new_cap[part_name] = dict(part, data=redacted_data)
                    else:
                        new_cap[part_name] = part
            else:
                # public + not applying globals; still honor model mask toggle
                redacted_data, did4 = _apply_model_masks(data)
                if did4:
                    any_redacted = True
                    new_cap[part_name] = dict(part, data=redacted_data)
                else:
                    new_cap[part_name] = part
            continue
        # redacted mode (kind rules + optional global rules) — also apply model masks
        rules = policy.rules.get(kind, RedactRules())
        if policy.mask_usage:
            rules = rules.model_copy(update={"hide_usage": True})
        redacted_data, did = apply_rules(data, rules, registry=registry)
        if policy.global_rules:
            redacted_data, did2 = apply_rules(redacted_data, policy.global_rules, registry=registry)
            did = did or did2
        redacted_data, did3 = _apply_model_masks(redacted_data)
        any_redacted = any_redacted or did or did3
        new_cap[part_name] = dict(part, data=redacted_data)

    env["capture"] = new_cap
    if any_redacted:
        env["redacted"] = True
    return env