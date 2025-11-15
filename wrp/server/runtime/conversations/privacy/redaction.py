# wrp/server/runtime/conversations/privacy/redaction.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from weakref import WeakKeyDictionary
from copy import deepcopy

from wrp.server.runtime.telemetry.privacy.redaction import (
    apply_rules,
    _PATTERNS as _BASE_PATTERNS,
    _compile_custom_patterns,
)

from ..types import SanitizedChannelItem
from .policy import ConversationResourcePolicy, Visibility


# Compile cache keyed by policy instance (matches telemetry approach)
_COMPILED_CACHE: "WeakKeyDictionary[ConversationResourcePolicy, Dict[str, Any]]" = WeakKeyDictionary()


def _registry_for(policy: ConversationResourcePolicy) -> Dict[str, Any]:
    reg = dict(_BASE_PATTERNS)
    custom = _COMPILED_CACHE.get(policy)
    if custom is None:
        custom = _compile_custom_patterns(policy.custom_scrub_patterns)
        _COMPILED_CACHE[policy] = custom
    reg.update(custom)
    return reg


def _sanitize_payload(
    payload: Dict[str, Any],
    *,
    visibility: Visibility,
    role: Optional[str],
    channel: Optional[str],
    policy: ConversationResourcePolicy,
    registry: Dict[str, Any],
) -> tuple[Dict[str, Any], bool]:
    """
    Return (sanitized_payload, did_redact).
    """
    did = False
    data = deepcopy(payload)

    # overlays for redacted mode
    role_rules = policy.rules_by_role.get(role) if role else None
    channel_rules = policy.rules_by_channel.get(channel) if channel else None

    if visibility == "private":
        # Caller should drop the entire message; function shouldn't be called in this case.
        return {}, True

    if visibility == "public":
        # Optionally scrub in public for safety
        if policy.apply_global_in_public and policy.global_rules:
            data, did1 = apply_rules(data, policy.global_rules, registry=registry)
            did = did or did1
        return data, did

    # redacted
    # Precedence: base < channel < role < global.
    # The most specific rule wins (e.g., role overrides channel, channel overrides base).
    # Global rules are always applied last over everything.
    if policy.rules:
        data, d1 = apply_rules(data, policy.rules, registry=registry)
        did = did or d1
    if channel_rules:
        data, d2 = apply_rules(data, channel_rules, registry=registry)
        did = did or d2
    if role_rules:
        data, d3 = apply_rules(data, role_rules, registry=registry)
        did = did or d3
    if policy.global_rules:
        data, d4 = apply_rules(data, policy.global_rules, registry=registry)
        did = did or d4
    return data, did


def sanitize_conversation_items(
    items: Iterable[Any],
    policy: ConversationResourcePolicy,
) -> List[SanitizedChannelItem]:
    """
    Sanitize a sequence of conversation items into a list of SanitizedChannelItem instances.
    Items are duck-typed and expected to have `payload`, `channel`, and `ts` attributes.
    The SanitizedChannelItem type handles final JSON serialization.

    Private messages are dropped entirely.
    """
    out: List[SanitizedChannelItem] = []
    registry = _registry_for(policy)

    for it in items:
        payload = it.payload or {}
        role = payload.get("role")
        vis = policy.resolve_visibility(channel=it.channel, role=role)

        if vis == "private":
            # drop message entirely
            continue

        sanitized, did = _sanitize_payload(
            payload,
            visibility=vis,
            role=role,
            channel=getattr(it, "channel", None),
            policy=policy,
            registry=registry,
        )

        out.append(
            SanitizedChannelItem(
                payload=sanitized,
                channel=it.channel,
                ts=it.ts,
                redacted=bool(did or vis == "redacted"),
            )
        )

    return out