# wrp/server/runtime/conversations/privacy/guards.py
from __future__ import annotations

import re

from wrp.server.runtime.store.base import Store
from .policy import ConversationResourcePolicy

# Matches:
#   resource://runs/<run_id>/conversations
#   resource://runs/<run_id>/conversations/<channel>
_CONV_URI_RE = re.compile(
    r"^resource://runs/(?P<run_id>[^/]+)/conversations(?:/(?P<channel>[^/]+))?$"
)


async def is_private_only_conversations_uri(
    uri: str,
    policy: ConversationResourcePolicy,
    store: Store,  # kept for parity/future use; not required for current logic
) -> bool:
    """
    Returns True iff `uri` targets conversations AND, under `policy`, the view would be private-only.

    Strategy (fail-closed but policy-driven, no DB scan):
      - For a specific channel: if *every* role resolution and the (no-role) resolution is "private" → block.
      - For the aggregate: if default/no-channel is "private" AND all configured channel overrides are "private"
        AND all role overrides resolve to "private" → block.
      - If anything resolves to public/redacted → allow subscription.

    We prefer policy-based evaluation to avoid loading run data during subscribe.
    """

    m = _CONV_URI_RE.match(uri)
    if not m:
        return False

    channel = m.group("channel")

    def channel_private_only(ch: str | None) -> bool:
        # If any role override would *not* be private on this channel → not private-only
        for role in policy.visibility_by_role.keys():
            if policy.resolve_visibility(channel=ch, role=role) != "private":
                return False

        # Resolution when no role hint is supplied (channel override or default)
        if policy.resolve_visibility(channel=ch, role=None) != "private":
            return False

        return True

    if channel:
        return channel_private_only(channel)

    # Aggregate conversations:
    # If any *configured* channel override would be non-private, allow.
    for ch in policy.visibility_by_channel.keys():
        if not channel_private_only(ch):
            return False

    # Finally, consider default/no-channel path and role overrides.
    return channel_private_only(None)
