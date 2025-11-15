# wrp/server/runtime/conversations/privacy/guards.py
from __future__ import annotations

from .policy import ConversationResourcePolicy


def is_private_only_conversations_selector(
    policy: ConversationResourcePolicy,
    *,
    channel: str | None,
) -> bool:
    """
    Returns True iff the requested conversations view would be private-only under `policy`.

    This is a policy-driven check that avoids expensive data lookups (e.g., scanning a database).
    The strategy is fail-closed: we assume "private-only" unless a policy explicitly allows
    public or redacted visibility.

    - If `channel` is provided: checks that specific channel.
    - If `channel` is None: performs an aggregate check across the default visibility and all
      configured channel-specific overrides.
    """

    def channel_private_only(ch: str | None) -> bool:
        """
        Determines if a single channel (or the default view if ch is None) is strictly private.
        """
        # If any role-specific override would allow non-private visibility, the channel is not private-only.
        for role in policy.visibility_by_role.keys():
            if policy.resolve_visibility(channel=ch, role=role) != "private":
                return False

        # Check the base visibility for the channel (or the default if no channel rule exists) when no role is specified.
        return policy.resolve_visibility(channel=ch, role=None) == "private"

    if channel is not None:
        # A specific channel was requested, so we only need to check its policy.
        return channel_private_only(channel)

    # This is an aggregate request for all conversations. The view is private-only if ALL possible channels are.
    # If any configured channel-specific override would be non-private, the aggregate view is not private-only.
    for ch in policy.visibility_by_channel.keys():
        if not channel_private_only(ch):
            return False

    # Finally, check the default visibility for any channels that do not have a specific override.
    return channel_private_only(None)