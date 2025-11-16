# wrp/server/runtime/system_events.py
from __future__ import annotations as _annotations

import weakref
from typing import Any, Callable, TYPE_CHECKING

import wrp.types as types
from wrp.server.runtime.conversations.privacy.guards import (
    is_private_only_conversations_selector,
)
from wrp.server.runtime.conversations.privacy.policy import ConversationResourcePolicy

if TYPE_CHECKING:
    from wrp.server.session import ServerSession


class SystemEventsManager:
    """Runtime manager for system event subscriptions and fan-out."""

    def __init__(
        self,
        *,
        get_session: Callable[[], "ServerSession"],
        get_conversation_policy: Callable[[], ConversationResourcePolicy],
    ) -> None:
        # Accessors into the wider runtime (no circular deps)
        self._get_session = get_session
        self._get_conversation_policy = get_conversation_policy

        # System events subscriptions: key -> {seq:int, subs:WeakSet[ServerSession]}
        self._event_subscriptions: dict[str, dict[str, Any]] = {}
        # subscriptionId -> key
        self._event_subscription_index: dict[str, str] = {}

    # -------------------------
    # Key derivation
    # -------------------------
    def _key(
        self,
        topic: types.Topic,
        *,
        runs: types.RunsScope | None = None,
        span: types.SpanScope | None = None,
        channel: types.ChannelScope | None = None,
        session_sel: types.SystemSessionScope | None = None,
    ) -> str:
        if span:  # most specific
            return f"{topic}|ss:{span.system_session_id}|run:{span.run_id}|span:{span.span_id}"
        if channel:
            return f"{topic}|ss:{channel.system_session_id}|run:{channel.run_id}|ch:{channel.channel}"
        if runs:
            return f"{topic}|ss:{runs.system_session_id}|run:{runs.run_id}"
        if session_sel:
            return f"{topic}|ss:{session_sel.system_session_id}"
        return f"{topic}|global"

    # -------------------------
    # Subscription handlers
    # -------------------------
    async def subscribe(self, params: types.SystemEventsSubscribeParams) -> types.SystemEventsSubscribeResult:
        """Register the current session for updates on a topic/scope."""

        # Enforce policy for conversations subscriptions
        policy = self._get_conversation_policy()
        if params.topic == "conversations/channel":
            ch_id = params.channel.channel if params.channel else None
            if is_private_only_conversations_selector(policy, channel=ch_id):
                raise PermissionError("Subscription denied: channel is private under current conversation policy.")
        elif params.topic == "conversations/channels":
            # Aggregate guard: if the entire channels index would be private-only, deny
            if is_private_only_conversations_selector(policy, channel=None):
                raise PermissionError(
                    "Subscription denied: channels index is private under current conversation policy."
                )

        sess = self._get_session()
        key = self._key(
            params.topic,
            runs=params.runs,
            span=params.span,
            channel=params.channel,
            session_sel=params.session,
        )
        bucket = self._event_subscriptions.setdefault(key, {"seq": 0, "subs": weakref.WeakSet()})
        bucket["subs"].add(sess)

        # Assign an id and index it
        sub_id = f"sub_{id(sess)}_{len(self._event_subscription_index) + 1}"
        self._event_subscription_index[sub_id] = key

        # Initial seed
        if params.options and params.options.deliverInitial:
            bucket["seq"] += 1
            await sess.send_system_events_updated(
                topic=params.topic,
                sequence=bucket["seq"],
                change="refetch",
                runs=params.runs,
                span=params.span,
                channel=params.channel,
                session_sel=params.session,
            )

        return types.SystemEventsSubscribeResult(subscriptionId=sub_id)

    async def unsubscribe(self, params: types.SystemEventsUnsubscribeParams) -> None:
        """Remove the current session from a subscription (by id or by topic+scope)."""
        key: str | None = None

        if params.subscriptionId and params.subscriptionId in self._event_subscription_index:
            key = self._event_subscription_index.pop(params.subscriptionId, None)

        if key is None:
            key = (
                self._key(
                    params.topic,
                    runs=params.runs,
                    span=params.span,
                    channel=params.channel,
                    session_sel=params.session,
                )
                if params.topic
                else None
            )

        if key and key in self._event_subscriptions:
            # best-effort: remove current session from bucket
            try:
                sess = self._get_session()
                self._event_subscriptions[key]["subs"].discard(sess)  # type: ignore[index]
            except Exception:
                # Swallow errors: unsubscribe is best-effort cleanup
                pass

    # -------------------------
    # Fan-out
    # -------------------------
    async def emit(
        self,
        *,
        topic: types.Topic,
        change: types.ChangeKind | None,
        runs: types.RunsScope | None = None,
        span: types.SpanScope | None = None,
        channel: types.ChannelScope | None = None,
        session_sel: types.SystemSessionScope | None = None,
    ) -> None:
        """Fan-out an events/updated to matching subscribers."""
        key = self._key(topic, runs=runs, span=span, channel=channel, session_sel=session_sel)
        bucket = self._event_subscriptions.get(key)
        if not bucket:
            return

        bucket["seq"] += 1
        seq = bucket["seq"]
        subs = list(bucket["subs"])  # copy to avoid mutation during iteration

        for s in subs:
            try:
                await s.send_system_events_updated(
                    topic=topic,
                    sequence=seq,
                    change=change,
                    runs=runs,
                    span=span,
                    channel=channel,
                    session_sel=session_sel,
                )
            except Exception:
                # Drop dead sessions best-effort
                try:
                    bucket["subs"].discard(s)
                except Exception:
                    pass
