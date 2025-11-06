# wrp/server/runtime/runs/requests/options.py
from __future__ import annotations

from typing import Any

from wrp.server.runtime.conversations.seeding import ConversationSeeding, RunFilter, default_conversation_seeding, normalize_conversation_seeding


class RunRequestOptions:
    def __init__(
        self,
        *,
        thread_id: str | None,
        conversation_seeding: ConversationSeeding,
        run_filter: RunFilter,
        # flags so WorkflowManager can differentiate "caller omitted" vs "explicit none"
        conversation_seeding_specified: bool,
        run_filter_specified: bool,
        ignore_thread: bool,
    ):
        self.thread_id = thread_id
        self.conversation_seeding = conversation_seeding
        self.run_filter = run_filter
        self.conversation_seeding_specified = conversation_seeding_specified
        self.run_filter_specified = run_filter_specified
        self.ignore_thread = ignore_thread


def parse_run_request_options(meta: Any | None) -> RunRequestOptions:
    """
    Extract run options from request metadata.

    Expected keys (library-level; not part of workflow input):
      - wrp.thread: str | None
      - wrp.conversation_seeding: "none" | {"kind":"window","messages":int}
      - wrp.run_filter: { include_runs?, since_run_id?, until_run_id?, exclude_runs? }
      - wrp.ignore_thread: bool
    """
    if not meta or not isinstance(meta, dict):
        return RunRequestOptions(
            thread_id=None,
            conversation_seeding=default_conversation_seeding(),
            run_filter=RunFilter(),
            conversation_seeding_specified=False,
            run_filter_specified=False,
            ignore_thread=False,
        )

    wrp_ns = meta.get("wrp", {})
    thread_id = wrp_ns.get("thread")

    conversation_seeding_raw = wrp_ns.get("conversation_seeding", None)
    run_filter_raw = wrp_ns.get("run_filter", None)
    ignore_thread = bool(wrp_ns.get("ignore_thread", False))

    conversation_seeding = normalize_conversation_seeding(conversation_seeding_raw)
    run_filter = RunFilter.model_validate(run_filter_raw or {})

    # If ignore_thread is true, force "none" for seeding and clear run_filter.
    if ignore_thread:
        conversation_seeding = default_conversation_seeding()  # ConversationSeedingNone()
        run_filter = RunFilter()

    return RunRequestOptions(
        thread_id=thread_id,
        conversation_seeding=conversation_seeding,
        run_filter=run_filter,
        conversation_seeding_specified=(conversation_seeding_raw is not None),
        run_filter_specified=(run_filter_raw is not None),
        ignore_thread=ignore_thread,
    )