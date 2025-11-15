# wrp/server/runtime/runs/requests/options.py
from __future__ import annotations

from typing import Any

from wrp.server.runtime.conversations.seeding import ConversationSeeding, SeedingRunFilter, default_conversation_seeding, normalize_conversation_seeding


class RunRequestOptions:
    def __init__(
        self,
        *,
        system_session_id: str,
        system_session_name: str | None,
        thread_id: str | None,
        conversation_seeding: ConversationSeeding,
        seeding_run_filter: SeedingRunFilter,
        # flags so WorkflowManager can differentiate "caller omitted" vs "explicit none"
        conversation_seeding_specified: bool,
        seeding_run_filter_specified: bool,
        ignore_thread: bool,
    ):
        self.system_session_id = system_session_id
        self.system_session_name = system_session_name
        self.thread_id = thread_id
        self.conversation_seeding = conversation_seeding
        self.seeding_run_filter = seeding_run_filter
        self.conversation_seeding_specified = conversation_seeding_specified
        self.seeding_run_filter_specified = seeding_run_filter_specified
        self.ignore_thread = ignore_thread


def parse_run_request_options(meta: Any | None) -> RunRequestOptions:
    """
    Extract run options from request metadata.

    Expected keys (library-level; not part of workflow input):
      - wrp.system_session: str
      - wrp.system_session_name: str | None
      - wrp.thread: str | None
      - wrp.conversation_seeding: "none" | {"kind":"window","messages":int}
      - wrp.seeding_run_filter: { include_runs?, since_run_id?, until_run_id?, exclude_runs? }
      - wrp.ignore_thread: bool
    """
    if not meta or not isinstance(meta, dict):
        return RunRequestOptions(
            system_session_id="",
            system_session_name=None,
            thread_id=None,
            conversation_seeding=default_conversation_seeding(),
            seeding_run_filter=SeedingRunFilter(),
            conversation_seeding_specified=False,
            seeding_run_filter_specified=False,
            ignore_thread=False,
        )

    wrp_ns = meta.get("wrp", {})
    system_session_id = wrp_ns.get("system_session", "") or ""
    system_session_name = wrp_ns.get("system_session_name")
    thread_id = wrp_ns.get("thread")

    conversation_seeding_raw = wrp_ns.get("conversation_seeding", None)
    seeding_run_filter_raw = wrp_ns.get("seeding_run_filter", None)
    ignore_thread = bool(wrp_ns.get("ignore_thread", False))

    conversation_seeding = normalize_conversation_seeding(conversation_seeding_raw)
    seeding_run_filter = SeedingRunFilter.model_validate(seeding_run_filter_raw or {})

    # If ignore_thread is true, force "none" for seeding and clear seeding_run_filter.
    if ignore_thread:
        conversation_seeding = default_conversation_seeding()  # ConversationSeedingNone()
        seeding_run_filter = SeedingRunFilter()

    return RunRequestOptions(
        system_session_id=system_session_id,
        system_session_name=system_session_name,
        thread_id=thread_id,
        conversation_seeding=conversation_seeding,
        seeding_run_filter=seeding_run_filter,
        conversation_seeding_specified=(conversation_seeding_raw is not None),
        seeding_run_filter_specified=(seeding_run_filter_raw is not None),
        ignore_thread=ignore_thread,
    )