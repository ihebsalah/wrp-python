# wrp/server/runtime/conversations/assembler.py
from __future__ import annotations

from typing import Iterable, Optional, Set

from .seeding import (
    ConversationSeeding,
    ConversationSeedingNone,
    ConversationSeedingWindow,
    RunFilter,
)
from .types import ConversationItem
from wrp.server.runtime.runs.types import RunMeta
from wrp.server.runtime.store.base import Store


async def select_runs(
    store: Store,
    *,
    workflow_name: str,
    thread_id: str | None,
    run_filter: RunFilter,
    exclude_run_id: str | None = None,
) -> list[RunMeta]:
    """Resolve which runs contribute to the seed, ordered by created_at asc."""
    # include_runs wins outright
    if run_filter.include_runs:
        metas: list[RunMeta] = []
        for rid in run_filter.include_runs:
            meta = await store.get_run(rid)
            if meta and meta.workflow_name == workflow_name:
                if exclude_run_id is None or meta.run_id != exclude_run_id:
                    metas.append(meta)
        return sorted(metas, key=lambda m: m.created_at)

    if not thread_id:
        return []

    candidates = await store.runs_in_thread(workflow_name, thread_id)

    # since / until
    if run_filter.since_run_id:
        try:
            idx = next(i for i, m in enumerate(candidates) if m.run_id == run_filter.since_run_id)
            candidates = candidates[idx + 1 :]  # exclusive
        except StopIteration:
            # If the since_run_id is not found, ignore the filter.
            pass
    if run_filter.until_run_id:
        try:
            idx = next(i for i, m in enumerate(candidates) if m.run_id == run_filter.until_run_id)
            candidates = candidates[: idx + 1]  # inclusive
        except StopIteration:
            pass

    # exclude list
    excl = set(run_filter.exclude_runs or [])
    seen = set()
    out = []
    for m in candidates:
        if (exclude_run_id is not None and m.run_id == exclude_run_id) or m.run_id in excl or m.run_id in seen:
            continue
        seen.add(m.run_id)
        out.append(m)
    return out


async def _load_tail(
    store: Store,
    run_id: str,
    *,
    limit: int,
    channels: Optional[Set[str]],
) -> list[ConversationItem]:
    """
    Load the last `limit` conversation items from a run, with optional channel filtering.

    This helper delegates to the store’s tail-aware method. Stores that don’t
    override it will hit the default fallback in Store (load all, slice tail).
    """
    return await store.load_conversation_tail(run_id, limit=limit, channels=channels)


async def assemble_seed(
    store: Store,
    runs: Iterable[RunMeta],
    seeding: ConversationSeeding,
    *,
    channels: Optional[Set[str]] = None,
) -> list[ConversationItem]:
    """
    Assemble a seed conversation from a set of runs, applying a Conversation seeding strategy.

    For window-based strategies, this uses a tail-aware strategy to efficiently
    load only the most recent messages needed, avoiding loading and sorting
    full conversation histories from all contributing runs.
    """
    # No seeding if the seeding strategy is None.
    if isinstance(seeding, ConversationSeedingNone):
        return []

    if isinstance(seeding, ConversationSeedingWindow):
        need = seeding.messages
        out: list[ConversationItem] = []

        # Pull from the newest runs first, as we only need the tail of the
        # combined conversations.
        # NOTE: `runs` is assumed to be created_at ASC (per select_runs);
        # we reverse it to iterate DESC here.
        runs_desc = list(runs)[::-1]
        for meta in runs_desc:
            if need <= 0:
                break
            # Optional fast-skip: if the store maintains reliable counts, we can
            # avoid loading a run that has no visible messages.
            if store.supports_message_counts():
                if channels:
                    visible = sum(meta.channel_counts.get(ch, 0) for ch in channels)
                else:
                    visible = meta.message_count
                if visible == 0:
                    continue
            # Load only the portion of the run's tail that we might need.
            tail = await _load_tail(store, meta.run_id, limit=need, channels=channels)
            # Collect items from each run; we'll do a final sort on the smaller
            # combined list later to establish correct global order.
            out.extend(tail)
            need -= len(tail)

        # The collected items are from different runs and must be sorted by
        # their absolute timestamp to establish the correct global order.
        out.sort(key=lambda i: i.ts)

        # Finally, bound the result to the exact window size.
        if len(out) > seeding.messages:
            out = out[-seeding.messages :]
        return out

    # For any future seeding strategies that are not explicitly handled, it's safest
    # to return an empty list to avoid unexpected behavior.
    return []