# wrp/server/runtime/conversations/assembler.py
from __future__ import annotations

from typing import Iterable, Optional, Set, List

from .seeding import (
    ConversationSeeding,
    ConversationSeedingNone,
    ConversationSeedingWindow,
    SeedingRunFilter,
)
from .types import ChannelItem
from wrp.server.runtime.runs.types import RunMeta
from wrp.server.runtime.store.base import Store


async def select_runs(
    store: Store,
    *,
    system_session_id: str,
    workflow_name: str,
    thread_id: str | None,
    seeding_run_filter: SeedingRunFilter,
    exclude_run_id: str | None = None,
) -> list[RunMeta]:
    """Resolve which runs contribute to the seed, ordered by created_at asc."""
    # include_runs wins outright
    if seeding_run_filter.include_runs:
        metas: list[RunMeta] = []
        for rid in seeding_run_filter.include_runs:
            meta = await store.get_run(system_session_id, rid)
            if meta and meta.workflow_name == workflow_name:
                if exclude_run_id is None or meta.run_id != exclude_run_id:
                    metas.append(meta)
        return sorted(metas, key=lambda m: m.created_at)

    if not thread_id:
        return []

    candidates = await store.runs_in_thread(system_session_id, workflow_name, thread_id)

    # since / until
    if seeding_run_filter.since_run_id:
        try:
            idx = next(i for i, m in enumerate(candidates) if m.run_id == seeding_run_filter.since_run_id)
            candidates = candidates[idx + 1 :]  # exclusive
        except StopIteration:
            # If the since_run_id is not found, ignore the filter.
            pass
    if seeding_run_filter.until_run_id:
        try:
            idx = next(i for i, m in enumerate(candidates) if m.run_id == seeding_run_filter.until_run_id)
            candidates = candidates[: idx + 1]  # inclusive
        except StopIteration:
            pass

    # exclude list
    excl = set(seeding_run_filter.exclude_runs or [])
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
    system_session_id: str,
    run_id: str,
    *,
    limit: int,
    channels: Optional[Set[str]],
) -> List[ChannelItem]:
    """
    Load the last `limit` conversation items from a run across the given channels,
    merging per-channel tails and returning the global tail in ascending ts.
    """
    if not channels:
        # no channel filter: get meta to discover channels, then merge
        metas = await store.list_channel_meta(system_session_id, run_id)
        channels = {m.id for m in metas}
    per_ch: List[ChannelItem] = []
    # Iterate through channels in a stable order to ensure deterministic
    # ordering of items that have the exact same timestamp.
    for ch in sorted(channels):
        ch_tail = await store.load_channel_items(system_session_id, run_id, channel=ch, limit=limit)
        per_ch.extend(ch_tail)
    per_ch.sort(key=lambda i: i.ts)
    if len(per_ch) > limit:
        per_ch = per_ch[-limit:]
    return per_ch


async def assemble_seed(
    store: Store,
    system_session_id: str,
    runs: Iterable[RunMeta],
    seeding: ConversationSeeding,
    *,
    channels: Optional[Set[str]] = None,
) -> list[ChannelItem]:
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
        out: list[ChannelItem] = []

        # Pull from the newest runs first, as we only need the tail of the
        # combined conversations.
        # NOTE: `runs` is assumed to be created_at ASC (per select_runs);
        # we reverse it to iterate DESC here.
        runs_desc = list(runs)[::-1]
        for meta in runs_desc:
            if need <= 0:
                break
            # Fast-skip via channel meta: if no messages in the relevant channels, skip
            metas = await store.list_channel_meta(system_session_id, meta.run_id)
            if channels:
                visible = sum((m.itemsCount or 0) for m in metas if m.id in channels)
            else:
                visible = sum((m.itemsCount or 0) for m in metas)
            if visible == 0:
                continue
            # Load only the portion of the run's tail that we might need.
            tail = await _load_tail(store, system_session_id, meta.run_id, limit=need, channels=channels)
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