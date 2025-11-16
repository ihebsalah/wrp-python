# wrp/server/runtime/telemetry/integrations/openai_hooks.py
from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Agents SDK
from agents.agent import Agent
from agents.model_settings import ModelSettings
from agents.items import ModelResponse, TResponseInputItem
from agents.lifecycle import RunHooksBase
from agents.run_context import RunContextWrapper

if TYPE_CHECKING:
    from agents.mcp import MCPServer


# ---- helpers -----------------------------------------------------------------

def _safe_model_name(agent: Agent[Any]) -> Optional[str]:
    # Agent.model can be str | Model | None. Only serialize str here.
    return agent.model if isinstance(agent.model, str) else None


def _agent_id_str(agent: Agent[Any]) -> str:
    """
    Readable, process-unique instance id for telemetry.
    - Prefix with a slug of the agent name for easier differentiation.
    - Suffix with the object's process-unique identity (id()) in hex.
    - If the agent exposes an explicit agent_id, prefer that, but still keep the name prefix.
    """
    name = getattr(agent, "name", "agent")
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-") or "agent"
    explicit = getattr(agent, "agent_id", None) or getattr(agent, "id", None)
    if explicit is not None:
        return f"{slug}:{explicit}"
    return f"{slug}:{format(id(agent), 'x')}"


def _tool_descriptors(agent: Agent[Any]) -> List[Dict[str, Any]]:
    # minimal snapshot: name/description/schema
    out: List[Dict[str, Any]] = []
    # We *don’t* await get_all_tools() here: this is called inside on_agent_start
    # and we prefer a quick snapshot from the static list. (MCP tools are captured by the runner span.)
    for t in agent.tools:
        # FunctionTool & hosted tool variants share .name; some have schema
        desc: Dict[str, Any] = {"name": getattr(t, "name", "<tool>")}
        schema = getattr(t, "params_json_schema", None)
        if schema:
            desc["params_json_schema"] = schema
        d = getattr(t, "description", None)
        if d:
            desc["description"] = d
        out.append(desc)
    return out


def _mcp_server_descriptors(servers: List["MCPServer"] | None) -> List[Dict[str, Any]]:
    # Best-effort JSON-safe snapshot; avoid heavy objects.
    out: List[Dict[str, Any]] = []
    for s in servers or []:
        out.append(
            {
                "name": getattr(s, "name", s.__class__.__name__),
                # try a few common fields; harmless if missing
                "transport": getattr(s, "transport", None),
                "uri": getattr(getattr(s, "config", None), "uri", getattr(s, "uri", None)),
            }
        )
    return out


def _guardrail_descriptors(objs: List[Any]) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = []
    for g in objs or []:
        name = None
        if hasattr(g, "get_name"):
            try:
                name = g.get_name()
            except Exception:
                name = None
        res.append({"name": name or g.__class__.__name__})
    return res


def _response_to_jsonable(response: ModelResponse) -> Dict[str, Any]:
    # Convert Pydantic output items to dicts; keep id & minimal metadata
    items: List[Dict[str, Any]] = []
    for it in response.output:
        if hasattr(it, "model_dump"):
            items.append(it.model_dump(exclude_unset=True))
        elif hasattr(it, "dict"):
            items.append(it.dict())
        else:
            # fallback best-effort
            items.append(asdict(it))  # type: ignore
    return {
        "response_id": response.response_id,
        "output": items,
        # Usage is reported in LlmUsage; we keep it out of this blob.
    }


def _maybe_tool_args_and_id(ctx: Any) -> tuple[str | None, str | None]:
    """
    If ctx is a ToolContext, return (call_id, args_raw) from ctx.tool_call.
    Else (e.g., RunContextWrapper), return (None, None).
    """
    tc = getattr(ctx, "tool_call", None)
    if tc is None:
        return (None, None)
    call_id = getattr(tc, "call_id", None) or getattr(tc, "id", None)
    args_raw = getattr(tc, "arguments", None)
    return (call_id, args_raw)


def _model_settings_to_dict(ms: Optional[ModelSettings]) -> Optional[Dict[str, Any]]:
    """
    JSON-safe dict for ModelSettings.
    Note: this is the agent-level config; the *effective* per-call settings (after
    run_config overrides and tool-choice resets) are not available here.
    """
    if ms is None:
        return None
    try:
        return ms.to_json_dict()
    except Exception:
        return None


# ---- snapshot helpers for handoff -------------------------------------------

def _output_type_label(agent: Agent[Any]) -> Optional[str]:
    """
    Mirror the label logic used in on_agent_start: prefer type.__name__,
    fall back to object.name, default "str" if output_type is None/opaque.
    """
    ot = getattr(agent, "output_type", None)
    if isinstance(ot, type):
        return getattr(ot, "__name__", None)
    name = getattr(ot, "name", None)
    if name is not None:
        return name
    return "str"

async def _agent_handoff_snapshot(
    context: RunContextWrapper[Any],
    agent: Agent[Any],
) -> Dict[str, Any]:
    try:
        system_prompt = await agent.get_system_prompt(context)
    except Exception:
        system_prompt = None
    return {
        "agent_id": _agent_id_str(agent),
        "model": _safe_model_name(agent),
        "system_prompt": system_prompt,
        "model_settings": _model_settings_to_dict(agent.model_settings),
        "output_type": _output_type_label(agent),
        "tools": _tool_descriptors(agent),
        "mcp_servers": _mcp_server_descriptors(agent.mcp_servers),
        "mcp_config": dict(agent.mcp_config or {}),
        "input_guardrails": _guardrail_descriptors(agent.input_guardrails),
        "output_guardrails": _guardrail_descriptors(agent.output_guardrails),
    }


# ---- hooks implementation -----------------------------------------------------

class OpenAITelemetryHooks(RunHooksBase[Any, Agent[Any]]):
    """
    Bridges Agents SDK lifecycle → WRP RunTelemetryService.
    We rely on ctx.run.telemetry (RunTelemetryService) being available.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        # span bookkeeping
        self._agent_span_by_key: Dict[int, str] = {}  # id(agent) -> agent span_id
        self._llm_span_by_agent: Dict[int, str] = {}  # id(agent) -> last llm span_id
        # Key by (id(agent), call_id) when available for function tools; fall back to tool_name
        self._tool_span_by_key: Dict[Tuple[int, str], str] = {}

    # -------- Agent --------

    async def on_agent_start(self, context: RunContextWrapper[Any], agent: Agent[Any]) -> None:
        # Resolve a system prompt snapshot if possible
        try:
            system_prompt = await agent.get_system_prompt(context)
        except Exception:
            system_prompt = None
        ms_dict = _model_settings_to_dict(agent.model_settings)

        span_id = await self._ctx.run.telemetry.agent_start(
            agent=agent.name,
            model=_safe_model_name(agent),
            agent_id=_agent_id_str(agent),
            system_prompt=system_prompt,
            model_settings=ms_dict,
            output_type=(getattr(agent.output_type, "__name__", None)
                         if isinstance(agent.output_type, type)
                         else (getattr(agent.output_type, "name", None)
                               if agent.output_type is not None else "str")),
            tools=_tool_descriptors(agent),
            mcp_servers=_mcp_server_descriptors(agent.mcp_servers),
            mcp_config=dict(agent.mcp_config or {}),
            input_guardrails=_guardrail_descriptors(agent.input_guardrails),
            output_guardrails=_guardrail_descriptors(agent.output_guardrails),
        )
        self._agent_span_by_key[id(agent)] = span_id

    async def on_agent_end(self, context: RunContextWrapper[Any], agent: Agent[Any], output: Any) -> None:
        agent_key = id(agent)
        span_id = self._agent_span_by_key.pop(agent_key, None)
        if not span_id:
            return  # nothing to close (defensive)

        await self._ctx.run.telemetry.agent_end(
            span_id=span_id,
            final_output=output,
            error=None,
            agent_id=_agent_id_str(agent),
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent[Any],
        to_agent: Agent[Any],
    ) -> None:
        # Build light headers + full encrypted payload snapshots for both sides
        from_snap = await _agent_handoff_snapshot(context, from_agent)
        to_snap = await _agent_handoff_snapshot(context, to_agent)

        await self._ctx.run.telemetry.handoff(
            from_agent=from_agent.name,
            to_agent=to_agent.name,
            # minimal public headers (service will place in span header)
            from_model=from_snap["model"],
            to_model=to_snap["model"],
            from_agent_id=from_snap["agent_id"],
            to_agent_id=to_snap["agent_id"],
            # full snapshots go into encrypted payload
            from_system_prompt=from_snap["system_prompt"],
            from_model_settings=from_snap["model_settings"],
            from_output_type=from_snap["output_type"],
            from_tools=from_snap["tools"],
            from_mcp_servers=from_snap["mcp_servers"],
            from_mcp_config=from_snap["mcp_config"],
            from_input_guardrails=from_snap["input_guardrails"],
            from_output_guardrails=from_snap["output_guardrails"],
            to_system_prompt=to_snap["system_prompt"],
            to_model_settings=to_snap["model_settings"],
            to_output_type=to_snap["output_type"],
            to_tools=to_snap["tools"],
            to_mcp_servers=to_snap["mcp_servers"],
            to_mcp_config=to_snap["mcp_config"],
            to_input_guardrails=to_snap["input_guardrails"],
            to_output_guardrails=to_snap["output_guardrails"],
        )

    # -------- LLM --------

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        system_prompt: Optional[str],
        input_items: List[TResponseInputItem],
    ) -> None:
        ms_dict = _model_settings_to_dict(agent.model_settings)
        span_id = await self._ctx.run.telemetry.llm_start(
            agent=agent.name,
            model=_safe_model_name(agent),
            agent_id=_agent_id_str(agent),
            model_settings=ms_dict,
            system_prompt=system_prompt,
            input_items=input_items,
        )
        self._llm_span_by_agent[id(agent)] = span_id

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent[Any],
        response: ModelResponse,
    ) -> None:
        span_id = self._llm_span_by_agent.pop(id(agent), None)
        if not span_id:
            return

        u = response.usage
        cached = getattr(u.input_tokens_details, "cached_tokens", None) if u.input_tokens_details else None
        reasoning = getattr(u.output_tokens_details, "reasoning_tokens", None) if u.output_tokens_details else None

        await self._ctx.run.telemetry.llm_end(
            span_id=span_id,
            response=_response_to_jsonable(response),
            error=None,
            provider="openai",
            model=_safe_model_name(agent),
            input_tokens=u.input_tokens or None,
            output_tokens=u.output_tokens or None,
            total_tokens=u.total_tokens or None,
            cache_read_tokens=cached,
            reasoning_tokens=reasoning,
            requests=u.requests or None,
            agent=agent.name,
            agent_id=_agent_id_str(agent),
            # cache_hit/details if you can surface them; else omit
        )

    # -------- Tool --------

    async def on_tool_start(
        self,
        context,  # ToolContext for function tools; RunContextWrapper for hosted tools
        agent: Agent[Any],
        tool,
    ) -> None:
        tool_name = getattr(tool, "name", "<tool>")
        call_id, args_raw = _maybe_tool_args_and_id(context)
        span_id = await self._ctx.run.telemetry.tool_start(
            tool=tool_name,
            agent=agent.name,
            agent_id=_agent_id_str(agent),
            args=args_raw,  # capture raw JSON arguments when we have a ToolContext
        )
        key = (id(agent), call_id or tool_name)
        self._tool_span_by_key[key] = span_id

    async def on_tool_end(
        self,
        context,  # ToolContext for function tools; RunContextWrapper for hosted tools
        agent: Agent[Any],
        tool,
        result: str,
    ) -> None:
        tool_name = getattr(tool, "name", "<tool>")
        call_id, _ = _maybe_tool_args_and_id(context)
        # Prefer call_id; fall back to tool_name (for hosted tools / no call id)
        span_id = None
        for candidate in [(id(agent), call_id or ""), (id(agent), tool_name)]:
            if candidate[1]:
                span_id = self._tool_span_by_key.pop(candidate, None)
                if span_id:
                    break
        if span_id:
            await self._ctx.run.telemetry.tool_end(
                span_id=span_id,
                tool_result=result,
                error=None,
                agent=agent.name,
                agent_id=_agent_id_str(agent),
            )