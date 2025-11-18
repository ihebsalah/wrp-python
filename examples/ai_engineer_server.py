# examples/ai_engineer_server.py
from __future__ import annotations

from typing import Literal, Optional

# --- Pydantic ---
from pydantic import BaseModel, Field

# --- OpenAI Agent SDK ---
from agents import Agent, ModelSettings, Runner, TResponseInputItem
from openai.types.shared.reasoning import Reasoning

# --- WRP imports ---
from wrp.server.runtime.settings.agents.settings import AgentSettings
from wrp.server.runtime.conversations.privacy import ConversationResourcePolicy
from wrp.server.runtime.conversations.seeding import (
    ConversationSeedingWindow,
    WorkflowConversationSeeding,
)
from wrp.server.runtime.settings.providers.settings import ProviderSettings
from wrp.server.runtime.server import WRP, Context
from wrp.server.runtime.store.stores.sqlite_store import SqliteStore
from wrp.server.runtime.telemetry.integrations.openai.openai_hooks import (
    OpenAITelemetryHooks,
)
from wrp.server.runtime.telemetry.privacy.presets import open_default
from wrp.server.runtime.settings.workflows import WorkflowSettings
from wrp.server.runtime.workflows.types import WorkflowInput, WorkflowOutput


# ---------------------------
# Agent output schemas (typed)
# ---------------------------
class DevAgentSchema(BaseModel):
    developer_report: str


class TestAgentSchema(BaseModel):
    test_report: str


# ---------------------------
# Agent settings
# ---------------------------
class DevAgentSettings(AgentSettings):
    provider_name: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = Field(
        default=0.3, ge=0.0, le=2.0, description="Creativity knob. 0=deterministic, 2=random."
    )
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, gt=0, description="Max output tokens.")
    parallel_tool_calls: bool = True
    # optional reasoning knob
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="Effort level for reasoning models.",
    )
    # locked by base; author-defined defaults
    allowed_providers: list[str] | None = ["openai"]
    allowed_models: dict[str, list[str]] | None = {
        "openai": ["gpt-4.1-mini", "gpt-4.1"],
    }


class TestAgentSettings(AgentSettings):
    provider_name: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1600, gt=0)
    parallel_tool_calls: bool = True
    # optional reasoning knob
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    allowed_providers: list[str] | None = ["openai"]
    allowed_models: dict[str, list[str]] | None = {
        "openai": ["gpt-4.1-mini", "gpt-4.1"],
    }


def build_dev_agent(agent_cfg: DevAgentSettings) -> Agent:
    return Agent(
        name="Dev Agent",
        instructions=(
            "You are an AI software engineer. When given a prompt, do the work mentally, "
            "then return a crisp progress REPORT of what you achieved. "
            "When asked to repair code based on testing feedback, modify the files accordingly "
            "and return a REPORT describing exactly what you changed (no code blocks)."
        ),
        model=agent_cfg.model,
        output_type=DevAgentSchema,
        model_settings=ModelSettings(
            temperature=agent_cfg.temperature,
            top_p=agent_cfg.top_p,
            max_tokens=agent_cfg.max_tokens,
            parallel_tool_calls=agent_cfg.parallel_tool_calls,
            reasoning=Reasoning(effort=agent_cfg.reasoning_effort) if agent_cfg.reasoning_effort is not None else None,
        ),
    )


def build_test_agent(agent_cfg: TestAgentSettings) -> Agent:
    return Agent(
        name="Test Agent",
        instructions=(
            "You are a test author. Given a prompt and input, produce a clear TEST REPORT "
            "describing test intent, expected behaviors, and likely failures. "
            "If helpful, you may outline pytest snippets in prose, but your structured output "
            "must be a report."
        ),
        model=agent_cfg.model,
        output_type=TestAgentSchema,
        model_settings=ModelSettings(
            temperature=agent_cfg.temperature,
            top_p=agent_cfg.top_p,
            max_tokens=agent_cfg.max_tokens,
            parallel_tool_calls=agent_cfg.parallel_tool_calls,
            reasoning=Reasoning(effort=agent_cfg.reasoning_effort) if agent_cfg.reasoning_effort is not None else None,
        ),
    )


# ---------------------------
# Server setup
# ---------------------------

# Privacy/serving policy for conversations (sanitization + visibility)
conv_policy = ConversationResourcePolicy.defaults()
# Example overrides (tweak as you like):
conv_policy.visibility_by_channel.update(
    {"debug": "private"}
)  # drop debug from served resources
conv_policy.visibility_by_role.update(
    {"system": "redacted"}
)  # serve system messages in redacted form

DEFAULT_SEEDING = WorkflowConversationSeeding(
    default_seeding=ConversationSeedingWindow(messages=20),
    allowed_channels=None,
)

server = WRP(
    name="AI Engineer WRP",
    instructions="Two workflows using the OpenAI Agent SDK with typed outputs and channel-scoped conversations.",
    store=SqliteStore(path="wrp_data/ai_engineer.sqlite", key=None),
    telemetry_policy=open_default(),
    conversation_policy=conv_policy,  # enable conversation sanitization/visibility
    default_seeding=DEFAULT_SEEDING,
    global_input_limit_bytes=2 * 1024 * 1024,  # 2 MiB cap for all workflows
)

# Register agent settings (providers are auto-registered by the registry)
server.register_agent_settings("dev-agent", DevAgentSettings(), allow_override=True)
server.register_agent_settings("test-agent", TestAgentSettings(), allow_override=True)

# ---------------------------
# Workflows
# ---------------------------

# ---------------------------
# Workflow I/O models
# ---------------------------


class DevIn(WorkflowInput):
    prompt: str = Field(..., description="Development prompt / ticket text")


class DevOut(WorkflowOutput):
    developer_report: str


class TestIn(WorkflowInput):
    prompt: str = Field(..., description="Testing prompt")
    test_input: str = Field(
        ..., description="Subject under test (code path, snippet, API contract, etc.)"
    )
    repair_code: bool = Field(
        default=False,
        description="If true, reuse Dev Agent to apply a fix and return a repair report",
    )


class TestOut(WorkflowOutput):
    test_report: str
    repair_report: Optional[str] = None  # report of changes made by dev agent (no code)


# ---------------------------
# Workflow-level settings (defaults)
# ---------------------------
class DevWorkflowSettings(WorkflowSettings):
    """
    Workflow-level knobs for the dev workflow.

    Intentionally kept free of model settings; those live in AgentSettings now.
    """

    pass


class TestWorkflowSettings(WorkflowSettings):
    """
    Workflow-level knobs for the test workflow.

    Demonstrates behavior-level control that sits above the raw workflow input.
    """

    allow_code_repair: bool = True


# ---------------------------
# Workflow definitions
# ---------------------------


@server.workflow(
    name="dev",
    title="AI Engineer: Development",
    description="Takes a prompt and returns a report of what was achieved.",
    input_model=DevIn,
    output_model=DevOut,
    seeding=DEFAULT_SEEDING,
    input_limit_bytes=1 * 1024 * 1024,  # 1 MiB cap (dev)
    settings_default=DevWorkflowSettings(),  # enable workflow settings with defaults
    settings_allow_override=True,  # allow overrides
)
async def dev_flow(wf_input: DevIn, ctx: Context) -> DevOut:
    # Use only the 'dev' channel (seed + live).
    ch = await ctx.run.conversations.get_channel(
        "dev",
        "Development",
        "Primary channel for development work",
        item_type=TResponseInputItem,
    )
    # Effective workflow settings (no name needed; inferred from current workflow)
    dev_cfg = ctx.get_workflow_settings()  # -> DevWorkflowSettings instance
    dev_agent_cfg = ctx.get_agent_settings("dev-agent")

    # Example: warn if any override has been applied (instance helper infers workflow via ctx)
    if dev_cfg.settings_overridden(ctx=ctx):
        await ctx.run.telemetry.annotation(
            message="dev: non-default workflow settings in effect",
            level="warning",
        )
    dev_agent = build_dev_agent(dev_agent_cfg)

    user_msg: list[TResponseInputItem] = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": wf_input.prompt}],
        }
    ]
    await ch.add_item(user_msg)

    await ctx.run.telemetry.annotation(
        message=f"dev: starting (model={dev_agent_cfg.model}, provider={dev_agent_cfg.provider_name})",
        level="info",
    )

    dev_result_temp = await Runner.run(
        dev_agent,
        input=ch.get_items(),  # typed: list[TResponseInputItem]
        hooks=OpenAITelemetryHooks(ctx),
    )

    # Persist + mirror assistant messages to the same handle
    for item in dev_result_temp.new_items:
        await ch.add_item(item.to_input_item())

    # ---- typed result ----
    dev_result = {
        "output_text": dev_result_temp.final_output.json(),
        "output_parsed": dev_result_temp.final_output.model_dump(),
    }
    developer_report = dev_result["output_parsed"]["developer_report"]

    await ctx.run.telemetry.annotation(message="dev: done", level="info")
    return DevOut(developer_report=developer_report)


@server.workflow(
    name="test",
    title="AI Engineer: Testing",
    description="Takes a testing prompt and input. Optionally repairs code by reusing Dev Agent and dev-channel conversation.",
    input_model=TestIn,
    output_model=TestOut,
    seeding=DEFAULT_SEEDING,
    input_limit_bytes=1 * 1024 * 1024,  # 1 MiB cap (test)
    settings_default=TestWorkflowSettings(),  # enable workflow settings with defaults
    settings_allow_override=False,  # demonstrate all-or-nothing: no overrides permitted
)
async def test_flow(wf_input: TestIn, ctx: Context) -> TestOut:
    # Primary channel for Test Agent uses only 'test' (seed + live).
    test_ch = await ctx.run.conversations.get_channel(
        "test",
        "Testing",
        "Primary channel for test design & results",
        item_type=TResponseInputItem,
    )
    test_cfg = ctx.get_workflow_settings()  # -> TestWorkflowSettings instance
    test_agent_cfg = ctx.get_agent_settings("test-agent")

    # Alternative override detection via manager flag (canonical, cheap):
    if ctx.wrp._workflow_manager.settings_overridden(ctx.run.workflow_name):
        await ctx.run.telemetry.annotation(
            message=(
                "test: non-default workflow settings in effect "
                f"(allow_code_repair={test_cfg.allow_code_repair})"
            ),
            level="warning",
        )

    test_agent = build_test_agent(test_agent_cfg)

    user_test_msg: TResponseInputItem = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": f"TEST PROMPT:\n{wf_input.prompt}"},
            {"type": "input_text", "text": f"TEST INPUT:\n{wf_input.test_input}"},
        ],
    }
    await test_ch.add_item([user_test_msg])  # ensure list[TResponseInputItem]

    await ctx.run.telemetry.annotation(
        message=f"test: starting (model={test_agent_cfg.model}, provider={test_agent_cfg.provider_name})",
        level="info",
        data={"repair_code": wf_input.repair_code},
    )

    test_result_temp = await Runner.run(
        test_agent,
        input=test_ch.get_items(),  # typed: list[TResponseInputItem]
        hooks=OpenAITelemetryHooks(ctx),
    )

    for item in test_result_temp.new_items:
        await test_ch.add_item(item.to_input_item())

    # ---- typed result ----
    test_result = {
        "output_text": test_result_temp.final_output.json(),
        "output_parsed": test_result_temp.final_output.model_dump(),
    }
    test_report = test_result["output_parsed"]["test_report"]

    repair_report: Optional[str] = None

    # Workflow-level control over whether repair is permitted at all.
    repair_requested = wf_input.repair_code
    repair_enabled = bool(test_cfg.allow_code_repair)

    if repair_requested and not repair_enabled:
        await ctx.run.telemetry.annotation(
            message="test: repair requested via input but disabled by workflow settings",
            level="warning",
            data={"repair_code": True, "allow_code_repair": False},
        )

    if repair_requested and repair_enabled:
        # Reuse dev agent with 'dev' channel (seed + live).
        dev_ch = await ctx.run.conversations.get_channel(
            "dev",
            "Development",
            "Primary channel for development work",
            item_type=TResponseInputItem,
        )
        dev_agent_cfg = ctx.get_agent_settings("dev-agent")
        dev_agent = build_dev_agent(dev_agent_cfg)

        repair_prompt = (
            "Apply a minimal, safe fix based on the following testing feedback. "
            "Modify the repository files as needed, then return ONLY a concise REPORT of changes made "
            "(no code blocks).\n\n"
            f"=== TEST REPORT ===\n{test_report}\n\n"
            f"=== SUBJECT UNDER TEST ===\n{wf_input.test_input}\n"
        )
        repair_user_msg: list[TResponseInputItem] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": repair_prompt}],
            }
        ]
        await dev_ch.add_item(repair_user_msg)

        await ctx.run.telemetry.annotation(
            message=(
                "test: repairing with dev agent "
                f"(model={dev_agent_cfg.model}, provider={dev_agent_cfg.provider_name})"
            ),
            level="info",
        )

        repair_result_temp = await Runner.run(
            dev_agent,
            input=dev_ch.get_items(),  # typed: list[TResponseInputItem]
            hooks=OpenAITelemetryHooks(ctx),
        )

        for item in repair_result_temp.new_items:
            await dev_ch.add_item(item.to_input_item())

        # ---- typed result (dev agent again) ----
        repair_result = {
            "output_text": repair_result_temp.final_output.json(),
            "output_parsed": repair_result_temp.final_output.model_dump(),
        }
        repair_report = repair_result["output_parsed"]["developer_report"]

    await ctx.run.telemetry.annotation(
        message="test: done",
        level="info",
        data={"repaired": bool(repair_report)},
    )
    return TestOut(test_report=test_report, repair_report=repair_report)


# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    # Run with stdio
    # For HTTP/SSE, use: server.run("streamable-http") or server.run("sse")
    server.run("stdio")