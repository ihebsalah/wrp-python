# examples/ai_engineer_server.py
from __future__ import annotations

from typing import Optional

# --- Pydantic ---
from pydantic import BaseModel, Field

# --- OpenAI Agent SDK ---
from agents import (
    Agent,
    ModelSettings,
    Runner,
)

# --- WRP imports ---
from wrp.server.runtime.server import WRP, Context
from wrp.server.runtime.telemetry.privacy.presets import open_default
from wrp.server.runtime.store.stores.sqlite_store import SqliteStore
from wrp.server.runtime.conversations.seeding import WorkflowConversationSeeding, ConversationSeedingWindow
from wrp.server.runtime.conversations.privacy import ConversationResourcePolicy
from wrp.server.runtime.workflows.settings import WorkflowSettings
from wrp.server.runtime.workflows.types import WorkflowInput, WorkflowOutput
from wrp.server.runtime.telemetry.integrations.openai.openai_hooks import OpenAITelemetryHooks


# ---------------------------
# Agent output schemas (typed)
# ---------------------------

class DevAgentSchema(BaseModel):
    developer_report: str


class TestAgentSchema(BaseModel):
    test_report: str


# ---------------------------
# Workflow-level settings (defaults)
# ---------------------------
class DevSettings(WorkflowSettings):
    model: str = "gpt-4.1-mini"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 2048
    parallel_tool_calls: bool = True
    store: bool = True
    # Demonstrate fine-grained lock: cannot change the model via overrides
    locked = {"model"}


class TestSettings(WorkflowSettings):
    model: str = "gpt-4.1-mini"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 1600
    parallel_tool_calls: bool = True
    store: bool = True


def build_dev_agent(cfg: DevSettings) -> Agent:
    return Agent(
        name="Dev Agent",
        instructions=(
            "You are an AI software engineer. When given a prompt, do the work mentally, "
            "then return a crisp progress REPORT of what you achieved. "
            "When asked to repair code based on testing feedback, modify the files accordingly "
            "and return a REPORT describing exactly what you changed (no code blocks)."
        ),
        model=cfg.model,
        output_type=DevAgentSchema,
        model_settings=ModelSettings(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            parallel_tool_calls=cfg.parallel_tool_calls,
            store=cfg.store,
        ),
    )


def build_test_agent(cfg: TestSettings) -> Agent:
    return Agent(
        name="Test Agent",
        instructions=(
            "You are a test author. Given a prompt and input, produce a clear TEST REPORT "
            "describing test intent, expected behaviors, and likely failures. "
            "If helpful, you may outline pytest snippets in prose, but your structured output "
            "must be a report."
        ),
        model=cfg.model,
        output_type=TestAgentSchema,
        model_settings=ModelSettings(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            parallel_tool_calls=cfg.parallel_tool_calls,
            store=cfg.store,
        ),
    )


# ---------------------------
# Workflow I/O models
# ---------------------------

class DevIn(WorkflowInput):
    prompt: str = Field(..., description="Development prompt / ticket text")


class DevOut(WorkflowOutput):
    developer_report: str


class TestIn(WorkflowInput):
    prompt: str = Field(..., description="Testing prompt")
    test_input: str = Field(..., description="Subject under test (code path, snippet, API contract, etc.)")
    repair_code: bool = Field(default=False, description="If true, reuse Dev Agent to apply a fix and return a repair report")


class TestOut(WorkflowOutput):
    test_report: str
    repair_report: Optional[str] = None  # report of changes made by dev agent (no code)


# ---------------------------
# Server setup
# ---------------------------

# Privacy/serving policy for conversations (sanitization + visibility)
conv_policy = ConversationResourcePolicy.defaults()
# Example overrides (tweak as you like):
conv_policy.visibility_by_channel.update({"debug": "private"})   # drop debug from served resources
conv_policy.visibility_by_role.update({"system": "redacted"})    # serve system messages in redacted form

DEFAULT_SEEDING = WorkflowConversationSeeding(
    default_seeding=ConversationSeedingWindow(messages=20),
    default_channels=["default"],
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

# ---------------------------
# Workflows
# ---------------------------

@server.workflow(
    name="dev",
    title="AI Engineer: Development",
    description="Takes a prompt and returns a report of what was achieved.",
    input_model=DevIn,
    output_model=DevOut,
    seeding=DEFAULT_SEEDING,
    input_limit_bytes=1 * 1024 * 1024,  # 1 MiB cap (dev)
    settings_default=DevSettings(),      # enable workflow settings with defaults
    settings_allow_override=True,        # allow overrides, but 'model' is locked via DevSettings.locked
)
async def dev_flow(wf_input: DevIn, ctx: Context) -> DevOut:
    # Use only the 'dev' channel conversation (seed + live).
    conv = await ctx.run.conversations.get(channels=["dev"])
    # Effective workflow settings (no name needed; inferred from current workflow)
    dev_cfg = ctx.get_workflow_settings()  # -> DevSettings instance
    # Example: warn if any override has been applied (instance helper infers workflow via ctx)
    if dev_cfg.settings_overridden(ctx=ctx):
        await ctx.run.telemetry.annotation(
            message=f"dev: non-default settings in effect (model={dev_cfg.model})",
            level="warning",
        )
    dev_agent = build_dev_agent(dev_cfg)

    user_msg = {
        "role": "user",
        "content": [{"type": "input_text", "text": wf_input.prompt}],
    }
    await conv.add_item(user_msg, channel="dev")

    await ctx.run.telemetry.annotation(message="dev: starting", level="info")

    dev_result_temp = await Runner.run(dev_agent, input=conv.get_items(), hooks=OpenAITelemetryHooks(ctx))

    # Persist + mirror assistant messages to the same handle
    for item in dev_result_temp.new_items:
        await conv.add_item(item.to_input_item(), channel="dev")

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
    settings_default=TestSettings(),     # enable workflow settings with defaults
    settings_allow_override=False,       # demonstrate all-or-nothing: no overrides permitted
)
async def test_flow(wf_input: TestIn, ctx: Context) -> TestOut:
    # Primary conversation for Test Agent uses only 'test' channel (seed + live).
    test_conv = await ctx.run.conversations.get(channels=["test"])
    test_cfg = ctx.get_workflow_settings()

    # Alternative override detection via manager flag (canonical, cheap):
    if ctx.wrp._workflow_manager.settings_overridden(ctx.run.workflow_name):
        await ctx.run.telemetry.annotation(
            message=f"test: non-default settings in effect (model={test_cfg.model})",
            level="warning",
        )

    test_agent = build_test_agent(test_cfg)

    user_test_msg = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": f"TEST PROMPT:\n{wf_input.prompt}"},
            {"type": "input_text", "text": f"TEST INPUT:\n{wf_input.test_input}"},
        ],
    }
    await test_conv.add_item(user_test_msg, channel="test")

    await ctx.run.telemetry.annotation(
        message="test: starting",
        level="info",
        data={"repair_code": wf_input.repair_code},
    )

    test_result_temp = await Runner.run(test_agent, input=test_conv.get_items(), hooks=OpenAITelemetryHooks(ctx))

    for item in test_result_temp.new_items:
        await test_conv.add_item(item.to_input_item(), channel="test")

    # ---- typed result ----
    test_result = {
        "output_text": test_result_temp.final_output.json(),
        "output_parsed": test_result_temp.final_output.model_dump(),
    }
    test_report = test_result["output_parsed"]["test_report"]

    repair_report: Optional[str] = None

    if wf_input.repair_code:
        # Reuse dev agent with 'dev' channel (seed + live).
        dev_conv = await ctx.run.conversations.get(channels=["dev"])
        # Pull another workflow’s settings by name when needed
        dev_cfg = ctx.get_workflow_settings("dev")
        dev_agent = build_dev_agent(dev_cfg)

        repair_prompt = (
            "Apply a minimal, safe fix based on the following testing feedback. "
            "Modify the repository files as needed, then return ONLY a concise REPORT of changes made "
            "(no code blocks).\n\n"
            f"=== TEST REPORT ===\n{test_report}\n\n"
            f"=== SUBJECT UNDER TEST ===\n{wf_input.test_input}\n"
        )
        repair_user_msg = {
            "role": "user",
            "content": [{"type": "input_text", "text": repair_prompt}],
        }
        await dev_conv.add_item(repair_user_msg, channel="dev")

        await ctx.run.telemetry.annotation(message="test: repairing with dev agent", level="info")

        repair_result_temp = await Runner.run(dev_agent, input=dev_conv.get_items(), hooks=OpenAITelemetryHooks(ctx))

        for item in repair_result_temp.new_items:
            await dev_conv.add_item(item.to_input_item(), channel="dev")

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