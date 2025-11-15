# wrp/server/runtime/telemetry/privacy/policy.py
from __future__ import annotations
from typing import Literal, Dict, List, Set, Callable, Any
from pydantic import BaseModel, Field

Visibility = Literal["public", "redacted", "private"]

def vis(v: Visibility) -> Visibility:
    return v

class RedactRules(BaseModel):
    # JSONPath-lite paths to drop entirely (e.g., "system_prompt", "args.password", "mcp_config.token")
    drop: List[str] = Field(default_factory=list)
    # Paths to mask values with "***"
    mask: List[str] = Field(default_factory=list)
    # For strings: apply scrub regexes by name (see redaction.py)
    scrub: List[str] = Field(default_factory=list)
    # Hide usage counters for this kind ("usage", "usage_total", "usage_llms")
    hide_usage: bool = False

class TelemetryResourcePolicy(BaseModel):
    """
    Serving-only policy for telemetry payloads and spans.
    Storage is not affected (v1).
    """
    # Per payload-kind visibility
    visibility: Dict[str, Visibility] = Field(default_factory=dict)
    # Per payload-kind redaction rules (applied when visibility=="redacted")
    rules: Dict[str, RedactRules] = Field(default_factory=dict)
    # Optional global rules applied in addition to per-kind rules when serving redacted
    global_rules: RedactRules | None = None
    # Author-extensible scrub registry: { "name": "<regex string>" }
    # (kept as strings for JSON-ability; compiled at use)
    custom_scrub_patterns: Dict[str, str] = Field(default_factory=dict)

    # Span overlays (applied only when serving spans via any endpoint we add)
    mask_model_in_spans: bool = False
    mask_tool_names_in_spans: bool = False
    mask_usage: bool = False  # global "hide usage" toggle; also applied to payload kinds if set
    # If True, apply global_rules even when a part is served as `public`
    # (lets us keep defaults open, but still scrub obvious secrets).
    apply_global_in_public: bool = True

    @classmethod
    def defaults(cls, *, mask_usage: bool = False,
                 mask_model_in_spans: bool = False,
                 mask_tool_names_in_spans: bool = False) -> "TelemetryResourcePolicy":
        # hard rules
        v: Dict[str, Visibility] = {
            "run.start": "public",
            "run.end": "public",
        }
        # OSS-first stance: internals are PUBLIC by default
        for k in ("agent.start","agent.end","llm.start","llm.end","tool.start","tool.end"):
            v[k] = "public"
        # point-span payload kinds are sensitive by default
        v["guardrail_result"] = "redacted"
        v["annotation"] = "redacted"
        v["handoff"] = "redacted"

        rules: Dict[str, RedactRules] = {
            # Keep system_prompt visible by default for OSS, but still mask secrets.
            "agent.start": RedactRules(
                mask=[
                    "tools[*].config.api_key",
                    "tools[*].config.authorization",
                    "tools[*].config.cookie",
                    "mcp_servers[*].token",
                    "mcp_config.authorization",
                    # be explicit for model_settings too (globals also cover these)
                    "model_settings.api_key",
                    "model_settings.authorization",
                    "model_settings.headers.*",
                ],
                hide_usage=False,
            ),
            "agent.end": RedactRules(
                mask=["final_output"],
                hide_usage=True,
            ),
            # Keep system_prompt visible; still mask sensitive model settings.
            "llm.start": RedactRules(
                mask=[
                    "model_settings.api_key",
                    "model_settings.authorization",
                    "model_settings.headers.*",
                ],
                hide_usage=False,
            ),
            "llm.end": RedactRules(
                hide_usage=True,
            ),
            "tool.start": RedactRules(
                mask=["args.password", "args.api_key", "args.jwt", "args.secret", "args.token"],
                hide_usage=False,
            ),
            "tool.end": RedactRules(
                mask=["result.password", "result.api_key", "result.jwt", "result.secret", "result.token"],
                hide_usage=False,
            ),
            # Guardrail payloads can include user input/output; serve with aggressive redaction.
            "guardrail_result": RedactRules(
                drop=["input_items"],                     # drop raw user input list
                mask=["agent_output", "output_info"],    # mask model output + reasoning/details
                hide_usage=False,
            ),
        }

        pol = cls(
            visibility=v,
            rules=rules,
            # Open-by-default but safe: scrub common secrets EVEN WHEN PUBLIC
            global_rules=RedactRules(
                # drop nothing by default (avoid data loss surprises)
                drop=[],
                # mask likely secret fields by name, anywhere (include common capitalizations)
                mask=[
                    "**.password","**.secret","**.token","**.api_key",
                    "**.authorization","**.Authorization",
                    "**.jwt","**.refresh_token",
                    "**.x-api-key","**.X-API-Key"
                ],
                # scrub common secret-looking strings in arbitrary text
                scrub=["email", "url_creds", "api_key_like", "jwt", "bearer", "pem_private_key"],
                hide_usage=False,
            ),
            mask_model_in_spans=mask_model_in_spans,
            mask_tool_names_in_spans=mask_tool_names_in_spans,
            mask_usage=mask_usage,
            apply_global_in_public=True,
        )
        return pol

    def with_overrides(self, vis_overrides: Dict[str, Visibility] | None = None,
                       rule_overrides: Dict[str, RedactRules] | None = None,
                       *, global_rules: RedactRules | None = None,
                       custom_scrub_patterns: Dict[str, str] | None = None) -> "TelemetryResourcePolicy":
        v = dict(self.visibility)
        r = {k: rules_obj.model_copy(deep=True) for k, rules_obj in self.rules.items()}
        if vis_overrides:
            v.update(vis_overrides)
        if rule_overrides:
            r.update(rule_overrides)
        # merge custom regexes (policy-level)
        c = dict(self.custom_scrub_patterns)
        if custom_scrub_patterns:
            c.update(custom_scrub_patterns)
        return TelemetryResourcePolicy(
            visibility=v,
            rules=r,
            global_rules=global_rules if global_rules is not None else (self.global_rules.model_copy(deep=True) if self.global_rules else None),
            custom_scrub_patterns=c,
            mask_model_in_spans=self.mask_model_in_spans,
            mask_tool_names_in_spans=self.mask_tool_names_in_spans,
            mask_usage=self.mask_usage,
            apply_global_in_public=self.apply_global_in_public,
        )

    def mode_for_kind(self, kind: str) -> Visibility:
        # hard rules: run start/end always public
        if kind in ("run.start", "run.end"):
            return "public"
        return self.visibility.get(kind, "redacted")

    # convenience: author-friendly toggles
    def set_all_internals_private(self) -> "TelemetryResourcePolicy":
        v = dict(self.visibility)
        for k in ("agent.start","agent.end","llm.start","llm.end","tool.start","tool.end",
                  "guardrail_result","annotation","handoff"):
            v[k] = "private"
        return self.with_overrides(v)