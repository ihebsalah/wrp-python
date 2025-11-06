# wrp/server/runtime/conversations/privacy/policy.py
from __future__ import annotations

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field

from wrp.server.runtime.telemetry.privacy.policy import RedactRules  # reuse same rule shape

Visibility = Literal["public", "redacted", "private"]


class ConversationResourcePolicy(BaseModel):
    """
    Serving policy for conversation resources.
    - visibility_default: fallback visibility for messages
    - visibility_by_channel: per-channel overrides (e.g., {"debug": "private"})
    - visibility_by_role: per-role overrides (e.g., {"system": "redacted"})
    - rules: redaction rules when visibility == "redacted" (applied to all roles unless overridden)
    - rules_by_role: extra/overrides per role
    - rules_by_channel: extra/overrides per channel (e.g., tighter for "dev")
    - global_rules: optional rules also applied in redacted mode;
                    optionally applied even in public mode when apply_global_in_public=True
    - custom_scrub_patterns: additional named regexes to use inside RedactRules.scrub
    - apply_global_in_public: scrub obvious secrets even when serving public
    """
    visibility_default: Visibility = "public"
    visibility_by_channel: Dict[str, Visibility] = Field(default_factory=dict)
    visibility_by_role: Dict[str, Visibility] = Field(default_factory=dict)

    rules: RedactRules | None = None
    rules_by_role: Dict[str, RedactRules] = Field(default_factory=dict)
    rules_by_channel: Dict[str, RedactRules] = Field(default_factory=dict)
    global_rules: RedactRules | None = None

    custom_scrub_patterns: Dict[str, str] = Field(default_factory=dict)
    apply_global_in_public: bool = True

    def resolve_visibility(self, *, channel: Optional[str], role: Optional[str]) -> Visibility:
        if role and role in self.visibility_by_role:
            return self.visibility_by_role[role]
        if channel and channel in self.visibility_by_channel:
            return self.visibility_by_channel[channel]
        return self.visibility_default

    @classmethod
    def defaults(cls) -> "ConversationResourcePolicy":
        return cls(
            visibility_default="public",
            visibility_by_channel={},
            visibility_by_role={},
            rules=None,                          # if redacted, apply only if author opts in
            rules_by_role={},                    # open-by-default; authors can tighten
            rules_by_channel={},                 # open-by-default; authors can tighten
            # Open-by-default but safe: scrub common secrets EVEN WHEN PUBLIC
            global_rules=RedactRules(
                drop=[],  # avoid data-loss surprises
                mask=[
                    "**.password", "**.secret", "**.token", "**.api_key",
                    "**.authorization", "**.Authorization",
                    "**.jwt", "**.refresh_token",
                    "**.x-api-key", "**.X-API-Key",
                ],
                scrub=["email", "url_creds", "api_key_like", "jwt", "bearer", "pem_private_key"],
                hide_usage=False,
            ),
            custom_scrub_patterns={},            # authors can add org-specific patterns
            apply_global_in_public=True,
        )