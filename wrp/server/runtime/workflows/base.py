# wrp/server/runtime/workflows/base.py
from __future__ import annotations as _annotations

import functools
import inspect
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, Field

from mcp.server.fastmcp.utilities.context_injection import find_context_parameter
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata
from mcp.server.fastmcp.utilities.logging import get_logger

from wrp.server.runtime.exceptions import WorkflowError
from wrp.server.runtime.conversations.seeding import WorkflowConversationSeeding
from wrp.types import Icon
from .types import WorkflowDescriptor, WorkflowInput, WorkflowOutput
from .settings import WorkflowSettings

logger = get_logger(__name__)


def _is_async_callable(obj: Any) -> bool:
    """Return True if obj is an async callable (function or __call__)."""
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.iscoroutinefunction(obj) or (
        callable(obj) and inspect.iscoroutinefunction(getattr(obj, "__call__", None))
    )


class Workflow(BaseModel):
    """Internal registration for a workflow.

    A workflow is a single entry function with one business parameter annotated as a
    subclass of `WorkflowInput`, optional `Context` injection, and a return value
    annotated as a subclass of `WorkflowOutput`.
    """

    fn: Callable[..., Any] = Field(exclude=True)
    name: str = Field(description="Name of the workflow.")
    title: str | None = Field(default=None, description="Human-readable title.")
    description: str = Field(default="", description="Description of what the workflow does.")
    input_model: type[WorkflowInput] = Field(description="Declared WorkflowInput model class.")
    output_model: type[WorkflowOutput] = Field(description="Declared WorkflowOutput model class.")
    is_async: bool = Field(description="Whether the workflow function is async.")
    context_kwarg: str | None = Field(
        default=None,
        description="Name of the kwarg that should receive the Context object, if requested.",
    )
    icons: list[Icon] | None = Field(default=None, description="Optional icons for this workflow.")
    fn_metadata: FuncMetadata = Field(exclude=True)
    business_param_name: str = Field(
        description="The name of the single business parameter for this workflow."
    )
    seeding: WorkflowConversationSeeding | None = Field(
        default=None, description="Author-controlled defaults/limits for seeding and channels."
    )
    input_limit_bytes: int | None = Field(
        default=None, description="Max size (bytes) for raw workflow input payload."
    )
    # Workflow-level, non-run settings contract comes from the *default* instance
    settings_default: WorkflowSettings | None = Field(
        default=None, description="Default settings instance; defines the settings schema."
    )
    # Gate: allow/disallow *any* client override for this workflow
    settings_allow_override: bool = Field(
        default=True, description="Whether clients may override this workflow's settings."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def input_schema(self) -> dict[str, Any]:
        """The JSON schema for the workflow's input model (the raw payload)."""
        return self.input_model.model_json_schema(by_alias=True)

    @property
    def output_schema(self) -> dict[str, Any]:
        """The JSON schema for the workflow's output model."""
        return self.output_model.model_json_schema(by_alias=True)

    def descriptor(self) -> WorkflowDescriptor:
        """Generates a public-facing descriptor for this workflow."""
        desc = WorkflowDescriptor(
            name=self.name,
            title=self.title,
            description=self.description,
            inputSchema=self.input_schema,
            outputSchema=self.output_schema,
            icons=self.icons,
        )
        if self.settings_default is not None:
            desc.settingsSchema = self.settings_default.__class__.model_json_schema(by_alias=True)
        return desc

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        input_model: type[WorkflowInput] | None = None,
        output_model: type[WorkflowOutput] | None = None,
        context_kwarg: str | None = None,
        icons: list[Icon] | None = None,
        seeding: WorkflowConversationSeeding | None = None,
        input_limit_bytes: int | None = None,
        settings_default: WorkflowSettings | None = None,
        settings_allow_override: bool = True,
    ) -> "Workflow":
        """Create a Workflow registration from a developer's function.

        This method introspects a function to ensure it conforms to the workflow
        contract and extracts the necessary metadata.

        The function should look like:

            async def run_something(wf_input: MyInput, ctx: Context) -> MyOutput: ...

        - One business parameter annotated as a subclass of WorkflowInput.
        - Optional Context parameter (detected automatically).
        - Return annotation is a subclass of WorkflowOutput.
        """
        if not callable(fn):
            raise WorkflowError("fn must be callable")

        wf_name = name or fn.__name__
        if wf_name == "<lambda>":
            raise WorkflowError("You must provide a name for lambda functions")

        doc = description or fn.__doc__ or ""

        # Discover context parameter (optional) if not explicitly provided.
        if context_kwarg is None:
            context_kwarg = find_context_parameter(fn)

        fn_meta = func_metadata(
            fn,
            skip_names=[context_kwarg] if context_kwarg else [],
            structured_output=False,  # Workflows have a strict output model contract.
        )

        # A workflow must have exactly one business parameter.
        fields = fn_meta.arg_model.model_fields
        if len(fields) != 1:
            raise WorkflowError(
                f"Workflow '{wf_name}' must have exactly one input parameter (excluding Context). "
                f"Found {len(fields)}."
            )

        (business_param_name, field_info) = next(iter(fields.items()))

        # Resolve and validate the input model.
        # The parameter's annotation must be a subclass of WorkflowInput.
        annotated_input = field_info.annotation
        if not (isinstance(annotated_input, type) and issubclass(annotated_input, WorkflowInput)):
            raise WorkflowError(
                f"Workflow '{wf_name}' input parameter '{business_param_name}' must be annotated "
                f"with a subclass of WorkflowInput"
            )
        # Use the explicit model if provided, otherwise fall back to the annotation.
        final_input_model = input_model or annotated_input

        # Resolve and validate the output model.
        # The function's return annotation must be a subclass of WorkflowOutput.
        annotated_output = get_type_hints(fn).get("return", inspect.signature(fn).return_annotation)
        if not (isinstance(annotated_output, type) and issubclass(annotated_output, WorkflowOutput)):
            raise WorkflowError(
                f"Workflow '{wf_name}' return annotation must be a subclass of WorkflowOutput"
            )
        # Use the explicit model if provided, otherwise fall back to the annotation.
        final_output_model = output_model or annotated_output

        return cls(
            fn=fn,
            name=wf_name,
            title=title,
            description=doc,
            input_model=final_input_model,
            output_model=final_output_model,
            is_async=_is_async_callable(fn),
            context_kwarg=context_kwarg,
            icons=icons,
            fn_metadata=fn_meta,
            business_param_name=business_param_name,
            seeding=seeding,
            input_limit_bytes=input_limit_bytes,
            settings_default=settings_default,
            settings_allow_override=settings_allow_override,
        )

    async def run(self, input_data: dict[str, Any], ctx: "Context") -> WorkflowOutput:  # type: ignore[name-defined]
        """Pre-parses and validates input, executes workflow, validates output, and returns the result."""
        # 1) Use the pre-computed function metadata to validate the input.
        #    The raw input data is wrapped in a dictionary with the business
        #    parameter's name as the key.
        args_dict = {self.business_param_name: input_data}
        try:
            pre_parsed_args = self.fn_metadata.pre_parse_json(args_dict)
            arg_model_instance = self.fn_metadata.arg_model.model_validate(pre_parsed_args)
            call_kwargs = arg_model_instance.model_dump_one_level()
        except Exception as e:
            raise WorkflowError(f"Invalid input for workflow '{self.name}': {e}") from e

        # 2) Inject the context object if the workflow function requested it.
        if self.context_kwarg is not None:
            call_kwargs[self.context_kwarg] = ctx

        # 3) Call the underlying workflow function.
        try:
            result = await self.fn(**call_kwargs) if self.is_async else self.fn(**call_kwargs)
        except Exception as e:
            logger.exception("Exception while running workflow '%s'", self.name)
            raise WorkflowError(str(e)) from e

        # 4) Validate and coerce the function's return value into the declared output model.
        try:
            return (
                result
                if isinstance(result, self.output_model)
                else self.output_model.model_validate(result)
            )
        except Exception as e:
            raise WorkflowError(f"Invalid output from workflow '{self.name}': {e}") from e