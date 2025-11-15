# wrp/types.py
from collections.abc import Callable
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, FileUrl, RootModel
from pydantic.networks import AnyUrl, UrlConstraints
from wrp.server.runtime.workflows.types import WorkflowDescriptor, RunWorkflowResult
from wrp.server.runtime.runs.types import RunMeta, RunOutcome, RunState
from wrp.server.runtime.conversations.types import (
    ChannelMeta,
    ChannelView,
)
from wrp.server.runtime.system_sessions.types import SystemSession
from wrp.server.runtime.telemetry.events import TelemetrySpanView
from wrp.server.runtime.telemetry.payloads.types import SpanPayloadEnvelope

"""
Workflow Runtime Protocol bindings for Python
"""

LATEST_PROTOCOL_VERSION = "2025-11-01"

"""
The default negotiated version of a Workflow Runtime Protocol when no version is specified.
We need this to satisfy the WRP specification, which requires the server to assume a
specific version if none is provided by the client.
"""
DEFAULT_NEGOTIATED_VERSION = "2025-11-01"

ProgressToken = str | int
Cursor = str
Role = Literal["user", "assistant"]
RequestId = Annotated[int, Field(strict=True)] | str
AnyFunction: TypeAlias = Callable[..., Any]


class RequestParams(BaseModel):
    class Meta(BaseModel):
        progressToken: ProgressToken | None = None
        """
        If specified, the caller requests out-of-band progress notifications for
        this request (as represented by notifications/progress). The value of this
        parameter is an opaque token that will be attached to any subsequent
        notifications. The receiver is not obligated to provide these notifications.
        """

        model_config = ConfigDict(extra="allow")

    meta: Meta | None = Field(alias="_meta", default=None)


class PaginatedRequestParams(RequestParams):
    cursor: Cursor | None = None
    """
    An opaque token representing the current pagination position.
    If provided, the server should return results starting after this cursor.
    """


class NotificationParams(BaseModel):
    class Meta(BaseModel):
        model_config = ConfigDict(extra="allow")

    meta: Meta | None = Field(alias="_meta", default=None)
    """
    See [MCP specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/47339c03c143bb4ec01a26e721a1b8fe66634ebe/docs/specification/draft/basic/index.mdx#general-fields)
    for notes on _meta usage.
    """


RequestParamsT = TypeVar("RequestParamsT", bound=RequestParams | dict[str, Any] | None)
NotificationParamsT = TypeVar("NotificationParamsT", bound=NotificationParams | dict[str, Any] | None)
MethodT = TypeVar("MethodT", bound=str)


class Request(BaseModel, Generic[RequestParamsT, MethodT]):
    """Base class for JSON-RPC requests."""

    method: MethodT
    params: RequestParamsT
    model_config = ConfigDict(extra="allow")


class PaginatedRequest(Request[PaginatedRequestParams | None, MethodT], Generic[MethodT]):
    """Base class for paginated requests,
    matching the schema's PaginatedRequest interface."""

    params: PaginatedRequestParams | None = None


class Notification(BaseModel, Generic[NotificationParamsT, MethodT]):
    """Base class for JSON-RPC notifications."""

    method: MethodT
    params: NotificationParamsT
    model_config = ConfigDict(extra="allow")


class Result(BaseModel):
    """Base class for JSON-RPC results."""

    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    """
    See [MCP specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/47339c03c143bb4ec01a26e721a1b8fe66634ebe/docs/specification/draft/basic/index.mdx#general-fields)
    for notes on _meta usage.
    """
    model_config = ConfigDict(extra="allow")


class PaginatedResult(Result):
    nextCursor: Cursor | None = None
    """
    An opaque token representing the pagination position after the last returned result.
    If present, there may be more results available.
    """


class JSONRPCRequest(Request[dict[str, Any] | None, str]):
    """A request that expects a response."""

    jsonrpc: Literal["2.0"]
    id: RequestId
    method: str
    params: dict[str, Any] | None = None


class JSONRPCNotification(Notification[dict[str, Any] | None, str]):
    """A notification which does not expect a response."""

    jsonrpc: Literal["2.0"]
    params: dict[str, Any] | None = None


class JSONRPCResponse(BaseModel):
    """A successful (non-error) response to a request."""

    jsonrpc: Literal["2.0"]
    id: RequestId
    result: dict[str, Any]
    model_config = ConfigDict(extra="allow")


# SDK error codes
CONNECTION_CLOSED = -32000
# REQUEST_TIMEOUT = -32001  # the typescript sdk uses this
# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ErrorData(BaseModel):
    """Error information for JSON-RPC error responses."""

    code: int
    """The error type that occurred."""

    message: str
    """
    A short description of the error. The message SHOULD be limited to a concise single
    sentence.
    """

    data: Any | None = None
    """
    Additional information about the error. The value of this member is defined by the
    sender (e.g. detailed error information, nested errors etc.).
    """

    model_config = ConfigDict(extra="allow")


class JSONRPCError(BaseModel):
    """A response to a request that indicates an error occurred."""

    jsonrpc: Literal["2.0"]
    id: str | int
    error: ErrorData
    model_config = ConfigDict(extra="allow")


class JSONRPCMessage(RootModel[JSONRPCRequest | JSONRPCNotification | JSONRPCResponse | JSONRPCError]):
    pass


class EmptyResult(Result):
    """A response that indicates success but carries no data."""


class BaseMetadata(BaseModel):
    """Base class for entities with name and optional title fields."""

    name: str
    """The programmatic name of the entity."""

    title: str | None = None
    """
    Intended for UI and end-user contexts — optimized to be human-readable and easily understood,
    even by those unfamiliar with domain-specific terminology.

    If not provided, the name should be used for display.
    """


class Icon(BaseModel):
    """An icon for display in user interfaces."""

    src: str
    """URL or data URI for the icon."""

    mimeType: str | None = None
    """Optional MIME type for the icon."""

    sizes: list[str] | None = None
    """Optional list of strings specifying icon dimensions (e.g., ["48x48", "96x96"])."""

    model_config = ConfigDict(extra="allow")


class Implementation(BaseMetadata):
    """Describes the name and version of a WRP implementation."""

    version: str

    websiteUrl: str | None = None
    """An optional URL of the website for this implementation."""

    icons: list[Icon] | None = None
    """An optional list of icons for this implementation."""

    model_config = ConfigDict(extra="allow")


class RootsCapability(BaseModel):
    """Capability for root operations."""

    listChanged: bool | None = None
    """Whether the client supports notifications for changes to the roots list."""
    model_config = ConfigDict(extra="allow")


class ElicitationCapability(BaseModel):
    """Capability for elicitation operations."""

    model_config = ConfigDict(extra="allow")


class ClientCapabilities(BaseModel):
    """Capabilities a client may support."""

    experimental: dict[str, dict[str, Any]] | None = None
    """Experimental, non-standard capabilities that the client supports."""
    elicitation: ElicitationCapability | None = None
    """Present if the client supports elicitation from the user."""
    roots: RootsCapability | None = None
    """Present if the client supports listing roots."""
    model_config = ConfigDict(extra="allow")


class ResourcesCapability(BaseModel):
    """Capability for resources operations."""

    subscribe: bool | None = None
    """Whether this server supports subscribing to resource updates."""
    listChanged: bool | None = None
    """Whether this server supports notifications for changes to the resource list."""
    model_config = ConfigDict(extra="allow")


class WorkflowSettingsCapability(BaseModel):
    """Flags for reading/updating workflow settings + schema."""

    read: bool | None = None  # resource://workflows/{wf}/settings
    update: bool | None = None  # HTTP PUT /workflows/{wf}/settings
    jsonSchema: bool | None = None  # resource://workflows/{wf}/settings/schema
    model_config = ConfigDict(extra="allow")


class WorkflowsCapability(BaseModel):
    """Capability for workflow operations."""

    listChanged: bool | None = None
    """Whether this server supports notifications for changes to the workflow list."""
    settings: WorkflowSettingsCapability | None = None
    model_config = ConfigDict(extra="allow")


class ProviderSettingsCapability(BaseModel):
    """Flags for reading/updating provider settings + schema."""

    read: bool | None = None  # providers/settings/read
    update: bool | None = None  # providers/settings/update
    jsonSchema: bool | None = None  # providers/settings/schema
    model_config = ConfigDict(extra="allow")


class ProvidersCapability(BaseModel):
    """Capability for provider operations (global, non-run)."""

    settings: ProviderSettingsCapability | None = None
    model_config = ConfigDict(extra="allow")


class AgentSettingsCapability(BaseModel):
    """Flags for reading/updating agent settings + schema."""

    read: bool | None = None  # agents/settings/read
    update: bool | None = None  # agents/settings/update
    jsonSchema: bool | None = None  # agents/settings/schema
    model_config = ConfigDict(extra="allow")


class AgentsCapability(BaseModel):
    """Capability for agent operations (global, provider-aware)."""

    settings: AgentSettingsCapability | None = None
    model_config = ConfigDict(extra="allow")


class LoggingCapability(BaseModel):
    """Capability for logging operations."""

    model_config = ConfigDict(extra="allow")


class SystemSessionsCapability(BaseModel):
    """Capability for system sessions handlers (+ subscribe)."""

    list: bool | None = None  # systemSessions/list
    read: bool | None = None  # systemSessions/read
    subscribe: bool | None = None
    model_config = ConfigDict(extra="allow")


class RunsIOCapability(BaseModel):
    read: bool | None = None
    model_config = ConfigDict(extra="allow")


class RunsCapability(BaseModel):
    """Capability for run input/output handlers (+ subscribe)."""

    list: bool | None = None  # runs/list
    read: bool | None = None  # runs/read
    input: RunsIOCapability | None = None  # runs/input/read
    output: RunsIOCapability | None = None  # runs/output/read
    subscribe: bool | None = None
    model_config = ConfigDict(extra="allow")


class TelemetrySpansCapability(BaseModel):
    list: bool | None = None  # .../telemetry/spans
    read: bool | None = None  # .../telemetry/spans/{span_id}
    model_config = ConfigDict(extra="allow")


class TelemetryPayloadsCapability(BaseModel):
    read: bool | None = None  # .../telemetry/spans/{span_id}/payload
    model_config = ConfigDict(extra="allow")


class TelemetryCapability(BaseModel):
    """Capability for telemetry handlers (+ subscribe)."""

    spans: TelemetrySpansCapability | None = None
    payloads: TelemetryPayloadsCapability | None = None
    subscribe: bool | None = None
    model_config = ConfigDict(extra="allow")


class ChannelsCapability(BaseModel):
    list: bool | None = None  # .../conversations/channels/list
    read: bool | None = None  # .../conversations/channel/read/{channel}
    model_config = ConfigDict(extra="allow")


class ConversationsCapability(BaseModel):
    """Capability for conversations handlers (+ subscribe)."""

    channels: ChannelsCapability | None = None
    subscribe: bool | None = None
    model_config = ConfigDict(extra="allow")


class ServerCapabilities(BaseModel):
    """Capabilities that a server may support."""

    experimental: dict[str, dict[str, Any]] | None = None
    """Experimental, non-standard capabilities that the server supports."""
    logging: LoggingCapability | None = None
    """Present if the server supports sending log messages to the client."""
    resources: ResourcesCapability | None = None
    """Present if the server offers any resources to read."""
    workflows: WorkflowsCapability | None = None
    """Present if the server offers any workflows to run."""
    systemSessions: SystemSessionsCapability | None = None
    """Present if the server supports system sessions handlers."""
    runs: RunsCapability | None = None
    """Present if the server supports run input/output handlers."""
    telemetry: TelemetryCapability | None = None
    """Present if the server supports telemetry handlers."""
    conversations: ConversationsCapability | None = None
    """Present if the server supports conversation handlers."""
    providers: ProvidersCapability | None = None
    """Present if the server exposes provider-level settings APIs."""
    agents: AgentsCapability | None = None
    """Present if the server exposes agent-level settings APIs."""
    model_config = ConfigDict(extra="allow")


class InitializeRequestParams(RequestParams):
    """Parameters for the initialize request."""

    protocolVersion: str | int
    """The latest version of the protocol that the client supports."""
    capabilities: ClientCapabilities
    clientInfo: Implementation
    model_config = ConfigDict(extra="allow")


class InitializeRequest(Request[InitializeRequestParams, Literal["initialize"]]):
    """
    This request is sent from the client to the server when it first connects, asking it
    to begin initialization.
    """

    method: Literal["initialize"] = "initialize"
    params: InitializeRequestParams


class InitializeResult(Result):
    """After receiving an initialize request from the client, the server sends this."""

    protocolVersion: str | int
    """The version of the protocol that the server wants to use."""
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: str | None = None
    """Instructions describing how to use the server and its features."""
    model_config = ConfigDict(extra="allow")


class InitializedNotification(Notification[NotificationParams | None, Literal["notifications/initialized"]]):
    """
    This notification is sent from the client to the server after initialization has
    finished.
    """

    method: Literal["notifications/initialized"] = "notifications/initialized"
    params: NotificationParams | None = None


class PingRequest(Request[RequestParams | None, Literal["ping"]]):
    """
    A ping, issued by either the server or the client, to check that the other party is
    still alive.
    """

    method: Literal["ping"] = "ping"
    params: RequestParams | None = None


class ProgressNotificationParams(NotificationParams):
    """Parameters for progress notifications."""

    progressToken: ProgressToken
    """
    The progress token which was given in the initial request, used to associate this
    notification with the request that is proceeding.
    """
    progress: float
    """
    The progress thus far. This should increase every time progress is made, even if the
    total is unknown.
    """
    total: float | None = None
    """Total number of items to process (or total progress required), if known."""
    message: str | None = None
    """
    Message related to progress. This should provide relevant human readable
    progress information.
    """
    model_config = ConfigDict(extra="allow")


class ProgressNotification(Notification[ProgressNotificationParams, Literal["notifications/progress"]]):
    """
    An out-of-band notification used to inform the receiver of a progress update for a
    long-running request.
    """

    method: Literal["notifications/progress"] = "notifications/progress"
    params: ProgressNotificationParams


class ListResourcesRequest(PaginatedRequest[Literal["resources/list"]]):
    """Sent from the client to request a list of resources the server has."""

    method: Literal["resources/list"] = "resources/list"


class Annotations(BaseModel):
    audience: list[Role] | None = None
    priority: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    model_config = ConfigDict(extra="allow")


class Resource(BaseMetadata):
    """A known resource that the server is capable of reading."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """The URI of this resource."""
    description: str | None = None
    """A description of what this resource represents."""
    mimeType: str | None = None
    """The MIME type of this resource, if known."""
    size: int | None = None
    """
    The size of the raw resource content, in bytes (i.e., before base64 encoding
    or any tokenization), if known.

    This can be used by Hosts to display file sizes and estimate context window usage.
    """
    icons: list[Icon] | None = None
    """An optional list of icons for this resource."""
    annotations: Annotations | None = None
    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    """
    See [MCP specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/47339c03c143bb4ec01a26e721a1b8fe66634ebe/docs/specification/draft/basic/index.mdx#general-fields)
    for notes on _meta usage.
    """
    model_config = ConfigDict(extra="allow")


class ResourceTemplate(BaseMetadata):
    """A template description for resources available on the server."""

    uriTemplate: str
    """
    A URI template (according to RFC 6570) that can be used to construct resource
    URIs.
    """
    description: str | None = None
    """A human-readable description of what this template is for."""
    mimeType: str | None = None
    """
    The MIME type for all resources that match this template. This should only be
    included if all resources matching this template have the same type.
    """
    icons: list[Icon] | None = None
    """An optional list of icons for this resource template."""
    annotations: Annotations | None = None
    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    """
    See [MCP specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/47339c03c143bb4ec01a26e721a1b8fe66634ebe/docs/specification/draft/basic/index.mdx#general-fields)
    for notes on _meta usage.
    """
    model_config = ConfigDict(extra="allow")


class ListResourcesResult(PaginatedResult):
    """The server's response to a resources/list request from the client."""

    resources: list[Resource]


class ListResourceTemplatesRequest(PaginatedRequest[Literal["resources/templates/list"]]):
    """Sent from the client to request a list of resource templates the server has."""

    method: Literal["resources/templates/list"] = "resources/templates/list"


class ListResourceTemplatesResult(PaginatedResult):
    """The server's response to a resources/templates/list request from the client."""

    resourceTemplates: list[ResourceTemplate]


class ReadResourceRequestParams(RequestParams):
    """Parameters for reading a resource."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """
    The URI of the resource to read. The URI can use any protocol; it is up to the
    server how to interpret it.
    """
    model_config = ConfigDict(extra="allow")


class ReadResourceRequest(Request[ReadResourceRequestParams, Literal["resources/read"]]):
    """Sent from the client to the server, to read a specific resource URI."""

    method: Literal["resources/read"] = "resources/read"
    params: ReadResourceRequestParams


class ResourceContents(BaseModel):
    """The contents of a specific resource or sub-resource."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """The URI of this resource."""
    mimeType: str | None = None
    """The MIME type of this resource, if known."""
    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    """
    See [MCP specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/47339c03c143bb4ec01a26e721a1b8fe66634ebe/docs/specification/draft/basic/index.mdx#general-fields)
    for notes on _meta usage.
    """
    model_config = ConfigDict(extra="allow")


class TextResourceContents(ResourceContents):
    """Text contents of a resource."""

    text: str
    """
    The text of the item. This must only be set if the item can actually be represented
    as text (not binary data).
    """


class BlobResourceContents(ResourceContents):
    """Binary contents of a resource."""

    blob: str
    """A base64-encoded string representing the binary data of the item."""


class ReadResourceResult(Result):
    """The server's response to a resources/read request from the client."""

    contents: list[TextResourceContents | BlobResourceContents]


class ResourceListChangedNotification(
    Notification[NotificationParams | None, Literal["notifications/resources/list_changed"]]
):
    """
    An optional notification from the server to the client, informing it that the list
    of resources it can read from has changed.
    """

    method: Literal["notifications/resources/list_changed"] = "notifications/resources/list_changed"
    params: NotificationParams | None = None


class SubscribeRequestParams(RequestParams):
    """Parameters for subscribing to a resource."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """
    The URI of the resource to subscribe to. The URI can use any protocol; it is up to
    the server how to interpret it.
    """
    model_config = ConfigDict(extra="allow")


class SubscribeRequest(Request[SubscribeRequestParams, Literal["resources/subscribe"]]):
    """
    Sent from the client to request resources/updated notifications from the server
    whenever a particular resource changes.
    """

    method: Literal["resources/subscribe"] = "resources/subscribe"
    params: SubscribeRequestParams


class UnsubscribeRequestParams(RequestParams):
    """Parameters for unsubscribing from a resource."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """The URI of the resource to unsubscribe from."""
    model_config = ConfigDict(extra="allow")


class UnsubscribeRequest(Request[UnsubscribeRequestParams, Literal["resources/unsubscribe"]]):
    """
    Sent from the client to request cancellation of resources/updated notifications from
    the server.
    """

    method: Literal["resources/unsubscribe"] = "resources/unsubscribe"
    params: UnsubscribeRequestParams


class ResourceUpdatedNotificationParams(NotificationParams):
    """Parameters for resource update notifications."""

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)]
    """
    The URI of the resource that has been updated. This might be a sub-resource of the
    one that the client actually subscribed to.
    """
    model_config = ConfigDict(extra="allow")


class ResourceUpdatedNotification(
    Notification[ResourceUpdatedNotificationParams, Literal["notifications/resources/updated"]]
):
    """
    A notification from the server to the client, informing it that a resource has
    changed and may need to be read again.
    """

    method: Literal["notifications/resources/updated"] = "notifications/resources/updated"
    params: ResourceUpdatedNotificationParams


class ListWorkflowsRequestParams(RequestParams):
    """Parameters for a 'workflows/list' request. Currently empty."""

    model_config = ConfigDict(extra="allow")


class ListWorkflowsRequest(Request[ListWorkflowsRequestParams | None, Literal["workflows/list"]]):
    """Request sent from the client to the server to get a list of available workflows."""

    method: Literal["workflows/list"] = "workflows/list"
    params: ListWorkflowsRequestParams | None = None


class ListWorkflowsResult(Result):
    """Result of a 'workflows/list' request, containing the list of workflows."""

    workflows: list[WorkflowDescriptor]
    model_config = ConfigDict(extra="allow")


class RunWorkflowRequestParams(RequestParams):
    """Parameters for a 'workflows/run' request."""

    name: str = Field(description="The name of the workflow to run.")
    input: dict[str, Any] | None = Field(
        default=None,
        description="The input data for the workflow, matching its inputSchema.",
    )
    model_config = ConfigDict(extra="allow")


class RunWorkflowRequest(Request[RunWorkflowRequestParams, Literal["workflows/run"]]):
    """Request sent from the client to the server to execute a workflow."""

    method: Literal["workflows/run"] = "workflows/run"
    params: RunWorkflowRequestParams


#
# Workflow Settings API (lowlevel-defined)
#


class WorkflowSettingsReadRequestParams(RequestParams):
    """Parameters for workflows/settings/read."""

    workflow: str
    model_config = ConfigDict(extra="allow")


class WorkflowSettingsReadRequest(Request[WorkflowSettingsReadRequestParams, Literal["workflows/settings/read"]]):
    """Read the effective (possibly persisted/overridden) settings for a workflow."""

    method: Literal["workflows/settings/read"] = "workflows/settings/read"
    params: WorkflowSettingsReadRequestParams


class WorkflowSettingsSchemaRequestParams(RequestParams):
    """Parameters for workflows/settings/schema."""

    workflow: str
    model_config = ConfigDict(extra="allow")


class WorkflowSettingsSchemaRequest(
    Request[WorkflowSettingsSchemaRequestParams, Literal["workflows/settings/schema"]]
):
    """Read the JSON schema for a workflow's settings."""

    method: Literal["workflows/settings/schema"] = "workflows/settings/schema"
    params: WorkflowSettingsSchemaRequestParams


class WorkflowSettingsUpdateRequestParams(RequestParams):
    """Parameters for workflows/settings/update (partial upsert)."""

    workflow: str
    values: dict[str, Any] | None = None
    model_config = ConfigDict(extra="allow")


class WorkflowSettingsUpdateRequest(
    Request[WorkflowSettingsUpdateRequestParams, Literal["workflows/settings/update"]]
):
    """Update (merge) the workflow's settings."""

    method: Literal["workflows/settings/update"] = "workflows/settings/update"
    params: WorkflowSettingsUpdateRequestParams


class WorkflowsListChangedNotification(
    Notification[NotificationParams | None, Literal["notifications/workflows/list_changed"]]
):
    """Notification sent from the server to the client when the list of available workflows changes."""

    method: Literal["notifications/workflows/list_changed"] = "notifications/workflows/list_changed"
    params: NotificationParams | None = None


class WorkflowSettingsReadResult(Result):
    """Result for workflows/settings/read and workflows/settings/update."""

    values: dict[str, Any] | None = None
    overridden: bool | None = None
    allowOverride: bool | None = None
    locked: list[str] | None = None


class WorkflowSettingsSchemaResult(Result):
    """Result for workflows/settings/schema."""

    jsonSchema: dict[str, Any] | None = None


#
# Provider Settings API (lowlevel-defined)
#


class ProviderSettingsReadRequestParams(RequestParams):
    """Parameters for providers/settings/read."""

    provider: str
    model_config = ConfigDict(extra="allow")


class ProviderSettingsReadRequest(
    Request[ProviderSettingsReadRequestParams, Literal["providers/settings/read"]]
):
    """Read the effective (possibly persisted/overridden) settings for a provider."""

    method: Literal["providers/settings/read"] = "providers/settings/read"
    params: ProviderSettingsReadRequestParams


class ProviderSettingsSchemaRequestParams(RequestParams):
    """Parameters for providers/settings/schema."""

    provider: str
    model_config = ConfigDict(extra="allow")


class ProviderSettingsSchemaRequest(
    Request[ProviderSettingsSchemaRequestParams, Literal["providers/settings/schema"]]
):
    """Read the JSON schema for a provider's settings."""

    method: Literal["providers/settings/schema"] = "providers/settings/schema"
    params: ProviderSettingsSchemaRequestParams


class ProviderSettingsUpdateRequestParams(RequestParams):
    """Parameters for providers/settings/update (partial upsert)."""

    provider: str
    values: dict[str, Any] | None = None
    model_config = ConfigDict(extra="allow")


class ProviderSettingsUpdateRequest(
    Request[ProviderSettingsUpdateRequestParams, Literal["providers/settings/update"]]
):
    """Update (merge) the provider's settings."""

    method: Literal["providers/settings/update"] = "providers/settings/update"
    params: ProviderSettingsUpdateRequestParams


class ProviderSecretSummary(BaseModel):
    """Metadata for a secret provider field (never exposes the raw secret)."""

    hasValue: bool
    model_config = ConfigDict(extra="allow")


class ProviderSettingsReadResult(Result):
    """Result for providers/settings/read and providers/settings/update."""

    values: dict[str, Any] | None = None
    overridden: bool | None = None
    allowOverride: bool | None = None
    locked: list[str] | None = None
    secrets: dict[str, ProviderSecretSummary] | None = None


class ProviderSettingsSchemaResult(Result):
    """Result for providers/settings/schema."""

    jsonSchema: dict[str, Any] | None = None


#
# Agent Settings API (lowlevel-defined)
#


class AgentSettingsReadRequestParams(RequestParams):
    """Parameters for agents/settings/read."""

    agent: str
    model_config = ConfigDict(extra="allow")


class AgentSettingsReadRequest(
    Request[AgentSettingsReadRequestParams, Literal["agents/settings/read"]]
):
    """Read the effective (possibly persisted/overridden) settings for an agent."""

    method: Literal["agents/settings/read"] = "agents/settings/read"
    params: AgentSettingsReadRequestParams


class AgentSettingsSchemaRequestParams(RequestParams):
    """Parameters for agents/settings/schema."""

    agent: str
    model_config = ConfigDict(extra="allow")


class AgentSettingsSchemaRequest(
    Request[AgentSettingsSchemaRequestParams, Literal["agents/settings/schema"]]
):
    """Read the JSON schema for an agent's settings."""

    method: Literal["agents/settings/schema"] = "agents/settings/schema"
    params: AgentSettingsSchemaRequestParams


class AgentSettingsUpdateRequestParams(RequestParams):
    """Parameters for agents/settings/update (partial upsert)."""

    agent: str
    values: dict[str, Any] | None = None
    model_config = ConfigDict(extra="allow")


class AgentSettingsUpdateRequest(
    Request[AgentSettingsUpdateRequestParams, Literal["agents/settings/update"]]
):
    """Update (merge) the agent's settings."""

    method: Literal["agents/settings/update"] = "agents/settings/update"
    params: AgentSettingsUpdateRequestParams


class AgentSettingsReadResult(Result):
    """Result for agents/settings/read and agents/settings/update."""

    values: dict[str, Any] | None = None
    overridden: bool | None = None
    allowOverride: bool | None = None
    locked: list[str] | None = None


class AgentSettingsSchemaResult(Result):
    """Result for agents/settings/schema."""

    jsonSchema: dict[str, Any] | None = None


LoggingLevel = Literal["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"]


class SetLevelRequestParams(RequestParams):
    """Parameters for setting the logging level."""

    level: LoggingLevel
    """The level of logging that the client wants to receive from the server."""
    model_config = ConfigDict(extra="allow")


class SetLevelRequest(Request[SetLevelRequestParams, Literal["logging/setLevel"]]):
    """A request from the client to the server, to enable or adjust logging."""

    method: Literal["logging/setLevel"] = "logging/setLevel"
    params: SetLevelRequestParams


class LoggingMessageNotificationParams(NotificationParams):
    """Parameters for logging message notifications."""

    level: LoggingLevel
    """The severity of this log message."""
    logger: str | None = None
    """An optional name of the logger issuing this message."""
    data: Any
    """
    The data to be logged, such as a string message or an object. Any JSON serializable
    type is allowed here.
    """
    model_config = ConfigDict(extra="allow")


class LoggingMessageNotification(Notification[LoggingMessageNotificationParams, Literal["notifications/message"]]):
    """Notification of a log message passed from server to client."""

    method: Literal["notifications/message"] = "notifications/message"
    params: LoggingMessageNotificationParams


class ListRootsRequest(Request[RequestParams | None, Literal["roots/list"]]):
    """
    Sent from the server to request a list of root URIs from the client. Roots allow
    servers to ask for specific directories or files to operate on. A common example
    for roots is providing a set of repositories or directories a server should operate
    on.

    This request is typically used when the server needs to understand the file system
    structure or access specific locations that the client has permission to read from.
    """

    method: Literal["roots/list"] = "roots/list"
    params: RequestParams | None = None


class Root(BaseModel):
    """Represents a root directory or file that the server can operate on."""

    uri: FileUrl
    """
    The URI identifying the root. This *must* start with file:// for now.
    This restriction may be relaxed in future versions of the protocol to allow
    other URI schemes.
    """
    name: str | None = None
    """
    An optional name for the root. This can be used to provide a human-readable
    identifier for the root, which may be useful for display purposes or for
    referencing the root in other parts of the application.
    """
    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    """
    See [MCP specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/47339c03c143bb4ec01a26e721a1b8fe66634ebe/docs/specification/draft/basic/index.mdx#general-fields)
    for notes on _meta usage.
    """
    model_config = ConfigDict(extra="allow")


class ListRootsResult(Result):
    """
    The client's response to a roots/list request from the server.
    This result contains an array of Root objects, each representing a root directory
    or file that the server can operate on.
    """

    roots: list[Root]


class RootsListChangedNotification(
    Notification[NotificationParams | None, Literal["notifications/roots/list_changed"]]
):
    """
    A notification from the client to the server, informing it that the list of
    roots has changed.

    This notification should be sent whenever the client adds, removes, or
    modifies any root. The server should then request an updated list of roots
    using the ListRootsRequest.
    """

    method: Literal["notifications/roots/list_changed"] = "notifications/roots/list_changed"
    params: NotificationParams | None = None


class CancelledNotificationParams(NotificationParams):
    """Parameters for cancellation notifications."""

    requestId: RequestId
    """The ID of the request to cancel."""
    reason: str | None = None
    """An optional string describing the reason for the cancellation."""
    model_config = ConfigDict(extra="allow")


class CancelledNotification(Notification[CancelledNotificationParams, Literal["notifications/cancelled"]]):
    """
    This notification can be sent by either side to indicate that it is canceling a
    previously-issued request.
    """

    method: Literal["notifications/cancelled"] = "notifications/cancelled"
    params: CancelledNotificationParams


# Type for elicitation schema - a JSON Schema dict
ElicitRequestedSchema: TypeAlias = dict[str, Any]
"""Schema for elicitation requests."""


class ElicitRequestParams(RequestParams):
    """Parameters for elicitation requests."""

    message: str
    requestedSchema: ElicitRequestedSchema
    model_config = ConfigDict(extra="allow")


class ElicitRequest(Request[ElicitRequestParams, Literal["elicitation/create"]]):
    """A request from the server to elicit information from the client."""

    method: Literal["elicitation/create"] = "elicitation/create"
    params: ElicitRequestParams


class ElicitResult(Result):
    """The client's response to an elicitation request."""

    action: Literal["accept", "decline", "cancel"]
    """
    The user action in response to the elicitation.
    - "accept": User submitted the form/confirmed the action
    - "decline": User explicitly declined the action
    - "cancel": User dismissed without making an explicit choice
    """

    content: dict[str, str | int | float | bool | None] | None = None
    """
    The submitted form data, only present when action is "accept".
    Contains values matching the requested schema.
    """


# ------------------------------
# System Events: topics, selectors, subscribe/unsubscribe, updates
# ------------------------------

Topic = Literal[
    "runs/input",
    "runs/output",
    "telemetry/spans",
    "telemetry/span",
    "telemetry/payload",
    "conversations/channels",
    "conversations/channel",
    "systemSessions/list",
    "systemSessions/read",
]


class RunsScope(BaseModel):
    system_session_id: str
    run_id: str


class SpanScope(RunsScope):
    span_id: str


class ChannelScope(RunsScope):
    channel: str


class SystemSessionScope(BaseModel):
    system_session_id: str


class SubscribeOptions(BaseModel):
    deliverInitial: bool | None = None
    coalesceMs: int | None = None
    model_config = ConfigDict(extra="allow")


class SystemEventsSubscribeParams(RequestParams):
    topic: Topic
    runs: RunsScope | None = None
    span: SpanScope | None = None
    channel: ChannelScope | None = None
    session: SystemSessionScope | None = None
    options: SubscribeOptions | None = None
    model_config = ConfigDict(extra="allow")


class SystemEventsSubscribeRequest(Request[SystemEventsSubscribeParams, Literal["systemEvents/subscribe"]]):
    method: Literal["systemEvents/subscribe"] = "systemEvents/subscribe"
    params: SystemEventsSubscribeParams


class SystemEventsSubscribeResult(Result):
    subscriptionId: str


class SystemEventsUnsubscribeParams(RequestParams):
    subscriptionId: str | None = None
    topic: Topic | None = None
    runs: RunsScope | None = None
    span: SpanScope | None = None
    channel: ChannelScope | None = None
    session: SystemSessionScope | None = None
    model_config = ConfigDict(extra="allow")


class SystemEventsUnsubscribeRequest(Request[SystemEventsUnsubscribeParams, Literal["systemEvents/unsubscribe"]]):
    method: Literal["systemEvents/unsubscribe"] = "systemEvents/unsubscribe"
    params: SystemEventsUnsubscribeParams


ChangeKind = Literal["created", "updated", "deleted", "refetch"]


class SystemEventsUpdatedParams(NotificationParams):
    topic: Topic
    sequence: int
    change: ChangeKind | None = None
    runs: RunsScope | None = None
    span: SpanScope | None = None
    channel: ChannelScope | None = None
    session: SystemSessionScope | None = None
    model_config = ConfigDict(extra="allow")


class SystemEventsUpdatedNotification(
    Notification[SystemEventsUpdatedParams, Literal["notifications/systemEvents/updated"]]
):
    method: Literal["notifications/systemEvents/updated"] = "notifications/systemEvents/updated"
    params: SystemEventsUpdatedParams


# ------------------------------
# System handlers: list/read requests & results
# ------------------------------


class RunsIOReadRequestParams(RequestParams):
    runs: RunsScope
    model_config = ConfigDict(extra="allow")


class RunsInputReadRequest(Request[RunsIOReadRequestParams, Literal["runs/input/read"]]):
    method: Literal["runs/input/read"] = "runs/input/read"
    params: RunsIOReadRequestParams


class RunsOutputReadRequest(Request[RunsIOReadRequestParams, Literal["runs/output/read"]]):
    method: Literal["runs/output/read"] = "runs/output/read"
    params: RunsIOReadRequestParams


class RunsIOReadResult(Result):
    data: Any | None = None
    workflow: str | None = None
    jsonSchema: dict[str, Any] | None = None
    system_session_id: str | None = None
    run_id: str | None = None


class RunsListRequestParams(RequestParams):
    system_session: SystemSessionScope
    workflow: str | None = None
    thread_id: str | None = None
    state: RunState | None = None
    outcome: RunOutcome | None = None
    model_config = ConfigDict(extra="allow")


class RunsListRequest(Request[RunsListRequestParams, Literal["runs/list"]]):
    method: Literal["runs/list"] = "runs/list"
    params: RunsListRequestParams


class RunsListResult(Result):
    runs: list[RunMeta]


class RunsReadRequestParams(RequestParams):
    runs: RunsScope
    model_config = ConfigDict(extra="allow")


class RunsReadRequest(Request[RunsReadRequestParams, Literal["runs/read"]]):
    method: Literal["runs/read"] = "runs/read"
    params: RunsReadRequestParams


class RunsReadResult(Result):
    run: RunMeta | None = None


class TelemetrySpansListRequestParams(RequestParams):
    runs: RunsScope
    model_config = ConfigDict(extra="allow")


class TelemetrySpansListRequest(Request[TelemetrySpansListRequestParams, Literal["telemetry/spans/list"]]):
    method: Literal["telemetry/spans/list"] = "telemetry/spans/list"
    params: TelemetrySpansListRequestParams


class TelemetrySpansListResult(Result):
    spans: list[TelemetrySpanView]


class TelemetrySpanReadRequestParams(RequestParams):
    span: SpanScope
    model_config = ConfigDict(extra="allow")


class TelemetrySpanReadRequest(Request[TelemetrySpanReadRequestParams, Literal["telemetry/span/read"]]):
    method: Literal["telemetry/span/read"] = "telemetry/span/read"
    params: TelemetrySpanReadRequestParams


class TelemetrySpanReadResult(Result):
    span: TelemetrySpanView | None = None


class TelemetryPayloadReadRequestParams(RequestParams):
    span: SpanScope
    model_config = ConfigDict(extra="allow")


class TelemetryPayloadReadRequest(Request[TelemetryPayloadReadRequestParams, Literal["telemetry/payload/read"]]):
    method: Literal["telemetry/payload/read"] = "telemetry/payload/read"
    params: TelemetryPayloadReadRequestParams


class TelemetryPayloadReadResult(Result):
    payload: SpanPayloadEnvelope | None = None


class ChannelsListRequestParams(RequestParams):
    runs: RunsScope
    model_config = ConfigDict(extra="allow")


class ChannelsListRequest(
    Request[ChannelsListRequestParams, Literal["conversations/channels/list"]]
):
    method: Literal["conversations/channels/list"] = "conversations/channels/list"
    params: ChannelsListRequestParams


class ChannelsListResult(Result):
    channels: list[ChannelMeta]


class ChannelReadRequestParams(RequestParams):
    channel: ChannelScope
    model_config = ConfigDict(extra="allow")


class ChannelReadRequest(
    Request[ChannelReadRequestParams, Literal["conversations/channel/read"]]
):
    method: Literal["conversations/channel/read"] = "conversations/channel/read"
    params: ChannelReadRequestParams


class ChannelReadResult(Result):
    channel: ChannelView


class SystemSessionsListRequest(Request[RequestParams | None, Literal["systemSessions/list"]]):
    method: Literal["systemSessions/list"] = "systemSessions/list"
    params: RequestParams | None = None


class SystemSessionsListResult(Result):
    sessions: list[SystemSession]


class SystemSessionReadRequestParams(RequestParams):
    session: SystemSessionScope
    model_config = ConfigDict(extra="allow")


class SystemSessionReadRequest(Request[SystemSessionReadRequestParams, Literal["systemSessions/read"]]):
    method: Literal["systemSessions/read"] = "systemSessions/read"
    params: SystemSessionReadRequestParams


class SystemSessionReadResult(Result):
    session: SystemSession | None = None


# ------------------------------
# Client/Server Message Unions
# ------------------------------


class ClientRequest(
    RootModel[
        PingRequest
        | InitializeRequest
        | SetLevelRequest
        | ListResourcesRequest
        | ListResourceTemplatesRequest
        | ReadResourceRequest
        | SubscribeRequest
        | UnsubscribeRequest
        | ListWorkflowsRequest
        | RunWorkflowRequest
        | WorkflowSettingsReadRequest
        | WorkflowSettingsSchemaRequest
        | WorkflowSettingsUpdateRequest
        | ProviderSettingsReadRequest
        | ProviderSettingsSchemaRequest
        | ProviderSettingsUpdateRequest
        | AgentSettingsReadRequest
        | AgentSettingsSchemaRequest
        | AgentSettingsUpdateRequest
        | SystemEventsSubscribeRequest
        | SystemEventsUnsubscribeRequest
        | RunsListRequest
        | RunsReadRequest
        | RunsInputReadRequest
        | RunsOutputReadRequest
        | TelemetrySpansListRequest
        | TelemetrySpanReadRequest
        | TelemetryPayloadReadRequest
        | ChannelsListRequest
        | ChannelReadRequest
        | SystemSessionsListRequest
        | SystemSessionReadRequest
    ]
):
    pass


class ClientNotification(
    RootModel[
        CancelledNotification
        | ProgressNotification
        | InitializedNotification
        | RootsListChangedNotification
    ]
):
    pass


class ClientResult(RootModel[EmptyResult | ListRootsResult | ElicitResult]):
    pass


class ServerRequest(RootModel[PingRequest | ListRootsRequest | ElicitRequest]):
    pass


class ServerNotification(
    RootModel[
        CancelledNotification
        | ProgressNotification
        | LoggingMessageNotification
        | ResourceUpdatedNotification
        | ResourceListChangedNotification
        | WorkflowsListChangedNotification
        | SystemEventsUpdatedNotification
    ]
):
    pass


class ServerResult(
    RootModel[
        EmptyResult
        | InitializeResult
        | ListResourcesResult
        | ListResourceTemplatesResult
        | ReadResourceResult
        | ListWorkflowsResult
        | RunWorkflowResult
        | WorkflowSettingsReadResult
        | WorkflowSettingsSchemaResult
        | ProviderSettingsReadResult
        | ProviderSettingsSchemaResult
        | AgentSettingsReadResult
        | AgentSettingsSchemaResult
        | SystemEventsSubscribeResult
        | RunsListResult
        | RunsReadResult
        | RunsIOReadResult
        | TelemetrySpansListResult
        | TelemetrySpanReadResult
        | TelemetryPayloadReadResult
        | ChannelsListResult
        | ChannelReadResult
        | SystemSessionsListResult
        | SystemSessionReadResult
    ]
):
    pass