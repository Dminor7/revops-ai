"""revops-ai: an extensible SDK for auditable, write-safe RevOps AI agents.

Quick start::

    from revops_ai import RevOpsEngine, LLMConfig

    engine = RevOpsEngine(llm=LLMConfig(model="openai:gpt-4o"))
    engine.register_connector("crm", MyHubSpotConnector(...))
    engine.register_agent(MyAgent())
    report = await engine.run(MyTask(...))
"""

from revops_ai.agents.base import BaseAgent, LLMAgent
from revops_ai.audit.evidence import EntityRef, Evidence, Finding, SourceRef
from revops_ai.connectors.base import Capability, Connector, SyncMetadata
from revops_ai.core.context import RunContext
from revops_ai.core.engine import RevOpsEngine
from revops_ai.core.settings import EngineSettings, FreshnessPolicy, LLMConfig
from revops_ai.exceptions import (
    CapabilityError,
    NoAgentForTaskError,
    RegistrationError,
    ReplayError,
    RevOpsError,
    StaleDataError,
    UnknownRoleError,
    WriteError,
)
from revops_ai.safety.policies import WritePolicy
from revops_ai.safety.write_intent import FieldChange, WriteIntent, WriteIntentStatus
from revops_ai.tasks.base import Report, Task
from revops_ai.tools.base import Tool

__version__ = "0.1.0a0"

__all__ = [
    "BaseAgent",
    "Capability",
    "CapabilityError",
    "Connector",
    "EngineSettings",
    "EntityRef",
    "Evidence",
    "FieldChange",
    "Finding",
    "FreshnessPolicy",
    "LLMAgent",
    "LLMConfig",
    "NoAgentForTaskError",
    "RegistrationError",
    "ReplayError",
    "Report",
    "RevOpsEngine",
    "RevOpsError",
    "RunContext",
    "SourceRef",
    "StaleDataError",
    "SyncMetadata",
    "Task",
    "Tool",
    "UnknownRoleError",
    "WriteError",
    "WriteIntent",
    "WriteIntentStatus",
    "WritePolicy",
]
