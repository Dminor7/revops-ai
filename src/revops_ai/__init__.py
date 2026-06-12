"""revops-ai: an extensible SDK for auditable, write-safe RevOps AI agents.

Quick start::

    from revops_ai import RevOpsEngine, LLMConfig

    engine = RevOpsEngine(llm=LLMConfig(model="openai:gpt-4o"))
    engine.register_connector("crm", MyHubSpotConnector(...))
    engine.register_agent(MyAgent())
    report = await engine.run(MyTask(...))
"""

from revops_ai.agents.base import BaseAgent, LLMAgent
from revops_ai.connectors.base import Capability, Connector, SyncMetadata
from revops_ai.core.context import RunContext
from revops_ai.core.engine import RevOpsEngine
from revops_ai.core.settings import EngineSettings, FreshnessPolicy, LLMConfig
from revops_ai.exceptions import (
    CapabilityError,
    NoAgentForTaskError,
    RegistrationError,
    RevOpsError,
    StaleDataError,
    UnknownRoleError,
)
from revops_ai.tasks.base import EntityRef, Evidence, Finding, Report, SourceRef, Task
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
    "Finding",
    "FreshnessPolicy",
    "LLMAgent",
    "LLMConfig",
    "NoAgentForTaskError",
    "RegistrationError",
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
]
