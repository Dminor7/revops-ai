from revops_ai.connectors.base import Capability, Connector, SyncMetadata

__all__ = ["Capability", "Connector", "SyncMetadata"]

# Concrete connectors (HubSpotConnector, StripeConnector, WarehouseConnector)
# are imported from their modules directly so optional dependencies stay
# optional: e.g. `from revops_ai.connectors.hubspot import HubSpotConnector`.
