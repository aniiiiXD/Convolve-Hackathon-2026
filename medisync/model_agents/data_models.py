"""
Data Models - Qdrant Only

All data is stored in Qdrant collections. No SQL models needed.
This file is kept for backwards compatibility imports.

User model: medisync.service_agents.gatekeeper_agent.User
Query/Interaction data: Qdrant feedback_analytics collection
Clinical records: Qdrant clinical_records collection
"""

# Re-export User from gatekeeper for backwards compatibility
from medisync.service_agents.gatekeeper_agent import User, UserRole
