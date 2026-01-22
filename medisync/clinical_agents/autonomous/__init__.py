"""
Autonomous Clinical Agents

These agents operate proactively without explicit user queries:
- VigilanceAgent: Monitors patient states and generates alerts
- ChangeDetectionAgent: Detects temporal changes in patient conditions
"""

from medisync.clinical_agents.autonomous.vigilance_agent import VigilanceAgent
from medisync.clinical_agents.autonomous.change_detection_agent import ChangeDetectionAgent

__all__ = ['VigilanceAgent', 'ChangeDetectionAgent']
