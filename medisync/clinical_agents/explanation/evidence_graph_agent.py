"""
Evidence Graph Agent

Generates visual evidence chains for clinical decisions, providing
explainable AI through graph-based reasoning visualization.

Key Features:
- Evidence chain construction
- Reasoning path visualization
- Confidence scoring with provenance
- Interactive graph export (JSON, DOT, ASCII)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import hashlib

from medisync.service_agents.encoding_agent import EmbeddingService

logger = logging.getLogger(__name__)


class NodeType(Enum):
    PATIENT = "patient"
    SYMPTOM = "symptom"
    EVIDENCE = "evidence"
    REASONING = "reasoning"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    RECOMMENDATION = "recommendation"


class EdgeType(Enum):
    PRESENTS_WITH = "presents_with"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    LEADS_TO = "leads_to"
    SIMILAR_TO = "similar_to"
    INFERRED_FROM = "inferred_from"
    RECOMMENDS = "recommends"


@dataclass
class GraphNode:
    """Node in the evidence graph"""
    node_id: str
    node_type: NodeType
    label: str
    description: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class GraphEdge:
    """Edge in the evidence graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
            "label": self.label,
            "metadata": self.metadata
        }


@dataclass
class EvidenceGraph:
    """Complete evidence graph for a clinical decision"""
    graph_id: str
    title: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    summary: str
    overall_confidence: float
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "title": self.title,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "summary": self.summary,
            "overall_confidence": self.overall_confidence,
            "created_at": self.created_at.isoformat()
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_dot(self) -> str:
        """Export to DOT format for Graphviz"""
        lines = ["digraph EvidenceGraph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=rounded];")

        # Node styling by type
        type_colors = {
            NodeType.PATIENT: "#FFE4B5",
            NodeType.SYMPTOM: "#ADD8E6",
            NodeType.EVIDENCE: "#98FB98",
            NodeType.REASONING: "#DDA0DD",
            NodeType.DIAGNOSIS: "#F0E68C",
            NodeType.TREATMENT: "#87CEEB",
            NodeType.OUTCOME: "#90EE90",
            NodeType.RECOMMENDATION: "#FFB6C1"
        }

        # Add nodes
        for node in self.nodes:
            color = type_colors.get(node.node_type, "#FFFFFF")
            label = f"{node.label}\\n({node.confidence:.0%})"
            lines.append(f'  "{node.node_id}" [label="{label}", fillcolor="{color}", style="filled,rounded"];')

        # Add edges
        for edge in self.edges:
            label = f"{edge.label} ({edge.weight:.0%})"
            style = "dashed" if edge.edge_type == EdgeType.CONTRADICTS else "solid"
            lines.append(f'  "{edge.source_id}" -> "{edge.target_id}" [label="{label}", style="{style}"];')

        lines.append("}")
        return "\n".join(lines)

    def to_ascii(self, max_width: int = 80) -> str:
        """Generate ASCII representation of the graph"""
        lines = []
        lines.append("=" * max_width)
        lines.append(f"EVIDENCE GRAPH: {self.title}")
        lines.append(f"Confidence: {self.overall_confidence:.0%}")
        lines.append("=" * max_width)

        # Group nodes by type
        nodes_by_type = {}
        for node in self.nodes:
            if node.node_type not in nodes_by_type:
                nodes_by_type[node.node_type] = []
            nodes_by_type[node.node_type].append(node)

        # Display order
        display_order = [
            NodeType.PATIENT, NodeType.SYMPTOM, NodeType.EVIDENCE,
            NodeType.REASONING, NodeType.DIAGNOSIS, NodeType.TREATMENT,
            NodeType.RECOMMENDATION, NodeType.OUTCOME
        ]

        for node_type in display_order:
            if node_type in nodes_by_type:
                lines.append(f"\n[{node_type.value.upper()}]")
                for node in nodes_by_type[node_type]:
                    conf_bar = "█" * int(node.confidence * 10) + "░" * (10 - int(node.confidence * 10))
                    lines.append(f"  • {node.label}")
                    lines.append(f"    {node.description[:60]}...")
                    lines.append(f"    Confidence: [{conf_bar}] {node.confidence:.0%}")

        # Show key connections
        lines.append(f"\n{'─' * max_width}")
        lines.append("KEY REASONING PATHS:")

        # Find paths from symptoms to recommendations
        symptom_nodes = [n for n in self.nodes if n.node_type == NodeType.SYMPTOM]
        rec_nodes = [n for n in self.nodes if n.node_type == NodeType.RECOMMENDATION]

        edge_map = {(e.source_id, e.target_id): e for e in self.edges}

        for symptom in symptom_nodes[:3]:
            for rec in rec_nodes[:2]:
                path = self._find_path(symptom.node_id, rec.node_id, edge_map)
                if path:
                    path_str = " → ".join([self._get_node_label(n) for n in path])
                    lines.append(f"\n  {path_str}")

        lines.append(f"\n{'=' * max_width}")
        lines.append(f"Summary: {self.summary}")
        lines.append("=" * max_width)

        return "\n".join(lines)

    def _find_path(
        self,
        start: str,
        end: str,
        edge_map: Dict[Tuple[str, str], GraphEdge],
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """Find path between two nodes (BFS)"""
        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        # Build adjacency from edge_map
        adjacency = {}
        for (src, tgt), edge in edge_map.items():
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(tgt)

        while queue and max_depth > 0:
            current, path = queue.pop(0)

            for neighbor in adjacency.get(current, []):
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

            max_depth -= 1

        return None

    def _get_node_label(self, node_id: str) -> str:
        """Get label for node ID"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node.label
        return node_id


class EvidenceGraphAgent:
    """
    Generates evidence graphs for clinical decisions.

    Creates explainable visualizations showing:
    - What evidence was retrieved
    - How it influenced the decision
    - Confidence levels at each step
    - Alternative reasoning paths
    """

    def __init__(self):
        self.embedder = EmbeddingService()
        self._node_counter = 0

    def generate_diagnostic_graph(
        self,
        patient_context: str,
        symptoms: List[str],
        evidence_records: List[Dict[str, Any]],
        diagnosis_candidates: List[Dict[str, Any]],
        recommendations: List[str]
    ) -> EvidenceGraph:
        """
        Generate evidence graph for a diagnostic decision.

        Args:
            patient_context: Patient demographics/history
            symptoms: List of presenting symptoms
            evidence_records: Retrieved evidence from search
            diagnosis_candidates: Potential diagnoses with scores
            recommendations: Final recommendations

        Returns:
            EvidenceGraph visualizing the reasoning
        """
        nodes = []
        edges = []

        # 1. Add patient node
        patient_node = self._create_node(
            NodeType.PATIENT,
            "Patient",
            patient_context[:100],
            confidence=1.0,
            metadata={"context": patient_context}
        )
        nodes.append(patient_node)

        # 2. Add symptom nodes
        symptom_nodes = []
        for symptom in symptoms:
            symptom_node = self._create_node(
                NodeType.SYMPTOM,
                symptom,
                f"Presenting symptom: {symptom}",
                confidence=0.95
            )
            nodes.append(symptom_node)
            symptom_nodes.append(symptom_node)

            # Connect patient to symptom
            edges.append(GraphEdge(
                source_id=patient_node.node_id,
                target_id=symptom_node.node_id,
                edge_type=EdgeType.PRESENTS_WITH,
                weight=1.0,
                label="presents with"
            ))

        # 3. Add evidence nodes
        evidence_nodes = []
        for i, evidence in enumerate(evidence_records[:5]):
            relevance = evidence.get('score', evidence.get('relevance_score', 0.5))
            text = evidence.get('text_content', evidence.get('snippet', ''))[:100]

            evidence_node = self._create_node(
                NodeType.EVIDENCE,
                f"Evidence {i+1}",
                text,
                confidence=relevance,
                metadata={"record_id": evidence.get('id', evidence.get('record_id'))}
            )
            nodes.append(evidence_node)
            evidence_nodes.append(evidence_node)

            # Connect relevant symptoms to evidence
            for symptom_node in symptom_nodes:
                if self._is_relevant(symptom_node.label, text):
                    edges.append(GraphEdge(
                        source_id=symptom_node.node_id,
                        target_id=evidence_node.node_id,
                        edge_type=EdgeType.SUPPORTS,
                        weight=relevance,
                        label=f"{relevance:.0%} similar"
                    ))

        # 4. Add reasoning nodes (connecting evidence to diagnoses)
        reasoning_nodes = []
        for diagnosis in diagnosis_candidates[:3]:
            diag_name = diagnosis.get('diagnosis', diagnosis.get('name', 'Unknown'))
            diag_conf = diagnosis.get('confidence', diagnosis.get('score', 0.5))
            if isinstance(diag_conf, str):
                diag_conf = {'high': 0.9, 'moderate': 0.7, 'low': 0.4, 'speculative': 0.2}.get(diag_conf, 0.5)

            # Reasoning node
            reasoning_node = self._create_node(
                NodeType.REASONING,
                f"Consider {diag_name}",
                f"Based on symptom pattern and similar cases",
                confidence=diag_conf * 0.9
            )
            nodes.append(reasoning_node)
            reasoning_nodes.append(reasoning_node)

            # Connect evidence to reasoning
            for evidence_node in evidence_nodes:
                if evidence_node.confidence > 0.5:
                    edges.append(GraphEdge(
                        source_id=evidence_node.node_id,
                        target_id=reasoning_node.node_id,
                        edge_type=EdgeType.INFERRED_FROM,
                        weight=evidence_node.confidence,
                        label="suggests"
                    ))

            # Diagnosis node
            diagnosis_node = self._create_node(
                NodeType.DIAGNOSIS,
                diag_name,
                diagnosis.get('explanation', f"Potential diagnosis: {diag_name}"),
                confidence=diag_conf,
                metadata={"tests": diagnosis.get('suggested_tests', [])}
            )
            nodes.append(diagnosis_node)

            # Connect reasoning to diagnosis
            edges.append(GraphEdge(
                source_id=reasoning_node.node_id,
                target_id=diagnosis_node.node_id,
                edge_type=EdgeType.LEADS_TO,
                weight=diag_conf,
                label=f"{diag_conf:.0%} confidence"
            ))

        # 5. Add recommendation nodes
        for i, rec in enumerate(recommendations[:3]):
            rec_node = self._create_node(
                NodeType.RECOMMENDATION,
                f"Action {i+1}",
                rec,
                confidence=0.85
            )
            nodes.append(rec_node)

            # Connect diagnoses to recommendations
            for node in nodes:
                if node.node_type == NodeType.DIAGNOSIS:
                    edges.append(GraphEdge(
                        source_id=node.node_id,
                        target_id=rec_node.node_id,
                        edge_type=EdgeType.RECOMMENDS,
                        weight=0.8,
                        label="recommends"
                    ))

        # Calculate overall confidence
        diagnosis_confidences = [d.get('confidence', 0.5) for d in diagnosis_candidates[:3]]
        if isinstance(diagnosis_confidences[0], str) if diagnosis_confidences else False:
            overall_conf = 0.7
        else:
            overall_conf = sum(c if isinstance(c, float) else 0.5 for c in diagnosis_confidences) / max(len(diagnosis_confidences), 1)

        # Generate summary
        summary = self._generate_summary(symptoms, diagnosis_candidates, recommendations)

        return EvidenceGraph(
            graph_id=self._generate_id("DG"),
            title=f"Diagnostic Analysis: {', '.join(symptoms[:2])}",
            nodes=nodes,
            edges=edges,
            summary=summary,
            overall_confidence=overall_conf
        )

    def generate_treatment_graph(
        self,
        diagnosis: str,
        treatment_options: List[Dict[str, Any]],
        similar_outcomes: List[Dict[str, Any]],
        recommendation: str
    ) -> EvidenceGraph:
        """
        Generate evidence graph for treatment decision.

        Shows how similar case outcomes influence treatment selection.
        """
        nodes = []
        edges = []

        # Diagnosis node (root)
        diag_node = self._create_node(
            NodeType.DIAGNOSIS,
            diagnosis,
            f"Confirmed diagnosis: {diagnosis}",
            confidence=0.95
        )
        nodes.append(diag_node)

        # Treatment option nodes
        treatment_nodes = []
        for treatment in treatment_options:
            name = treatment.get('name', treatment.get('treatment', 'Unknown'))
            effectiveness = treatment.get('effectiveness', treatment.get('success_rate', 0.7))

            treatment_node = self._create_node(
                NodeType.TREATMENT,
                name,
                f"Treatment option: {name}",
                confidence=effectiveness,
                metadata=treatment
            )
            nodes.append(treatment_node)
            treatment_nodes.append(treatment_node)

            edges.append(GraphEdge(
                source_id=diag_node.node_id,
                target_id=treatment_node.node_id,
                edge_type=EdgeType.LEADS_TO,
                weight=effectiveness,
                label="treatment for"
            ))

        # Outcome evidence nodes
        for outcome in similar_outcomes[:5]:
            outcome_type = outcome.get('outcome', outcome.get('type', 'neutral'))
            conf = 0.8 if outcome_type == 'positive' else 0.4 if outcome_type == 'negative' else 0.6

            outcome_node = self._create_node(
                NodeType.OUTCOME,
                f"Case: {outcome_type}",
                outcome.get('description', f"Similar case with {outcome_type} outcome")[:80],
                confidence=conf,
                metadata=outcome
            )
            nodes.append(outcome_node)

            # Connect to relevant treatments
            for treatment_node in treatment_nodes:
                if self._is_relevant(treatment_node.label, outcome.get('description', '')):
                    edge_type = EdgeType.SUPPORTS if outcome_type == 'positive' else EdgeType.CONTRADICTS
                    edges.append(GraphEdge(
                        source_id=outcome_node.node_id,
                        target_id=treatment_node.node_id,
                        edge_type=edge_type,
                        weight=conf,
                        label=outcome_type
                    ))

        # Final recommendation
        rec_node = self._create_node(
            NodeType.RECOMMENDATION,
            "Recommendation",
            recommendation,
            confidence=0.85
        )
        nodes.append(rec_node)

        # Connect best treatment to recommendation
        if treatment_nodes:
            best_treatment = max(treatment_nodes, key=lambda t: t.confidence)
            edges.append(GraphEdge(
                source_id=best_treatment.node_id,
                target_id=rec_node.node_id,
                edge_type=EdgeType.RECOMMENDS,
                weight=best_treatment.confidence,
                label="recommended"
            ))

        return EvidenceGraph(
            graph_id=self._generate_id("TG"),
            title=f"Treatment Analysis: {diagnosis}",
            nodes=nodes,
            edges=edges,
            summary=f"Treatment recommendation for {diagnosis} based on {len(similar_outcomes)} similar cases",
            overall_confidence=0.8
        )

    def generate_risk_assessment_graph(
        self,
        patient_id_hash: str,
        risk_factors: List[Dict[str, Any]],
        risk_score: float,
        similar_patients: List[Dict[str, Any]],
        alerts: List[str]
    ) -> EvidenceGraph:
        """
        Generate evidence graph for risk assessment.

        Shows risk factors, similar patient outcomes, and alerts.
        """
        nodes = []
        edges = []

        # Patient node
        patient_node = self._create_node(
            NodeType.PATIENT,
            f"Patient {patient_id_hash}",
            f"Risk assessment target",
            confidence=1.0
        )
        nodes.append(patient_node)

        # Risk factor nodes
        for factor in risk_factors:
            name = factor.get('name', factor.get('factor', 'Unknown'))
            severity = factor.get('severity', factor.get('weight', 0.5))

            factor_node = self._create_node(
                NodeType.SYMPTOM,  # Using symptom type for risk factors
                name,
                factor.get('description', f"Risk factor: {name}"),
                confidence=severity
            )
            nodes.append(factor_node)

            edges.append(GraphEdge(
                source_id=patient_node.node_id,
                target_id=factor_node.node_id,
                edge_type=EdgeType.PRESENTS_WITH,
                weight=severity,
                label=f"has ({severity:.0%})"
            ))

        # Similar patient evidence
        for sim_patient in similar_patients[:5]:
            outcome = sim_patient.get('outcome', 'neutral')
            similarity = sim_patient.get('similarity', 0.7)

            evidence_node = self._create_node(
                NodeType.EVIDENCE,
                f"Similar: {outcome}",
                f"Similar patient with {outcome} trajectory",
                confidence=similarity
            )
            nodes.append(evidence_node)

            edges.append(GraphEdge(
                source_id=patient_node.node_id,
                target_id=evidence_node.node_id,
                edge_type=EdgeType.SIMILAR_TO,
                weight=similarity,
                label=f"{similarity:.0%} similar"
            ))

        # Risk assessment node
        risk_level = "HIGH" if risk_score > 0.7 else "MODERATE" if risk_score > 0.4 else "LOW"
        risk_node = self._create_node(
            NodeType.REASONING,
            f"Risk: {risk_level}",
            f"Overall risk score: {risk_score:.0%}",
            confidence=risk_score
        )
        nodes.append(risk_node)

        # Connect factors to risk
        for node in nodes:
            if node.node_type == NodeType.SYMPTOM:
                edges.append(GraphEdge(
                    source_id=node.node_id,
                    target_id=risk_node.node_id,
                    edge_type=EdgeType.LEADS_TO,
                    weight=node.confidence,
                    label="contributes"
                ))

        # Alert/recommendation nodes
        for alert in alerts[:3]:
            alert_node = self._create_node(
                NodeType.RECOMMENDATION,
                "Alert",
                alert,
                confidence=0.9
            )
            nodes.append(alert_node)

            edges.append(GraphEdge(
                source_id=risk_node.node_id,
                target_id=alert_node.node_id,
                edge_type=EdgeType.RECOMMENDS,
                weight=0.9,
                label="triggers"
            ))

        return EvidenceGraph(
            graph_id=self._generate_id("RG"),
            title=f"Risk Assessment: {patient_id_hash}",
            nodes=nodes,
            edges=edges,
            summary=f"Risk level: {risk_level} ({risk_score:.0%}). {len(alerts)} alerts generated.",
            overall_confidence=risk_score
        )

    def _create_node(
        self,
        node_type: NodeType,
        label: str,
        description: str,
        confidence: float,
        metadata: Dict[str, Any] = None
    ) -> GraphNode:
        """Create a graph node"""
        self._node_counter += 1
        node_id = f"{node_type.value}_{self._node_counter}"

        return GraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            description=description,
            confidence=min(max(confidence, 0), 1),  # Clamp to [0, 1]
            metadata=metadata or {}
        )

    def _generate_id(self, prefix: str) -> str:
        """Generate unique graph ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}-{timestamp}-{hashlib.md5(timestamp.encode()).hexdigest()[:6]}"

    def _is_relevant(self, term: str, text: str) -> bool:
        """Check if term is relevant to text (simplified)"""
        return term.lower() in text.lower()

    def _generate_summary(
        self,
        symptoms: List[str],
        diagnoses: List[Dict],
        recommendations: List[str]
    ) -> str:
        """Generate summary for diagnostic graph"""
        symptom_str = ", ".join(symptoms[:3])
        top_diag = diagnoses[0].get('diagnosis', diagnoses[0].get('name', 'Unknown')) if diagnoses else "Unknown"

        return (
            f"Based on symptoms ({symptom_str}), primary consideration is {top_diag}. "
            f"{len(recommendations)} actions recommended."
        )


def main():
    """CLI entry point"""
    # Demo usage
    agent = EvidenceGraphAgent()

    # Example diagnostic graph
    graph = agent.generate_diagnostic_graph(
        patient_context="65 year old male with history of hypertension",
        symptoms=["chest pain", "shortness of breath", "fatigue"],
        evidence_records=[
            {"id": "1", "score": 0.85, "text_content": "Similar case with MI presentation"},
            {"id": "2", "score": 0.72, "text_content": "Chest pain resolved with nitroglycerin"},
            {"id": "3", "score": 0.68, "text_content": "Echo showed reduced ejection fraction"}
        ],
        diagnosis_candidates=[
            {"diagnosis": "Acute Coronary Syndrome", "confidence": 0.8, "suggested_tests": ["Troponin", "ECG"]},
            {"diagnosis": "Heart Failure", "confidence": 0.6, "suggested_tests": ["BNP", "Echo"]},
            {"diagnosis": "Angina", "confidence": 0.5, "suggested_tests": ["Stress test"]}
        ],
        recommendations=[
            "Order stat troponin and ECG",
            "Consider cardiology consult",
            "Start aspirin if no contraindications"
        ]
    )

    print(graph.to_ascii())
    print("\n--- DOT Format (for Graphviz) ---")
    print(graph.to_dot()[:500] + "...")


if __name__ == "__main__":
    main()
