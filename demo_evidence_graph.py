#!/usr/bin/env python3
"""
Evidence Graph Demo - Visual Reasoning Chains
Run: python3 demo_evidence_graph.py
"""

from medisync.clinical_agents.explanation.evidence_graph_agent import EvidenceGraphAgent
import json

def main():
    print("\n" + "="*70)
    print("          MEDISYNC EVIDENCE GRAPH DEMO")
    print("="*70 + "\n")

    agent = EvidenceGraphAgent()

    # Create a realistic clinical scenario
    graph = agent.generate_diagnostic_graph(
        patient_context="65 year old male with history of hypertension, diabetes, and hyperlipidemia",
        symptoms=[
            "crushing substernal chest pain",
            "radiation to left arm",
            "shortness of breath",
            "diaphoresis",
            "nausea"
        ],
        evidence_records=[
            {"id": "lab-001", "score": 0.95, "text_content": "Troponin I: 2.4 ng/mL (elevated, normal <0.04)"},
            {"id": "ecg-001", "score": 0.92, "text_content": "12-lead ECG: ST elevation in V1-V4, reciprocal changes in II, III, aVF"},
            {"id": "note-001", "score": 0.88, "text_content": "Patient clutching chest, diaphoretic, appears in distress"},
            {"id": "vitals-001", "score": 0.85, "text_content": "BP 88/56, HR 112, RR 24, SpO2 94% on RA"},
            {"id": "history-001", "score": 0.75, "text_content": "Similar episode 2 years ago, underwent PCI to LAD"}
        ],
        diagnosis_candidates=[
            {"diagnosis": "ST-Elevation Myocardial Infarction (STEMI)", "confidence": 0.94},
            {"diagnosis": "Unstable Angina / NSTEMI", "confidence": 0.60},
            {"diagnosis": "Aortic Dissection", "confidence": 0.20},
            {"diagnosis": "Pulmonary Embolism", "confidence": 0.15}
        ],
        recommendations=[
            "ACTIVATE CARDIAC CATH LAB - STEMI ALERT",
            "Aspirin 325mg chewed immediately",
            "Heparin 60 units/kg IV bolus",
            "Clopidogrel 600mg loading dose",
            "Morphine 2-4mg IV for pain",
            "Supplemental O2 to maintain SpO2 >94%",
            "Cardiology consult STAT"
        ]
    )

    # 1. ASCII Output (Terminal)
    print("=" * 70)
    print("                    1. ASCII VIEW (Terminal)")
    print("=" * 70)
    print(graph.to_ascii())

    # 2. Save DOT file for GraphViz
    dot_output = graph.to_dot()
    with open("evidence_graph.dot", "w") as f:
        f.write(dot_output)
    print("\n" + "=" * 70)
    print("                    2. GRAPHVIZ DOT FILE SAVED")
    print("=" * 70)
    print("Saved to: evidence_graph.dot")
    print("\nTo visualize:")
    print("  Option A: Go to https://dreampuf.github.io/GraphvizOnline/")
    print("            Paste the contents of evidence_graph.dot")
    print("  Option B: Install graphviz and run:")
    print("            dot -Tpng evidence_graph.dot -o evidence_graph.png")
    print("            dot -Tsvg evidence_graph.dot -o evidence_graph.svg")

    # 3. Save JSON for frontend
    json_output = graph.to_json()
    with open("evidence_graph.json", "w") as f:
        f.write(json_output)
    print("\n" + "=" * 70)
    print("                    3. JSON FILE SAVED")
    print("=" * 70)
    print("Saved to: evidence_graph.json")
    print("Use this for web frontend visualization")

    # 4. Graph Statistics
    print("\n" + "=" * 70)
    print("                    4. GRAPH STATISTICS")
    print("=" * 70)
    print(f"  Total Nodes: {len(graph.nodes)}")
    print(f"  Total Edges: {len(graph.edges)}")

    # Count by type
    from collections import Counter
    node_types = Counter(n.node_type.value for n in graph.nodes)
    print("\n  Node breakdown:")
    for ntype, count in node_types.items():
        print(f"    - {ntype}: {count}")

    print("\n" + "=" * 70)
    print("                    DEMO COMPLETE")
    print("=" * 70)
    print("\nFiles created:")
    print("  - evidence_graph.dot  (GraphViz visualization)")
    print("  - evidence_graph.json (Frontend data)")
    print()

if __name__ == "__main__":
    main()
