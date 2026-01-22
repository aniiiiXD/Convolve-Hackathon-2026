// Clinical types matching the backend agents

export type AlertSeverity = "critical" | "high" | "medium" | "low";
export type AlertType =
  | "critical_value"
  | "deterioration"
  | "drug_interaction"
  | "missed_followup"
  | "trend_alert"
  | "preventive_care";

export interface ClinicalAlert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  title: string;
  message: string;
  patient_id: string;
  created_at: string;
  acknowledged: boolean;
  evidence_ids: string[];
}

export interface DiagnosticCandidate {
  diagnosis: string;
  confidence: number;
  supporting_evidence: string[];
  differentiating_features: string[];
}

export interface DifferentialResult {
  primary_diagnosis: string;
  primary_confidence: number;
  alternatives: DiagnosticCandidate[];
  red_flags: string[];
  recommended_workup: string[];
  reasoning: string;
  evidence_ids: string[];
}

export type InsightType =
  | "temporal_trend"
  | "treatment_effectiveness"
  | "cohort_comparison"
  | "risk_pattern"
  | "correlation"
  | "anomaly"
  | "prediction";

export interface GeneratedInsight {
  id: string;
  insight_type: InsightType;
  title: string;
  description: string;
  confidence: number;
  supporting_data: Record<string, unknown>;
  evidence_ids: string[];
  actionable: boolean;
  recommended_actions: string[];
  generated_at: string;
}

export interface StateChange {
  id: string;
  category: string;
  description: string;
  previous_value: string;
  current_value: string;
  significance: string;
  detected_at: string;
}

export type NodeType =
  | "evidence"
  | "finding"
  | "reasoning"
  | "conclusion"
  | "recommendation";

export interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  content: string;
  confidence?: number;
  source_id?: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight?: number;
}

export interface EvidenceGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  root_id: string;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  chief_complaint?: string;
  status: "stable" | "monitoring" | "critical";
  last_visit: string;
  conditions: string[];
}

export interface VitalSigns {
  heart_rate: number;
  blood_pressure: { systolic: number; diastolic: number };
  temperature: number;
  respiratory_rate: number;
  oxygen_saturation: number;
  recorded_at: string;
}

export interface LabResult {
  id: string;
  test_name: string;
  value: number;
  unit: string;
  reference_range: { min: number; max: number };
  status: "normal" | "high" | "low" | "critical";
  recorded_at: string;
}

export interface SearchResult {
  id: string;
  score: number;
  content: string;
  metadata: {
    patient_id?: string;
    record_type: string;
    timestamp: string;
    [key: string]: unknown;
  };
}

export interface ComprehensiveAnalysis {
  patient_id: string;
  chief_complaint: string;
  differential_diagnosis: DifferentialResult;
  insights: GeneratedInsight[];
  alerts: ClinicalAlert[];
  changes: StateChange[];
  evidence_graph: EvidenceGraph;
  generated_at: string;
  processing_time_ms: number;
}

export interface User {
  id: string;
  name: string;
  role: "doctor" | "patient";
  avatar?: string;
}
