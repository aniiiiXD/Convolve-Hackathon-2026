"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Card, CardHeader, Button, Badge, ConfidenceMeter } from "@/components/ui";
import { DiagnosisPanel, EvidenceGraphPanel } from "@/components/dashboard";
import {
  Stethoscope,
  Search,
  Send,
  Loader2,
  User,
  FileText,
  Brain,
  ChevronRight,
  Sparkles,
} from "lucide-react";

// Mock differential diagnosis result
const mockDiagnosis = {
  primary_diagnosis: "Acute Coronary Syndrome (NSTEMI)",
  primary_confidence: 0.87,
  alternatives: [
    {
      diagnosis: "Unstable Angina",
      confidence: 0.72,
      supporting_evidence: ["Chest pain pattern", "ECG changes", "Risk factors"],
      differentiating_features: ["No troponin elevation"],
    },
    {
      diagnosis: "Pulmonary Embolism",
      confidence: 0.45,
      supporting_evidence: ["Shortness of breath", "Tachycardia"],
      differentiating_features: ["D-dimer, CT angiography needed"],
    },
    {
      diagnosis: "Aortic Dissection",
      confidence: 0.23,
      supporting_evidence: ["Severe chest pain", "Hypertension"],
      differentiating_features: ["Pain character, CT findings"],
    },
  ],
  red_flags: [
    "Elevated troponin with dynamic changes",
    "ST depression in lateral leads",
    "Ongoing chest pain at rest",
  ],
  recommended_workup: [
    "Serial cardiac enzymes (q6h x3)",
    "12-lead ECG (repeat in 30 min)",
    "Echocardiogram",
    "Cardiology consultation",
    "Consider coronary angiography",
  ],
  reasoning:
    "The presentation of crushing substernal chest pain with radiation to the left arm, associated diaphoresis, and elevated troponin levels in the context of multiple cardiovascular risk factors (hypertension, diabetes, hyperlipidemia) strongly suggests an acute coronary syndrome. The ECG findings of ST depression in leads V4-V6 and elevated troponin I (2.4 ng/mL) are consistent with NSTEMI. The Discovery API found 12 similar cases with 89% concordance.",
  evidence_ids: ["lab-001", "ecg-001", "note-001"],
};

const mockEvidenceGraph = {
  nodes: [
    { id: "e1", type: "evidence" as const, label: "Lab Result", content: "Troponin I: 2.4 ng/mL (elevated)", confidence: 1.0 },
    { id: "e2", type: "evidence" as const, label: "ECG", content: "ST depression V4-V6", confidence: 1.0 },
    { id: "e3", type: "evidence" as const, label: "Clinical Note", content: "Crushing chest pain, diaphoresis", confidence: 1.0 },
    { id: "e4", type: "evidence" as const, label: "History", content: "HTN, DM2, Hyperlipidemia", confidence: 1.0 },
    { id: "f1", type: "finding" as const, label: "Myocardial Injury", content: "Elevated cardiac biomarkers indicate myocardial damage", confidence: 0.95 },
    { id: "f2", type: "finding" as const, label: "Ischemic Changes", content: "ECG consistent with ischemia", confidence: 0.88 },
    { id: "f3", type: "finding" as const, label: "ACS Presentation", content: "Classic anginal symptoms", confidence: 0.92 },
    { id: "r1", type: "reasoning" as const, label: "Risk Stratification", content: "High-risk features present based on HEART score", confidence: 0.85 },
    { id: "c1", type: "conclusion" as const, label: "NSTEMI", content: "Non-ST elevation myocardial infarction", confidence: 0.87 },
    { id: "rec1", type: "recommendation" as const, label: "Urgent Intervention", content: "Early invasive strategy recommended", confidence: 0.82 },
  ],
  edges: [
    { id: "edge1", source: "e1", target: "f1", type: "supports" },
    { id: "edge2", source: "e2", target: "f2", type: "supports" },
    { id: "edge3", source: "e3", target: "f3", type: "supports" },
    { id: "edge4", source: "f1", target: "r1", type: "leads_to" },
    { id: "edge5", source: "f2", target: "r1", type: "leads_to" },
    { id: "edge6", source: "f3", target: "r1", type: "leads_to" },
    { id: "edge7", source: "r1", target: "c1", type: "leads_to" },
    { id: "edge8", source: "c1", target: "rec1", type: "leads_to" },
  ],
  root_id: "c1",
};

export default function DiagnosisPage() {
  const [symptoms, setSymptoms] = useState("");
  const [selectedPatient, setSelectedPatient] = useState("P-101");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [activeTab, setActiveTab] = useState<"diagnosis" | "evidence">("diagnosis");

  const handleAnalyze = async () => {
    if (!symptoms.trim()) return;

    setIsAnalyzing(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setIsAnalyzing(false);
    setShowResults(true);
  };

  const recentPatients = [
    { id: "P-101", name: "John Smith", complaint: "Chest pain" },
    { id: "P-102", name: "Sarah Johnson", complaint: "Headaches" },
    { id: "P-103", name: "Michael Chen", complaint: "Diabetes follow-up" },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-2xl font-bold text-clinical-white flex items-center gap-3"
        >
          <div className="p-2 rounded-xl bg-vital-pulse/20">
            <Stethoscope className="h-6 w-6 text-vital-pulse" />
          </div>
          Differential Diagnosis
        </motion.h1>
        <p className="text-clinical-muted mt-2">
          Enter patient symptoms to generate AI-powered differential diagnoses using the Discovery API
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Input Section */}
        <div className="lg:col-span-1 space-y-4">
          {/* Patient Selection */}
          <Card className="p-4">
            <CardHeader title="Select Patient" />
            <div className="space-y-2">
              {recentPatients.map((patient) => (
                <button
                  key={patient.id}
                  onClick={() => setSelectedPatient(patient.id)}
                  className={`w-full p-3 rounded-xl text-left transition-all ${
                    selectedPatient === patient.id
                      ? "bg-vital-pulse/10 border border-vital-pulse/50"
                      : "bg-midnight-50 border border-transparent hover:border-clinical-border"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-midnight-50 flex items-center justify-center">
                      <User className="h-4 w-4 text-clinical-muted" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-clinical-white">
                        {patient.name}
                      </p>
                      <p className="text-xs text-clinical-muted">{patient.id}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </Card>

          {/* Quick Templates */}
          <Card className="p-4">
            <CardHeader title="Quick Templates" />
            <div className="space-y-2">
              {[
                "Chest pain with dyspnea",
                "Acute abdominal pain",
                "Fever with rash",
                "Altered mental status",
              ].map((template) => (
                <button
                  key={template}
                  onClick={() => setSymptoms(template)}
                  className="w-full p-2 text-left text-sm text-clinical-muted hover:text-clinical-light hover:bg-midnight-50 rounded-lg transition-colors flex items-center gap-2"
                >
                  <FileText className="h-3 w-3" />
                  {template}
                </button>
              ))}
            </div>
          </Card>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3 space-y-6">
          {/* Input Card */}
          <Card className="p-6">
            <div className="flex items-start gap-4">
              <div className="flex-1">
                <label className="block text-sm font-medium text-clinical-light mb-2">
                  Presenting Symptoms
                </label>
                <div className="relative">
                  <textarea
                    value={symptoms}
                    onChange={(e) => setSymptoms(e.target.value)}
                    placeholder="Describe the patient's symptoms, history, and relevant findings..."
                    className="w-full h-32 bg-midnight-50/80 border border-clinical-border/50 rounded-xl px-4 py-3 text-clinical-light placeholder:text-clinical-muted/50 focus:outline-none focus:border-vital-pulse/50 focus:ring-2 focus:ring-vital-pulse/20 transition-all resize-none"
                  />
                  <div className="absolute bottom-3 right-3 flex items-center gap-2">
                    <span className="text-xs text-clinical-muted">
                      {symptoms.length} characters
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between mt-4 pt-4 border-t border-clinical-border/30">
              <div className="flex items-center gap-3">
                <Badge severity="info">Discovery API</Badge>
                <Badge severity="stable">Hybrid Search</Badge>
              </div>
              <Button
                variant="primary"
                onClick={handleAnalyze}
                disabled={!symptoms.trim() || isAnalyzing}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Generate Diagnosis
                  </>
                )}
              </Button>
            </div>
          </Card>

          {/* Results */}
          <AnimatePresence mode="wait">
            {isAnalyzing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <Card className="p-12">
                  <div className="flex flex-col items-center justify-center">
                    <div className="relative">
                      <div className="w-16 h-16 rounded-full bg-vital-pulse/20 flex items-center justify-center">
                        <Brain className="h-8 w-8 text-vital-pulse" />
                      </div>
                      <div className="absolute inset-0 rounded-full border-2 border-vital-pulse/30 animate-ping" />
                    </div>
                    <h3 className="text-lg font-medium text-clinical-white mt-6">
                      Analyzing Clinical Data
                    </h3>
                    <p className="text-sm text-clinical-muted mt-2 text-center max-w-md">
                      Running multi-stage hybrid retrieval with Discovery API refinement...
                    </p>
                    <div className="flex items-center gap-2 mt-4">
                      <div className="h-1 w-24 rounded-full bg-midnight-50 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: "100%" }}
                          transition={{ duration: 2, ease: "linear" }}
                          className="h-full bg-vital-pulse rounded-full"
                        />
                      </div>
                    </div>
                  </div>
                </Card>
              </motion.div>
            )}

            {showResults && !isAnalyzing && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-6"
              >
                {/* Tab Navigation */}
                <div className="flex items-center gap-2 border-b border-clinical-border/30 pb-2">
                  <button
                    onClick={() => setActiveTab("diagnosis")}
                    className={`px-4 py-2 rounded-t-lg font-medium transition-colors ${
                      activeTab === "diagnosis"
                        ? "bg-vital-pulse/10 text-vital-pulse border-b-2 border-vital-pulse"
                        : "text-clinical-muted hover:text-clinical-light"
                    }`}
                  >
                    <Stethoscope className="h-4 w-4 inline mr-2" />
                    Differential Diagnosis
                  </button>
                  <button
                    onClick={() => setActiveTab("evidence")}
                    className={`px-4 py-2 rounded-t-lg font-medium transition-colors ${
                      activeTab === "evidence"
                        ? "bg-vital-pulse/10 text-vital-pulse border-b-2 border-vital-pulse"
                        : "text-clinical-muted hover:text-clinical-light"
                    }`}
                  >
                    <Brain className="h-4 w-4 inline mr-2" />
                    Evidence Graph
                  </button>
                </div>

                {/* Tab Content */}
                <AnimatePresence mode="wait">
                  {activeTab === "diagnosis" ? (
                    <motion.div
                      key="diagnosis"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                    >
                      <DiagnosisPanel diagnosis={mockDiagnosis} />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="evidence"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                    >
                      <EvidenceGraphPanel graph={mockEvidenceGraph} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
