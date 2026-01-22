"use client";

import { motion } from "motion/react";
import { Card, CardHeader, VitalSign, Badge, Button } from "@/components/ui";
import { AlertPanel, PatientList, InsightsPanel } from "@/components/dashboard";
import {
  Activity,
  Users,
  AlertTriangle,
  TrendingUp,
  Clock,
  Stethoscope,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";

// Mock data for demonstration
const mockPatients = [
  {
    id: "P-101",
    name: "John Smith",
    age: 45,
    gender: "Male",
    chief_complaint: "Chest pain and shortness of breath",
    status: "critical" as const,
    last_visit: "2026-01-22",
    conditions: ["Hypertension", "Type 2 Diabetes", "CAD"],
  },
  {
    id: "P-102",
    name: "Sarah Johnson",
    age: 32,
    gender: "Female",
    chief_complaint: "Recurring headaches",
    status: "monitoring" as const,
    last_visit: "2026-01-21",
    conditions: ["Migraine"],
  },
  {
    id: "P-103",
    name: "Michael Chen",
    age: 58,
    gender: "Male",
    chief_complaint: "Follow-up for diabetes management",
    status: "stable" as const,
    last_visit: "2026-01-20",
    conditions: ["Type 2 Diabetes", "Hyperlipidemia"],
  },
  {
    id: "P-104",
    name: "Emily Davis",
    age: 67,
    gender: "Female",
    chief_complaint: "Post-operative care",
    status: "monitoring" as const,
    last_visit: "2026-01-22",
    conditions: ["Hip Replacement", "Osteoarthritis"],
  },
];

const mockAlerts = [
  {
    id: "A-001",
    type: "critical_value" as const,
    severity: "critical" as const,
    title: "Critical Potassium Level",
    message: "Patient P-101 has potassium level of 6.8 mEq/L. Immediate intervention required.",
    patient_id: "P-101",
    created_at: new Date().toISOString(),
    acknowledged: false,
    evidence_ids: ["lab-001"],
  },
  {
    id: "A-002",
    type: "drug_interaction" as const,
    severity: "high" as const,
    title: "Potential Drug Interaction",
    message: "Warfarin and Aspirin combination detected for patient P-103. Review anticoagulation therapy.",
    patient_id: "P-103",
    created_at: new Date(Date.now() - 3600000).toISOString(),
    acknowledged: false,
    evidence_ids: ["rx-001", "rx-002"],
  },
  {
    id: "A-003",
    type: "trend_alert" as const,
    severity: "medium" as const,
    title: "Rising HbA1c Trend",
    message: "Patient P-103 shows 0.5% increase in HbA1c over 3 months. Consider treatment adjustment.",
    patient_id: "P-103",
    created_at: new Date(Date.now() - 7200000).toISOString(),
    acknowledged: true,
    evidence_ids: ["lab-002", "lab-003"],
  },
];

const mockInsights = [
  {
    id: "I-001",
    insight_type: "risk_pattern" as const,
    title: "Elevated Cardiovascular Risk",
    description: "Based on recent lab trends and vital signs, patient P-101 shows increased risk factors for acute coronary syndrome.",
    confidence: 0.87,
    supporting_data: {},
    evidence_ids: ["lab-001", "vital-001"],
    actionable: true,
    recommended_actions: ["Order cardiac enzymes", "Schedule stress test", "Review medication compliance"],
    generated_at: new Date().toISOString(),
  },
  {
    id: "I-002",
    insight_type: "treatment_effectiveness" as const,
    title: "Metformin Response Analysis",
    description: "Patient P-103 showing positive response to Metformin 1000mg BID with 15% reduction in fasting glucose.",
    confidence: 0.92,
    supporting_data: {},
    evidence_ids: ["lab-004", "lab-005"],
    actionable: false,
    recommended_actions: [],
    generated_at: new Date().toISOString(),
  },
  {
    id: "I-003",
    insight_type: "temporal_trend" as const,
    title: "Blood Pressure Improvement",
    description: "Patient P-102 showing consistent BP reduction over 30 days following lifestyle modifications.",
    confidence: 0.78,
    supporting_data: {},
    evidence_ids: ["vital-002", "vital-003"],
    actionable: false,
    recommended_actions: [],
    generated_at: new Date().toISOString(),
  },
];

export default function DoctorDashboard() {
  const stats = [
    {
      label: "Active Patients",
      value: "24",
      icon: Users,
      trend: "+3 this week",
      color: "text-vital-pulse",
    },
    {
      label: "Critical Alerts",
      value: "2",
      icon: AlertTriangle,
      trend: "Requires attention",
      color: "text-vital-critical",
    },
    {
      label: "Pending Reviews",
      value: "8",
      icon: Clock,
      trend: "3 urgent",
      color: "text-vital-warning",
    },
    {
      label: "Diagnoses Today",
      value: "12",
      icon: Stethoscope,
      trend: "+15% accuracy",
      color: "text-vital-info",
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-3xl font-bold text-clinical-white"
          >
            Good morning, <span className="text-gradient">Dr. Strange</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="text-clinical-muted mt-1"
          >
            Here&apos;s your clinical overview for today
          </motion.p>
        </div>
        <Link href="/doctor/diagnosis">
          <Button variant="primary">
            <Stethoscope className="h-4 w-4 mr-2" />
            New Diagnosis
          </Button>
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="p-5">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-clinical-muted uppercase tracking-wider">
                    {stat.label}
                  </p>
                  <p className={`text-4xl font-bold font-mono mt-2 ${stat.color}`}>
                    {stat.value}
                  </p>
                  <p className="text-xs text-clinical-muted mt-1 flex items-center gap-1">
                    <TrendingUp className="h-3 w-3" />
                    {stat.trend}
                  </p>
                </div>
                <div className={`p-3 rounded-xl bg-midnight-50 ${stat.color}`}>
                  <stat.icon className="h-6 w-6" />
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Alerts Panel */}
        <div className="lg:col-span-1">
          <AlertPanel
            alerts={mockAlerts}
            onAcknowledge={(id) => console.log("Acknowledge:", id)}
            onDismiss={(id) => console.log("Dismiss:", id)}
          />
        </div>

        {/* Patients & Insights */}
        <div className="lg:col-span-2 space-y-6">
          {/* Recent Patients */}
          <Card className="p-6">
            <CardHeader
              title="Recent Patients"
              subtitle="Patients requiring attention"
              action={
                <Link href="/doctor/patients" className="text-vital-pulse text-sm flex items-center gap-1 hover:underline">
                  View all <ArrowRight className="h-4 w-4" />
                </Link>
              }
            />
            <PatientList
              patients={mockPatients}
              onSelect={(patient) => console.log("Selected:", patient)}
            />
          </Card>

          {/* Insights */}
          <InsightsPanel
            insights={mockInsights}
            onInsightClick={(insight) => console.log("Insight:", insight)}
          />
        </div>
      </div>

      {/* Quick Vitals Section */}
      <div>
        <h3 className="text-lg font-semibold text-clinical-white mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 text-vital-pulse" />
          Patient P-101 - Live Vitals
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <VitalSign
            label="Heart Rate"
            value="92"
            unit="bpm"
            trend="up"
            status="warning"
          />
          <VitalSign
            label="Blood Pressure"
            value="158/95"
            unit="mmHg"
            trend="up"
            status="critical"
          />
          <VitalSign
            label="Temperature"
            value="37.2"
            unit="°C"
            status="normal"
          />
          <VitalSign
            label="SpO2"
            value="96"
            unit="%"
            trend="stable"
            status="normal"
          />
          <VitalSign
            label="Resp Rate"
            value="18"
            unit="/min"
            status="normal"
          />
        </div>
      </div>
    </div>
  );
}
