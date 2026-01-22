"use client";

import { useState } from "react";
import { motion } from "motion/react";
import { Card, CardHeader, VitalSign, Badge, Button, ConfidenceMeter } from "@/components/ui";
import {
  Heart,
  Activity,
  Calendar,
  Pill,
  TrendingUp,
  TrendingDown,
  Clock,
  FileText,
  Lightbulb,
  ChevronRight,
  Plus,
  Send,
  Sparkles,
} from "lucide-react";
import Link from "next/link";

export default function PatientDashboard() {
  const [symptomInput, setSymptomInput] = useState("");

  const healthScore = 78;
  const recentSymptoms = [
    { date: "Jan 22", symptom: "Mild headache", severity: "low" },
    { date: "Jan 20", symptom: "Fatigue after exercise", severity: "low" },
    { date: "Jan 18", symptom: "Slight chest discomfort", severity: "medium" },
  ];

  const medications = [
    { name: "Metformin", dose: "500mg", frequency: "Twice daily", nextDose: "2:00 PM" },
    { name: "Lisinopril", dose: "10mg", frequency: "Once daily", nextDose: "Tomorrow" },
    { name: "Aspirin", dose: "81mg", frequency: "Once daily", nextDose: "Tomorrow" },
  ];

  const upcomingAppointments = [
    { date: "Jan 28, 2026", time: "10:00 AM", doctor: "Dr. Strange", type: "Follow-up" },
    { date: "Feb 15, 2026", time: "2:30 PM", doctor: "Dr. Palmer", type: "Annual Checkup" },
  ];

  const insights = [
    {
      type: "positive",
      title: "Blood Pressure Improving",
      description: "Your average BP has decreased by 8% over the last month",
      icon: TrendingDown,
    },
    {
      type: "info",
      title: "Medication Adherence",
      description: "You've taken 95% of your medications on time this week",
      icon: Pill,
    },
    {
      type: "action",
      title: "Activity Goal",
      description: "You're 2,000 steps away from your daily goal",
      icon: Activity,
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-3xl font-bold text-clinical-white"
        >
          Welcome back, <span className="text-gradient">John</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-clinical-muted mt-1"
        >
          Here&apos;s your health overview for today
        </motion.p>
      </div>

      {/* Health Score Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card glow="teal" className="p-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-radial from-vital-pulse/10 to-transparent" />
          <div className="relative flex items-center justify-between">
            <div>
              <p className="text-sm text-clinical-muted uppercase tracking-wider mb-2">
                Overall Health Score
              </p>
              <div className="flex items-baseline gap-2">
                <span className="text-6xl font-bold text-vital-pulse font-mono">
                  {healthScore}
                </span>
                <span className="text-2xl text-clinical-muted">/100</span>
              </div>
              <div className="flex items-center gap-2 mt-2">
                <TrendingUp className="h-4 w-4 text-vital-pulse" />
                <span className="text-sm text-vital-pulse">+5 from last week</span>
              </div>
            </div>
            <div className="w-32 h-32 relative">
              <svg className="transform -rotate-90 w-32 h-32">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  className="text-midnight-50"
                />
                <motion.circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  strokeLinecap="round"
                  className="text-vital-pulse"
                  initial={{ strokeDasharray: "0 352" }}
                  animate={{ strokeDasharray: `${(healthScore / 100) * 352} 352` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <Heart className="h-10 w-10 text-vital-pulse" />
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Vitals Grid */}
      <div>
        <h3 className="text-lg font-semibold text-clinical-white mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 text-vital-pulse" />
          Today&apos;s Vitals
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <VitalSign
            label="Heart Rate"
            value="72"
            unit="bpm"
            status="normal"
          />
          <VitalSign
            label="Blood Pressure"
            value="125/82"
            unit="mmHg"
            trend="down"
            status="normal"
          />
          <VitalSign
            label="Blood Sugar"
            value="105"
            unit="mg/dL"
            status="normal"
          />
          <VitalSign
            label="Weight"
            value="176"
            unit="lbs"
            trend="down"
            status="normal"
          />
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Symptom Logger */}
        <Card className="p-6">
          <CardHeader
            title="Log a Symptom"
            subtitle="Track how you're feeling"
          />
          <div className="space-y-4">
            <div className="relative">
              <textarea
                value={symptomInput}
                onChange={(e) => setSymptomInput(e.target.value)}
                placeholder="Describe how you're feeling..."
                className="w-full h-24 bg-midnight-50/80 border border-clinical-border/50 rounded-xl px-4 py-3 text-clinical-light placeholder:text-clinical-muted/50 focus:outline-none focus:border-vital-pulse/50 focus:ring-2 focus:ring-vital-pulse/20 transition-all resize-none"
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex gap-2">
                {["Headache", "Fatigue", "Pain"].map((quick) => (
                  <button
                    key={quick}
                    onClick={() => setSymptomInput(quick)}
                    className="px-3 py-1 text-xs bg-midnight-50 text-clinical-muted rounded-full hover:bg-midnight-50/80 transition-colors"
                  >
                    {quick}
                  </button>
                ))}
              </div>
              <Button variant="primary" size="sm">
                <Send className="h-4 w-4 mr-1" />
                Log
              </Button>
            </div>
          </div>

          {/* Recent Symptoms */}
          <div className="mt-6 pt-4 border-t border-clinical-border/30">
            <p className="text-sm text-clinical-muted mb-3">Recent Entries</p>
            <div className="space-y-2">
              {recentSymptoms.map((symptom, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-2 rounded-lg bg-midnight-50/50"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-clinical-muted font-mono">
                      {symptom.date}
                    </span>
                    <span className="text-sm text-clinical-light">
                      {symptom.symptom}
                    </span>
                  </div>
                  <Badge severity={symptom.severity === "medium" ? "warning" : "stable"}>
                    {symptom.severity}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Medications */}
        <Card className="p-6">
          <CardHeader
            title="Medications"
            subtitle="Your daily prescriptions"
            action={
              <Link href="/patient/records" className="text-vital-pulse text-sm hover:underline">
                View all
              </Link>
            }
          />
          <div className="space-y-3">
            {medications.map((med, i) => (
              <motion.div
                key={med.name}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1 }}
                className="flex items-center justify-between p-3 rounded-xl bg-midnight-50/50 hover:bg-midnight-50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-vital-info/20 flex items-center justify-center">
                    <Pill className="h-5 w-5 text-vital-info" />
                  </div>
                  <div>
                    <p className="font-medium text-clinical-white">{med.name}</p>
                    <p className="text-xs text-clinical-muted">
                      {med.dose} • {med.frequency}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-clinical-muted">Next dose</p>
                  <p className="text-sm font-mono text-vital-pulse">{med.nextDose}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      </div>

      {/* Insights Section */}
      <div>
        <h3 className="text-lg font-semibold text-clinical-white mb-4 flex items-center gap-2">
          <Lightbulb className="h-5 w-5 text-vital-warning" />
          Health Insights
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          {insights.map((insight, i) => (
            <motion.div
              key={insight.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.1 }}
            >
              <Card variant="hover" className="p-5 h-full">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center mb-3 ${
                  insight.type === "positive"
                    ? "bg-vital-pulse/20"
                    : insight.type === "info"
                    ? "bg-vital-info/20"
                    : "bg-vital-warning/20"
                }`}>
                  <insight.icon className={`h-5 w-5 ${
                    insight.type === "positive"
                      ? "text-vital-pulse"
                      : insight.type === "info"
                      ? "text-vital-info"
                      : "text-vital-warning"
                  }`} />
                </div>
                <h4 className="font-medium text-clinical-white mb-1">
                  {insight.title}
                </h4>
                <p className="text-sm text-clinical-muted">
                  {insight.description}
                </p>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Appointments */}
      <Card className="p-6">
        <CardHeader
          title="Upcoming Appointments"
          subtitle="Your scheduled visits"
          action={
            <Button variant="ghost" size="sm">
              <Plus className="h-4 w-4 mr-1" />
              Book New
            </Button>
          }
        />
        <div className="space-y-3">
          {upcomingAppointments.map((apt, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-center justify-between p-4 rounded-xl bg-midnight-50/50 hover:bg-midnight-50 transition-colors cursor-pointer group"
            >
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-scan-mri/20 flex items-center justify-center">
                  <Calendar className="h-6 w-6 text-scan-mri" />
                </div>
                <div>
                  <p className="font-medium text-clinical-white">{apt.type}</p>
                  <p className="text-sm text-clinical-muted">with {apt.doctor}</p>
                </div>
              </div>
              <div className="text-right flex items-center gap-4">
                <div>
                  <p className="text-sm font-medium text-clinical-white">{apt.date}</p>
                  <p className="text-sm text-clinical-muted">{apt.time}</p>
                </div>
                <ChevronRight className="h-5 w-5 text-clinical-muted opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* AI Assistant Prompt */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <Card glow="teal" className="p-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-vital-pulse to-vital-info flex items-center justify-center">
              <Sparkles className="h-6 w-6 text-midnight" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-clinical-white">
                Ask MediSync AI
              </h3>
              <p className="text-sm text-clinical-muted">
                Have questions about your health? Our AI companion can help explain your records and provide guidance.
              </p>
            </div>
            <Link href="/patient/insights">
              <Button variant="secondary">
                Start Conversation
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
          </div>
        </Card>
      </motion.div>
    </div>
  );
}
