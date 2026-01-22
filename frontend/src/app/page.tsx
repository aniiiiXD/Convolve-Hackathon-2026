"use client";

import { motion } from "motion/react";
import Link from "next/link";
import {
  Activity,
  Stethoscope,
  Heart,
  Shield,
  Brain,
  Sparkles,
  ArrowRight,
  Users,
  GitBranch,
  Lightbulb,
  Database,
} from "lucide-react";

export default function LandingPage() {
  const features = [
    {
      icon: Brain,
      title: "AI-Powered Diagnosis",
      description: "Multi-stage hybrid retrieval with Discovery API for accurate differential diagnoses",
      color: "from-scan-mri to-scan-ultrasound",
    },
    {
      icon: Lightbulb,
      title: "Clinical Insights",
      description: "Temporal trends, treatment effectiveness, and predictive analytics",
      color: "from-vital-warning to-scan-thermal",
    },
    {
      icon: Shield,
      title: "Vigilance Monitoring",
      description: "Autonomous alerts for critical values, drug interactions, and deterioration",
      color: "from-vital-critical to-vital-warning",
    },
    {
      icon: GitBranch,
      title: "Evidence Graphs",
      description: "Visual reasoning chains with full citation linking for audit trails",
      color: "from-vital-pulse to-vital-info",
    },
  ];

  return (
    <div className="min-h-screen bg-midnight relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 grid-overlay opacity-30" />
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-gradient-radial from-vital-pulse/10 via-transparent to-transparent" />
      <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-gradient-radial from-scan-mri/10 via-transparent to-transparent" />

      {/* ECG Line Animation */}
      <div className="absolute top-1/2 left-0 right-0 ecg-bg h-20 opacity-20" />

      {/* Navigation */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-vital-pulse to-vital-info flex items-center justify-center">
              <Activity className="h-5 w-5 text-midnight" />
            </div>
          </div>
          <span className="text-xl font-bold text-clinical-white">
            Medi<span className="text-gradient">Sync</span>
          </span>
        </div>
        <div className="flex items-center gap-4">
          <Link
            href="/patient"
            className="px-4 py-2 text-clinical-muted hover:text-clinical-light transition-colors"
          >
            Patient Portal
          </Link>
          <Link
            href="/doctor"
            className="btn-primary flex items-center gap-2"
          >
            <Stethoscope className="h-4 w-4" />
            Doctor Portal
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 pt-20 pb-32">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 bg-vital-pulse/10 border border-vital-pulse/30 rounded-full px-4 py-1.5 mb-6">
              <Sparkles className="h-4 w-4 text-vital-pulse" />
              <span className="text-sm text-vital-pulse font-medium">
                Powered by Qdrant Vector Search
              </span>
            </div>

            <h1 className="text-5xl lg:text-6xl font-bold text-clinical-white leading-tight mb-6">
              Clinical Intelligence
              <br />
              <span className="text-gradient">Reimagined</span>
            </h1>

            <p className="text-xl text-clinical-muted leading-relaxed mb-8 max-w-xl">
              AI-powered clinical decision support with multi-stage hybrid retrieval,
              differential diagnosis generation, and autonomous vigilance monitoring.
            </p>

            <div className="flex flex-wrap gap-4">
              <Link href="/doctor" className="btn-primary flex items-center gap-2">
                Enter Doctor Portal
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link href="/patient" className="btn-secondary flex items-center gap-2">
                <Heart className="h-4 w-4" />
                Patient Companion
              </Link>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-8 mt-12 pt-8 border-t border-clinical-border/30">
              <div>
                <p className="text-3xl font-bold text-vital-pulse font-mono">4-Stage</p>
                <p className="text-sm text-clinical-muted mt-1">Retrieval Pipeline</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-vital-info font-mono">K≥20</p>
                <p className="text-sm text-clinical-muted mt-1">Privacy Anonymity</p>
              </div>
              <div>
                <p className="text-3xl font-bold text-vital-warning font-mono">Real-time</p>
                <p className="text-sm text-clinical-muted mt-1">Vigilance Alerts</p>
              </div>
            </div>
          </motion.div>

          {/* Hero Visual */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="relative hidden lg:block"
          >
            <div className="relative">
              {/* Main Card */}
              <div className="glass-card p-6 rounded-2xl">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-vital-pulse/20 flex items-center justify-center">
                    <Brain className="h-5 w-5 text-vital-pulse" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-clinical-white">Differential Diagnosis</p>
                    <p className="text-xs text-clinical-muted">Discovery API Analysis</p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="p-3 rounded-xl bg-vital-pulse/10 border border-vital-pulse/30">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-clinical-white">Acute Coronary Syndrome</span>
                      <span className="text-sm font-mono text-vital-pulse">87%</span>
                    </div>
                    <div className="h-2 rounded-full bg-midnight-50 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: "87%" }}
                        transition={{ duration: 1, delay: 0.5 }}
                        className="h-full bg-vital-pulse rounded-full"
                      />
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-midnight-50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-clinical-muted">Unstable Angina</span>
                      <span className="text-sm font-mono text-vital-info">72%</span>
                    </div>
                    <div className="h-2 rounded-full bg-midnight overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: "72%" }}
                        transition={{ duration: 1, delay: 0.7 }}
                        className="h-full bg-vital-info rounded-full"
                      />
                    </div>
                  </div>

                  <div className="p-3 rounded-xl bg-midnight-50">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-clinical-muted">Pulmonary Embolism</span>
                      <span className="text-sm font-mono text-vital-warning">45%</span>
                    </div>
                    <div className="h-2 rounded-full bg-midnight overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: "45%" }}
                        transition={{ duration: 1, delay: 0.9 }}
                        className="h-full bg-vital-warning rounded-full"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Floating Alert Card */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 }}
                className="absolute -bottom-6 -left-6 glass-card p-4 border-l-4 border-l-vital-critical"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-vital-critical/20 flex items-center justify-center">
                    <Shield className="h-4 w-4 text-vital-critical" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-clinical-white">Critical Alert</p>
                    <p className="text-xs text-vital-critical">Troponin elevated</p>
                  </div>
                </div>
              </motion.div>

              {/* Floating Insight Card */}
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.4 }}
                className="absolute -top-4 -right-4 glass-card p-3"
              >
                <div className="flex items-center gap-2">
                  <Lightbulb className="h-4 w-4 text-vital-warning" />
                  <span className="text-xs text-clinical-muted">3 insights generated</span>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl font-bold text-clinical-white mb-4">
            Cutting-Edge Clinical Features
          </h2>
          <p className="text-clinical-muted max-w-2xl mx-auto">
            Built on Qdrant&apos;s advanced vector search capabilities with native prefetch chains,
            RRF fusion, and Discovery API integration
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-card-hover p-6 group"
            >
              <div
                className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} p-0.5 mb-4`}
              >
                <div className="w-full h-full rounded-[10px] bg-midnight flex items-center justify-center">
                  <feature.icon className="h-5 w-5 text-clinical-white" />
                </div>
              </div>
              <h3 className="text-lg font-semibold text-clinical-white mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-clinical-muted">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Tech Stack Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 py-20">
        <div className="glass-card p-8 lg:p-12">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-2xl font-bold text-clinical-white mb-4">
                Powered by Modern Architecture
              </h2>
              <p className="text-clinical-muted mb-6">
                MediSync leverages Qdrant&apos;s native features for optimal performance
                and accuracy in clinical decision support.
              </p>

              <div className="space-y-4">
                {[
                  { label: "Prefetch + RRF Fusion", desc: "Single API call for hybrid search" },
                  { label: "Discovery API", desc: "Context-aware diagnosis refinement" },
                  { label: "Sparse + Dense Vectors", desc: "Medical terminology + semantic search" },
                  { label: "K-Anonymity", desc: "Privacy-preserving cohort analysis" },
                ].map((item, i) => (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.1 }}
                    className="flex items-center gap-3"
                  >
                    <div className="w-2 h-2 rounded-full bg-vital-pulse" />
                    <div>
                      <span className="text-clinical-white font-medium">{item.label}</span>
                      <span className="text-clinical-muted ml-2">— {item.desc}</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            <div className="flex justify-center">
              <div className="relative">
                <div className="w-48 h-48 rounded-2xl bg-gradient-to-br from-vital-pulse/20 to-vital-info/20 flex items-center justify-center">
                  <Database className="h-20 w-20 text-vital-pulse" />
                </div>
                <div className="absolute inset-0 rounded-2xl border border-vital-pulse/30 animate-ping opacity-30" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-8 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl font-bold text-clinical-white mb-4">
            Ready to Experience the Future of Clinical AI?
          </h2>
          <p className="text-clinical-muted mb-8 max-w-xl mx-auto">
            Join healthcare professionals using MediSync for evidence-based clinical decision support
          </p>
          <div className="flex justify-center gap-4">
            <Link href="/doctor" className="btn-primary flex items-center gap-2">
              <Stethoscope className="h-4 w-4" />
              Doctor Portal
            </Link>
            <Link href="/patient" className="btn-secondary flex items-center gap-2">
              <Heart className="h-4 w-4" />
              Patient Portal
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-clinical-border/30 py-8">
        <div className="max-w-7xl mx-auto px-8 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="h-5 w-5 text-vital-pulse" />
            <span className="text-clinical-muted">
              MediSync © 2026 — Built for Qdrant Convolve 4.0 Hackathon
            </span>
          </div>
          <div className="flex items-center gap-4 text-sm text-clinical-muted">
            <span>Powered by Qdrant</span>
            <span>•</span>
            <span>Gemini AI</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
