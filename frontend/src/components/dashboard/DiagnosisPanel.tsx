"use client";

import { cn } from "@/lib/utils";
import { motion } from "motion/react";
import { DifferentialResult, DiagnosticCandidate } from "@/types";
import { Card, CardHeader, ConfidenceMeter, Badge } from "@/components/ui";
import {
  Stethoscope,
  AlertTriangle,
  ClipboardList,
  ChevronRight,
  Brain,
} from "lucide-react";

interface DiagnosisPanelProps {
  diagnosis: DifferentialResult;
  className?: string;
}

export default function DiagnosisPanel({
  diagnosis,
  className,
}: DiagnosisPanelProps) {
  return (
    <div className={cn("space-y-4", className)}>
      {/* Primary Diagnosis */}
      <Card variant="hover" glow="teal" className="p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-vital-pulse/20">
            <Stethoscope className="h-6 w-6 text-vital-pulse" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs text-vital-pulse uppercase tracking-wider font-medium">
                Primary Diagnosis
              </span>
              <Badge severity="stable">Most Likely</Badge>
            </div>
            <h3 className="text-xl font-bold text-clinical-white mb-3">
              {diagnosis.primary_diagnosis}
            </h3>
            <ConfidenceMeter
              value={diagnosis.primary_confidence}
              label="Confidence Score"
              size="md"
            />
          </div>
        </div>
      </Card>

      {/* Red Flags */}
      {diagnosis.red_flags.length > 0 && (
        <Card className="p-5 border-l-4 border-l-vital-critical bg-vital-critical/5">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-vital-critical flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-vital-critical mb-2">
                Red Flags - Immediate Attention Required
              </h4>
              <ul className="space-y-1.5">
                {diagnosis.red_flags.map((flag, index) => (
                  <motion.li
                    key={index}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="text-sm text-clinical-light flex items-center gap-2"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-vital-critical" />
                    {flag}
                  </motion.li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      )}

      {/* Alternative Diagnoses */}
      {diagnosis.alternatives.length > 0 && (
        <div>
          <h4 className="text-sm text-clinical-muted uppercase tracking-wider mb-3 flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Differential Diagnoses
          </h4>
          <div className="space-y-2">
            {diagnosis.alternatives.map((alt, index) => (
              <DiagnosisCard key={index} candidate={alt} rank={index + 2} />
            ))}
          </div>
        </div>
      )}

      {/* Recommended Workup */}
      {diagnosis.recommended_workup.length > 0 && (
        <Card className="p-5">
          <CardHeader
            title="Recommended Workup"
            subtitle="Suggested diagnostic tests"
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {diagnosis.recommended_workup.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-center gap-2 p-3 rounded-lg bg-midnight-50/50 hover:bg-midnight-50 transition-colors cursor-pointer group"
              >
                <ClipboardList className="h-4 w-4 text-vital-info" />
                <span className="text-sm text-clinical-light flex-1">{item}</span>
                <ChevronRight className="h-4 w-4 text-clinical-muted opacity-0 group-hover:opacity-100 transition-opacity" />
              </motion.div>
            ))}
          </div>
        </Card>
      )}

      {/* Clinical Reasoning */}
      {diagnosis.reasoning && (
        <Card className="p-5">
          <CardHeader
            title="Clinical Reasoning"
            subtitle="AI-generated analysis"
          />
          <p className="text-sm text-clinical-muted leading-relaxed">
            {diagnosis.reasoning}
          </p>
        </Card>
      )}
    </div>
  );
}

interface DiagnosisCardProps {
  candidate: DiagnosticCandidate;
  rank: number;
}

function DiagnosisCard({ candidate, rank }: DiagnosisCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      whileHover={{ x: 4 }}
      className="diagnosis-card"
    >
      <div className="flex items-center gap-4">
        <div className="w-8 h-8 rounded-full bg-midnight-50 flex items-center justify-center text-sm font-mono text-clinical-muted">
          #{rank}
        </div>
        <div className="flex-1">
          <h5 className="font-medium text-clinical-white mb-1">
            {candidate.diagnosis}
          </h5>
          <ConfidenceMeter
            value={candidate.confidence}
            showPercentage
            size="sm"
          />
        </div>
        <ChevronRight className="h-5 w-5 text-clinical-muted" />
      </div>

      {candidate.supporting_evidence.length > 0 && (
        <div className="mt-3 pl-12">
          <p className="text-xs text-clinical-muted uppercase tracking-wider mb-1.5">
            Supporting Evidence
          </p>
          <div className="flex flex-wrap gap-1.5">
            {candidate.supporting_evidence.slice(0, 3).map((evidence, i) => (
              <span
                key={i}
                className="text-xs bg-midnight-50 text-clinical-light px-2 py-1 rounded"
              >
                {evidence}
              </span>
            ))}
            {candidate.supporting_evidence.length > 3 && (
              <span className="text-xs text-clinical-muted">
                +{candidate.supporting_evidence.length - 3} more
              </span>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
