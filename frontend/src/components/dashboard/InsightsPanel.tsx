"use client";

import { cn } from "@/lib/utils";
import { motion } from "motion/react";
import { GeneratedInsight, InsightType } from "@/types";
import { Card, Badge, ConfidenceMeter } from "@/components/ui";
import {
  TrendingUp,
  Target,
  Users,
  AlertCircle,
  Link2,
  Zap,
  Clock,
  Lightbulb,
  ChevronRight,
} from "lucide-react";

interface InsightsPanelProps {
  insights: GeneratedInsight[];
  className?: string;
  onInsightClick?: (insight: GeneratedInsight) => void;
}

const insightIcons: Record<InsightType, React.ElementType> = {
  temporal_trend: TrendingUp,
  treatment_effectiveness: Target,
  cohort_comparison: Users,
  risk_pattern: AlertCircle,
  correlation: Link2,
  anomaly: Zap,
  prediction: Clock,
};

const insightColors: Record<InsightType, { bg: string; text: string; border: string }> = {
  temporal_trend: {
    bg: "bg-scan-ultrasound/10",
    text: "text-scan-ultrasound",
    border: "border-scan-ultrasound/30",
  },
  treatment_effectiveness: {
    bg: "bg-vital-pulse/10",
    text: "text-vital-pulse",
    border: "border-vital-pulse/30",
  },
  cohort_comparison: {
    bg: "bg-scan-mri/10",
    text: "text-scan-mri",
    border: "border-scan-mri/30",
  },
  risk_pattern: {
    bg: "bg-vital-warning/10",
    text: "text-vital-warning",
    border: "border-vital-warning/30",
  },
  correlation: {
    bg: "bg-vital-info/10",
    text: "text-vital-info",
    border: "border-vital-info/30",
  },
  anomaly: {
    bg: "bg-vital-critical/10",
    text: "text-vital-critical",
    border: "border-vital-critical/30",
  },
  prediction: {
    bg: "bg-scan-thermal/10",
    text: "text-scan-thermal",
    border: "border-scan-thermal/30",
  },
};

export default function InsightsPanel({
  insights,
  className,
  onInsightClick,
}: InsightsPanelProps) {
  const actionableInsights = insights.filter((i) => i.actionable);
  const informationalInsights = insights.filter((i) => !i.actionable);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Lightbulb className="h-5 w-5 text-vital-warning" />
          <h3 className="text-lg font-semibold text-clinical-white">
            Clinical Insights
          </h3>
          <span className="text-sm text-clinical-muted">
            {insights.length} generated
          </span>
        </div>
      </div>

      {/* Actionable Insights */}
      {actionableInsights.length > 0 && (
        <div>
          <h4 className="text-xs text-clinical-muted uppercase tracking-wider mb-3 flex items-center gap-2">
            <Zap className="h-3 w-3 text-vital-warning" />
            Actionable Insights
          </h4>
          <div className="grid gap-3">
            {actionableInsights.map((insight, index) => (
              <InsightCard
                key={insight.id}
                insight={insight}
                index={index}
                onClick={() => onInsightClick?.(insight)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Informational Insights */}
      {informationalInsights.length > 0 && (
        <div>
          <h4 className="text-xs text-clinical-muted uppercase tracking-wider mb-3">
            Observations
          </h4>
          <div className="grid gap-3">
            {informationalInsights.map((insight, index) => (
              <InsightCard
                key={insight.id}
                insight={insight}
                index={index + actionableInsights.length}
                onClick={() => onInsightClick?.(insight)}
                compact
              />
            ))}
          </div>
        </div>
      )}

      {insights.length === 0 && (
        <Card className="p-8 text-center">
          <Lightbulb className="h-12 w-12 text-clinical-muted mx-auto mb-3 opacity-50" />
          <p className="text-clinical-muted">No insights available</p>
          <p className="text-sm text-clinical-muted/60 mt-1">
            Add more patient data to generate clinical insights
          </p>
        </Card>
      )}
    </div>
  );
}

interface InsightCardProps {
  insight: GeneratedInsight;
  index: number;
  onClick?: () => void;
  compact?: boolean;
}

function InsightCard({ insight, index, onClick, compact }: InsightCardProps) {
  const Icon = insightIcons[insight.insight_type] || Lightbulb;
  const colors = insightColors[insight.insight_type];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ scale: 1.01 }}
      onClick={onClick}
      className={cn(
        "glass-card-hover p-4 cursor-pointer border-l-4",
        colors.border
      )}
    >
      <div className="flex items-start gap-3">
        <div className={cn("p-2 rounded-lg", colors.bg)}>
          <Icon className={cn("h-4 w-4", colors.text)} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={cn("text-xs uppercase tracking-wider font-medium", colors.text)}>
              {insight.insight_type.replace("_", " ")}
            </span>
            {insight.actionable && (
              <Badge severity="warning">Action Needed</Badge>
            )}
          </div>

          <h4 className="font-medium text-clinical-white mb-1 line-clamp-1">
            {insight.title}
          </h4>

          {!compact && (
            <p className="text-sm text-clinical-muted line-clamp-2 mb-3">
              {insight.description}
            </p>
          )}

          <div className="flex items-center justify-between">
            <ConfidenceMeter
              value={insight.confidence}
              size="sm"
              showPercentage={false}
              className="w-24"
            />
            <span className="text-xs text-clinical-muted font-mono">
              {Math.round(insight.confidence * 100)}% confidence
            </span>
          </div>

          {!compact && insight.recommended_actions.length > 0 && (
            <div className="mt-3 pt-3 border-t border-clinical-border/30">
              <p className="text-xs text-clinical-muted mb-2">Recommended:</p>
              <ul className="space-y-1">
                {insight.recommended_actions.slice(0, 2).map((action, i) => (
                  <li
                    key={i}
                    className="text-sm text-clinical-light flex items-center gap-2"
                  >
                    <ChevronRight className="h-3 w-3 text-vital-pulse" />
                    {action}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
