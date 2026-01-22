"use client";

import { cn } from "@/lib/utils";
import { motion } from "motion/react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface VitalSignProps {
  label: string;
  value: string | number;
  unit: string;
  trend?: "up" | "down" | "stable";
  status?: "normal" | "warning" | "critical";
  className?: string;
}

export default function VitalSign({
  label,
  value,
  unit,
  trend,
  status = "normal",
  className,
}: VitalSignProps) {
  const statusColors = {
    normal: "text-vital-pulse",
    warning: "text-vital-warning",
    critical: "text-vital-critical",
  };

  const statusGlow = {
    normal: "",
    warning: "shadow-glow-amber",
    critical: "shadow-glow-coral animate-pulse-vital",
  };

  const TrendIcon = trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;
  const trendColor = trend === "up" ? "text-vital-critical" : trend === "down" ? "text-vital-pulse" : "text-clinical-muted";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        "glass-card p-4 relative overflow-hidden",
        statusGlow[status],
        className
      )}
    >
      {/* ECG line background for critical */}
      {status === "critical" && (
        <div className="absolute inset-0 ecg-bg opacity-30" />
      )}

      <div className="relative z-10">
        <p className="text-xs text-clinical-muted uppercase tracking-wider mb-1">
          {label}
        </p>
        <div className="flex items-baseline gap-1">
          <span className={cn("text-3xl font-bold font-mono", statusColors[status])}>
            {value}
          </span>
          <span className="text-sm text-clinical-muted">{unit}</span>
        </div>
        {trend && (
          <div className={cn("flex items-center gap-1 mt-2", trendColor)}>
            <TrendIcon className="h-3 w-3" />
            <span className="text-xs font-mono">
              {trend === "stable" ? "Stable" : trend === "up" ? "Rising" : "Falling"}
            </span>
          </div>
        )}
      </div>

      {/* Status indicator dot */}
      <div
        className={cn(
          "absolute top-3 right-3 w-2 h-2 rounded-full",
          status === "normal" && "bg-vital-pulse",
          status === "warning" && "bg-vital-warning",
          status === "critical" && "bg-vital-critical animate-ping"
        )}
      />
    </motion.div>
  );
}
