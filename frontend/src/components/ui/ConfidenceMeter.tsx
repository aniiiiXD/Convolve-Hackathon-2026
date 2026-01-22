"use client";

import { cn, getConfidenceColor } from "@/lib/utils";
import { motion } from "motion/react";

interface ConfidenceMeterProps {
  value: number; // 0-1
  label?: string;
  showPercentage?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
}

export default function ConfidenceMeter({
  value,
  label,
  showPercentage = true,
  size = "md",
  className,
}: ConfidenceMeterProps) {
  const percentage = Math.round(value * 100);
  const colorClass = getConfidenceColor(value);

  const heights = {
    sm: "h-1",
    md: "h-2",
    lg: "h-3",
  };

  return (
    <div className={cn("w-full", className)}>
      {(label || showPercentage) && (
        <div className="flex justify-between items-center mb-1.5">
          {label && (
            <span className="text-sm text-clinical-muted">{label}</span>
          )}
          {showPercentage && (
            <span className="text-sm font-mono text-clinical-light">
              {percentage}%
            </span>
          )}
        </div>
      )}
      <div className={cn("confidence-bar", heights[size])}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={cn("confidence-fill", colorClass)}
        />
      </div>
    </div>
  );
}
