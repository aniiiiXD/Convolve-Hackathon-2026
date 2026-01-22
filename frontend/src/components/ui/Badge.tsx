"use client";

import { cn, getSeverityBadgeClass } from "@/lib/utils";
import { AlertSeverity } from "@/types";

interface BadgeProps {
  severity?: AlertSeverity | "stable" | "info" | "warning";
  variant?: "default" | "outline" | "solid";
  children: React.ReactNode;
  className?: string;
  pulse?: boolean;
}

export default function Badge({
  severity = "info",
  variant = "default",
  children,
  className,
  pulse = false,
}: BadgeProps) {
  const baseClasses = getSeverityBadgeClass(severity);

  return (
    <span
      className={cn(
        baseClasses,
        pulse && "animate-pulse-vital",
        className
      )}
    >
      {pulse && (
        <span className="relative flex h-2 w-2 mr-1.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-current" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-current" />
        </span>
      )}
      {children}
    </span>
  );
}
