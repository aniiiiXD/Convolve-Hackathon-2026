"use client";

import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "motion/react";
import { ClinicalAlert } from "@/types";
import { Badge } from "@/components/ui";
import {
  AlertTriangle,
  Activity,
  Pill,
  Calendar,
  TrendingUp,
  Shield,
  Bell,
  Check,
  X,
} from "lucide-react";

interface AlertPanelProps {
  alerts: ClinicalAlert[];
  onAcknowledge?: (id: string) => void;
  onDismiss?: (id: string) => void;
  className?: string;
}

const alertIcons: Record<string, React.ElementType> = {
  critical_value: AlertTriangle,
  deterioration: Activity,
  drug_interaction: Pill,
  missed_followup: Calendar,
  trend_alert: TrendingUp,
  preventive_care: Shield,
};

const alertStyles: Record<string, string> = {
  critical: "alert-critical",
  high: "alert-warning",
  medium: "alert-info",
  low: "glass-card p-4 border-l-4 border-l-vital-pulse bg-vital-pulse/5",
};

export default function AlertPanel({
  alerts,
  onAcknowledge,
  onDismiss,
  className,
}: AlertPanelProps) {
  const sortedAlerts = [...alerts].sort((a, b) => {
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    return severityOrder[a.severity] - severityOrder[b.severity];
  });

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Bell className="h-5 w-5 text-vital-pulse" />
          <h3 className="text-lg font-semibold text-clinical-white">
            Clinical Alerts
          </h3>
          {alerts.filter((a) => !a.acknowledged).length > 0 && (
            <span className="bg-vital-critical text-white text-xs font-bold px-2 py-0.5 rounded-full">
              {alerts.filter((a) => !a.acknowledged).length}
            </span>
          )}
        </div>
      </div>

      <AnimatePresence mode="popLayout">
        {sortedAlerts.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-card p-8 text-center"
          >
            <Shield className="h-12 w-12 text-vital-pulse mx-auto mb-3 opacity-50" />
            <p className="text-clinical-muted">No active alerts</p>
            <p className="text-sm text-clinical-muted/60 mt-1">
              All monitored patients are within normal parameters
            </p>
          </motion.div>
        ) : (
          sortedAlerts.map((alert, index) => {
            const Icon = alertIcons[alert.type] || AlertTriangle;

            return (
              <motion.div
                key={alert.id}
                layout
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20, scale: 0.95 }}
                transition={{ delay: index * 0.05 }}
                className={cn(
                  alertStyles[alert.severity],
                  alert.acknowledged && "opacity-60"
                )}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={cn(
                      "p-2 rounded-lg",
                      alert.severity === "critical" && "bg-vital-critical/20",
                      alert.severity === "high" && "bg-vital-warning/20",
                      alert.severity === "medium" && "bg-vital-info/20",
                      alert.severity === "low" && "bg-vital-pulse/20"
                    )}
                  >
                    <Icon
                      className={cn(
                        "h-5 w-5",
                        alert.severity === "critical" && "text-vital-critical",
                        alert.severity === "high" && "text-vital-warning",
                        alert.severity === "medium" && "text-vital-info",
                        alert.severity === "low" && "text-vital-pulse"
                      )}
                    />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-medium text-clinical-white truncate">
                        {alert.title}
                      </h4>
                      <Badge severity={alert.severity} pulse={alert.severity === "critical"}>
                        {alert.severity}
                      </Badge>
                    </div>
                    <p className="text-sm text-clinical-muted line-clamp-2">
                      {alert.message}
                    </p>
                    <div className="flex items-center gap-4 mt-2">
                      <span className="text-xs text-clinical-muted/60 font-mono">
                        {new Date(alert.created_at).toLocaleTimeString()}
                      </span>
                      <span className="text-xs text-clinical-muted/60">
                        Patient: {alert.patient_id}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-1">
                    {!alert.acknowledged && onAcknowledge && (
                      <button
                        onClick={() => onAcknowledge(alert.id)}
                        className="p-1.5 rounded-lg text-vital-pulse hover:bg-vital-pulse/20 transition-colors"
                        title="Acknowledge"
                      >
                        <Check className="h-4 w-4" />
                      </button>
                    )}
                    {onDismiss && (
                      <button
                        onClick={() => onDismiss(alert.id)}
                        className="p-1.5 rounded-lg text-clinical-muted hover:text-vital-critical hover:bg-vital-critical/20 transition-colors"
                        title="Dismiss"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })
        )}
      </AnimatePresence>
    </div>
  );
}
