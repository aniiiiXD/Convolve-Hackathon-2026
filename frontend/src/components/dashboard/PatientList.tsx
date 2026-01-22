"use client";

import { cn } from "@/lib/utils";
import { motion } from "motion/react";
import { Patient } from "@/types";
import { Badge } from "@/components/ui";
import {
  User,
  Calendar,
  Activity,
  ChevronRight,
  AlertCircle,
  Search,
} from "lucide-react";

interface PatientListProps {
  patients: Patient[];
  selectedId?: string;
  onSelect?: (patient: Patient) => void;
  className?: string;
}

export default function PatientList({
  patients,
  selectedId,
  onSelect,
  className,
}: PatientListProps) {
  const statusStyles = {
    stable: {
      badge: "badge-stable",
      dot: "bg-vital-pulse",
    },
    monitoring: {
      badge: "badge-warning",
      dot: "bg-vital-warning",
    },
    critical: {
      badge: "badge-critical",
      dot: "bg-vital-critical animate-ping",
    },
  };

  return (
    <div className={cn("space-y-2", className)}>
      {patients.length === 0 ? (
        <div className="glass-card p-8 text-center">
          <Search className="h-12 w-12 text-clinical-muted mx-auto mb-3 opacity-50" />
          <p className="text-clinical-muted">No patients found</p>
        </div>
      ) : (
        patients.map((patient, index) => (
          <motion.div
            key={patient.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            whileHover={{ x: 4 }}
            onClick={() => onSelect?.(patient)}
            className={cn(
              "glass-card p-4 cursor-pointer transition-all duration-200",
              selectedId === patient.id
                ? "border-vital-pulse/50 shadow-glow-teal bg-vital-pulse/5"
                : "hover:border-clinical-border"
            )}
          >
            <div className="flex items-center gap-4">
              {/* Avatar */}
              <div className="relative">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-midnight-50 to-slate-850 flex items-center justify-center border border-clinical-border/50">
                  <User className="h-6 w-6 text-clinical-muted" />
                </div>
                {/* Status indicator */}
                <div className="absolute -bottom-0.5 -right-0.5">
                  <div className="relative">
                    <div
                      className={cn(
                        "w-3 h-3 rounded-full",
                        statusStyles[patient.status].dot
                      )}
                    />
                    {patient.status === "critical" && (
                      <div className="absolute inset-0 w-3 h-3 rounded-full bg-vital-critical animate-ping" />
                    )}
                  </div>
                </div>
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h4 className="font-medium text-clinical-white truncate">
                    {patient.name}
                  </h4>
                  <Badge severity={patient.status === "critical" ? "critical" : patient.status === "monitoring" ? "warning" : "stable"}>
                    {patient.status}
                  </Badge>
                </div>
                <div className="flex items-center gap-4 text-sm text-clinical-muted">
                  <span className="flex items-center gap-1">
                    <User className="h-3 w-3" />
                    {patient.age}y, {patient.gender}
                  </span>
                  <span className="flex items-center gap-1">
                    <Calendar className="h-3 w-3" />
                    {new Date(patient.last_visit).toLocaleDateString()}
                  </span>
                </div>
                {patient.chief_complaint && (
                  <p className="text-sm text-clinical-muted/80 mt-1 truncate">
                    <Activity className="h-3 w-3 inline mr-1" />
                    {patient.chief_complaint}
                  </p>
                )}
              </div>

              {/* Conditions count */}
              {patient.conditions.length > 0 && (
                <div className="text-right">
                  <div className="flex items-center gap-1 text-vital-warning">
                    <AlertCircle className="h-4 w-4" />
                    <span className="text-sm font-mono">{patient.conditions.length}</span>
                  </div>
                  <span className="text-xs text-clinical-muted">conditions</span>
                </div>
              )}

              <ChevronRight className="h-5 w-5 text-clinical-muted" />
            </div>

            {/* Conditions tags */}
            {patient.conditions.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5 pl-16">
                {patient.conditions.slice(0, 3).map((condition, i) => (
                  <span
                    key={i}
                    className="text-xs bg-midnight-50 text-clinical-muted px-2 py-0.5 rounded"
                  >
                    {condition}
                  </span>
                ))}
                {patient.conditions.length > 3 && (
                  <span className="text-xs text-clinical-muted">
                    +{patient.conditions.length - 3} more
                  </span>
                )}
              </div>
            )}
          </motion.div>
        ))
      )}
    </div>
  );
}
