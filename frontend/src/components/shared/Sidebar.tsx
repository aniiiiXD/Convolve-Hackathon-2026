"use client";

import { cn } from "@/lib/utils";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";
import {
  Activity,
  Users,
  Stethoscope,
  Lightbulb,
  Bell,
  Settings,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  FileText,
  GitBranch,
} from "lucide-react";
import { useState } from "react";

const sidebarLinks = [
  {
    section: "Overview",
    items: [
      { href: "/doctor", label: "Dashboard", icon: Activity },
      { href: "/doctor/patients", label: "Patients", icon: Users },
    ],
  },
  {
    section: "Clinical Tools",
    items: [
      { href: "/doctor/diagnosis", label: "Diagnosis", icon: Stethoscope },
      { href: "/doctor/insights", label: "Insights", icon: Lightbulb },
      { href: "/doctor/evidence", label: "Evidence Graph", icon: GitBranch },
    ],
  },
  {
    section: "Monitoring",
    items: [
      { href: "/doctor/alerts", label: "Alerts", icon: Bell },
      { href: "/doctor/analytics", label: "Analytics", icon: BarChart3 },
      { href: "/doctor/records", label: "Records", icon: FileText },
    ],
  },
  {
    section: "System",
    items: [
      { href: "/doctor/settings", label: "Settings", icon: Settings },
    ],
  },
];

interface SidebarProps {
  className?: string;
}

export default function Sidebar({ className }: SidebarProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "fixed left-0 top-16 bottom-0 z-40",
        "bg-midnight-100/50 backdrop-blur-xl border-r border-clinical-border/30",
        "transition-all duration-300",
        collapsed ? "w-16" : "w-64",
        className
      )}
    >
      <div className="flex flex-col h-full py-4">
        {/* Collapse button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute -right-3 top-6 w-6 h-6 rounded-full bg-midnight-50 border border-clinical-border flex items-center justify-center text-clinical-muted hover:text-clinical-light transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="h-3 w-3" />
          ) : (
            <ChevronLeft className="h-3 w-3" />
          )}
        </button>

        {/* Navigation */}
        <nav className="flex-1 px-3 space-y-6 overflow-y-auto">
          {sidebarLinks.map((group) => (
            <div key={group.section}>
              {!collapsed && (
                <h3 className="px-3 mb-2 text-xs font-medium text-clinical-muted uppercase tracking-wider">
                  {group.section}
                </h3>
              )}
              <div className="space-y-1">
                {group.items.map((link) => {
                  const isActive = pathname === link.href;
                  const Icon = link.icon;

                  return (
                    <Link
                      key={link.href}
                      href={link.href}
                      className={cn(
                        "relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200",
                        collapsed && "justify-center",
                        isActive
                          ? "bg-vital-pulse/10 text-vital-pulse"
                          : "text-clinical-muted hover:text-clinical-light hover:bg-midnight-50"
                      )}
                      title={collapsed ? link.label : undefined}
                    >
                      <Icon className={cn("h-5 w-5 flex-shrink-0", isActive && "text-vital-pulse")} />
                      {!collapsed && (
                        <span className="text-sm font-medium">{link.label}</span>
                      )}
                      {isActive && (
                        <motion.div
                          layoutId="sidebar-indicator"
                          className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-r-full bg-vital-pulse"
                        />
                      )}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Bottom section */}
        {!collapsed && (
          <div className="px-3 pt-4 border-t border-clinical-border/30">
            <div className="glass-card p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-lg bg-vital-pulse/20 flex items-center justify-center">
                  <Activity className="h-4 w-4 text-vital-pulse" />
                </div>
                <div>
                  <p className="text-sm font-medium text-clinical-white">
                    System Status
                  </p>
                  <p className="text-xs text-vital-pulse">All systems operational</p>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-clinical-muted">API Health</span>
                  <span className="text-vital-pulse">100%</span>
                </div>
                <div className="h-1.5 rounded-full bg-midnight-50 overflow-hidden">
                  <div className="h-full w-full bg-vital-pulse rounded-full" />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
