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
  Search,
  User,
  LogOut,
  Heart,
} from "lucide-react";

interface NavigationProps {
  userRole: "doctor" | "patient";
  userName: string;
  className?: string;
}

const doctorLinks = [
  { href: "/doctor", label: "Dashboard", icon: Activity },
  { href: "/doctor/patients", label: "Patients", icon: Users },
  { href: "/doctor/diagnosis", label: "Diagnosis", icon: Stethoscope },
  { href: "/doctor/insights", label: "Insights", icon: Lightbulb },
  { href: "/doctor/alerts", label: "Alerts", icon: Bell },
];

const patientLinks = [
  { href: "/patient", label: "My Health", icon: Heart },
  { href: "/patient/records", label: "Records", icon: Activity },
  { href: "/patient/insights", label: "Insights", icon: Lightbulb },
];

export default function Navigation({
  userRole,
  userName,
  className,
}: NavigationProps) {
  const pathname = usePathname();
  const links = userRole === "doctor" ? doctorLinks : patientLinks;

  return (
    <nav
      className={cn(
        "fixed top-0 left-0 right-0 z-50",
        "bg-midnight/80 backdrop-blur-xl border-b border-clinical-border/30",
        className
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href={userRole === "doctor" ? "/doctor" : "/patient"} className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-vital-pulse to-vital-info flex items-center justify-center">
                <Activity className="h-5 w-5 text-midnight" />
              </div>
              {/* Pulse effect */}
              <div className="absolute inset-0 rounded-xl bg-vital-pulse/30 animate-ping" />
            </div>
            <div>
              <span className="text-xl font-bold text-clinical-white">
                Medi<span className="text-gradient">Sync</span>
              </span>
              <span className="block text-xs text-clinical-muted -mt-0.5">
                {userRole === "doctor" ? "Clinical Portal" : "Patient Companion"}
              </span>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center gap-1">
            {links.map((link) => {
              const isActive = pathname === link.href;
              const Icon = link.icon;

              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={cn(
                    "relative px-4 py-2 rounded-lg flex items-center gap-2 transition-all duration-200",
                    isActive
                      ? "text-vital-pulse"
                      : "text-clinical-muted hover:text-clinical-light hover:bg-midnight-50"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="text-sm font-medium">{link.label}</span>
                  {isActive && (
                    <motion.div
                      layoutId="nav-indicator"
                      className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-vital-pulse"
                    />
                  )}
                </Link>
              );
            })}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Search */}
            <button className="p-2 rounded-lg text-clinical-muted hover:text-clinical-light hover:bg-midnight-50 transition-colors">
              <Search className="h-5 w-5" />
            </button>

            {/* Notifications */}
            <button className="relative p-2 rounded-lg text-clinical-muted hover:text-clinical-light hover:bg-midnight-50 transition-colors">
              <Bell className="h-5 w-5" />
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-vital-critical" />
            </button>

            {/* User menu */}
            <div className="flex items-center gap-3 pl-3 border-l border-clinical-border/30">
              <div className="text-right hidden sm:block">
                <p className="text-sm font-medium text-clinical-light">
                  {userName}
                </p>
                <p className="text-xs text-clinical-muted capitalize">
                  {userRole}
                </p>
              </div>
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-vital-pulse/20 to-vital-info/20 flex items-center justify-center border border-vital-pulse/30">
                <User className="h-4 w-4 text-vital-pulse" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile navigation */}
      <div className="md:hidden border-t border-clinical-border/30">
        <div className="flex items-center justify-around py-2">
          {links.slice(0, 4).map((link) => {
            const isActive = pathname === link.href;
            const Icon = link.icon;

            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "flex flex-col items-center gap-1 py-1 px-3 rounded-lg transition-colors",
                  isActive
                    ? "text-vital-pulse"
                    : "text-clinical-muted"
                )}
              >
                <Icon className="h-5 w-5" />
                <span className="text-xs">{link.label}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
