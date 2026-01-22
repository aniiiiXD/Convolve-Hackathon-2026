"use client";

import { cn } from "@/lib/utils";
import { motion, HTMLMotionProps } from "motion/react";
import { forwardRef, ReactNode } from "react";

interface CardProps extends Omit<HTMLMotionProps<"div">, "ref"> {
  variant?: "default" | "hover" | "alert" | "metric";
  glow?: "teal" | "amber" | "coral" | "none";
  children: ReactNode;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = "default", glow = "none", children, ...props }, ref) => {
    const variants = {
      default: "glass-card",
      hover: "glass-card-hover",
      alert: "alert-card",
      metric: "metric-card",
    };

    const glowStyles = {
      teal: "shadow-glow-teal border-vital-pulse/30",
      amber: "shadow-glow-amber border-vital-warning/30",
      coral: "shadow-glow-coral border-vital-critical/30",
      none: "",
    };

    return (
      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className={cn(variants[variant], glowStyles[glow], className)}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);

Card.displayName = "Card";

interface CardHeaderProps {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  className?: string;
}

export function CardHeader({ title, subtitle, action, className }: CardHeaderProps) {
  return (
    <div className={cn("flex items-start justify-between mb-4", className)}>
      <div>
        <h3 className="text-lg font-semibold text-clinical-white">{title}</h3>
        {subtitle && (
          <p className="text-sm text-clinical-muted mt-0.5">{subtitle}</p>
        )}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}

interface CardContentProps {
  children: ReactNode;
  className?: string;
}

export function CardContent({ children, className }: CardContentProps) {
  return <div className={cn("", className)}>{children}</div>;
}

export default Card;
