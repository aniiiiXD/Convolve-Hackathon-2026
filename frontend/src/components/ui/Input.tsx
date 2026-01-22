"use client";

import { cn } from "@/lib/utils";
import { forwardRef, InputHTMLAttributes } from "react";
import { Search, X } from "lucide-react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  icon?: React.ReactNode;
  onClear?: () => void;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, icon, onClear, type = "text", ...props }, ref) => {
    return (
      <div className="w-full">
        {label && (
          <label className="block text-sm font-medium text-clinical-light mb-1.5">
            {label}
          </label>
        )}
        <div className="relative">
          {icon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-clinical-muted">
              {icon}
            </div>
          )}
          <input
            ref={ref}
            type={type}
            className={cn(
              "input-clinical",
              icon && "pl-10",
              onClear && props.value && "pr-10",
              error && "border-vital-critical/50 focus:border-vital-critical/50 focus:ring-vital-critical/20",
              className
            )}
            {...props}
          />
          {onClear && props.value && (
            <button
              type="button"
              onClick={onClear}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-clinical-muted hover:text-clinical-light transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        {error && (
          <p className="mt-1.5 text-sm text-vital-critical">{error}</p>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";

export default Input;

// Specialized Search Input
interface SearchInputProps extends Omit<InputProps, "icon" | "type"> {
  onSearch?: (value: string) => void;
}

export function SearchInput({ onSearch, ...props }: SearchInputProps) {
  return (
    <Input
      type="search"
      icon={<Search className="h-4 w-4" />}
      placeholder="Search patients, conditions, records..."
      {...props}
    />
  );
}
