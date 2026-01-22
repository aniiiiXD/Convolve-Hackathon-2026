import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Core palette - Medical Modernist
        midnight: {
          DEFAULT: "#0a0e14",
          50: "#1a2130",
          100: "#151b26",
          200: "#10151d",
          300: "#0a0e14",
        },
        slate: {
          850: "#1e2736",
          950: "#0f1419",
        },
        // Clinical accent colors
        vital: {
          pulse: "#00d4aa",      // Healthy vitals - teal green
          warning: "#ffb347",    // Caution - warm amber
          critical: "#ff6b6b",   // Alert - coral red
          info: "#4ecdc4",       // Information - cyan
          calm: "#95e1d3",       // Stable - mint
        },
        // Medical imaging inspired
        scan: {
          ultrasound: "#1e90ff",
          xray: "#00ced1",
          thermal: "#ff7f50",
          mri: "#9370db",
        },
        // Neutral clinical
        clinical: {
          white: "#f8fafc",
          light: "#e2e8f0",
          muted: "#94a3b8",
          border: "#334155",
        },
      },
      fontFamily: {
        display: ["var(--font-instrument)", "system-ui", "sans-serif"],
        body: ["var(--font-satoshi)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "monospace"],
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "grid-pattern": `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%231e2736' fill-opacity='0.4'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        "ecg-line": `url("data:image/svg+xml,%3Csvg width='200' height='40' viewBox='0 0 200 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 20 L40 20 L45 20 L50 10 L55 30 L60 5 L65 35 L70 20 L75 20 L200 20' stroke='%2300d4aa' stroke-width='1' fill='none' opacity='0.15'/%3E%3C/svg%3E")`,
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "pulse-vital": "pulseVital 1.5s ease-in-out infinite",
        "slide-up": "slideUp 0.5s ease-out",
        "slide-down": "slideDown 0.3s ease-out",
        "fade-in": "fadeIn 0.4s ease-out",
        "scan-line": "scanLine 3s linear infinite",
        "glow": "glow 2s ease-in-out infinite alternate",
        "ecg": "ecg 2s linear infinite",
      },
      keyframes: {
        pulseVital: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.7", transform: "scale(1.05)" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideDown: {
          "0%": { opacity: "0", transform: "translateY(-10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        scanLine: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        glow: {
          "0%": { boxShadow: "0 0 5px currentColor, 0 0 10px currentColor" },
          "100%": { boxShadow: "0 0 10px currentColor, 0 0 20px currentColor, 0 0 30px currentColor" },
        },
        ecg: {
          "0%": { backgroundPosition: "200px 0" },
          "100%": { backgroundPosition: "0 0" },
        },
      },
      boxShadow: {
        "glow-teal": "0 0 20px rgba(0, 212, 170, 0.3)",
        "glow-amber": "0 0 20px rgba(255, 179, 71, 0.3)",
        "glow-coral": "0 0 20px rgba(255, 107, 107, 0.3)",
        "inner-glow": "inset 0 1px 0 0 rgba(255, 255, 255, 0.05)",
      },
    },
  },
  plugins: [],
};
export default config;
