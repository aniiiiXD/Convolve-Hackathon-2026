# MediSync Frontend

**Next.js 15 + Tailwind CSS v4 + Medical Modernist Design**

A production-grade clinical decision support interface for the MediSync healthcare AI system.

---

## Quick Start

```bash
# Install dependencies
npm install

# Development
npm run dev

# Production build
npm run build && npm run start
```

Open http://localhost:3000

---

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with animated hero, feature showcase |
| `/doctor` | Doctor dashboard - patients, alerts, insights |
| `/doctor/diagnosis` | Differential diagnosis tool |
| `/patient` | Patient portal - health score, vitals, history |

---

## Design System

### Medical Modernist Theme

Dark theme optimized for clinical environments with ECG-inspired animations.

```css
/* Core Colors */
--color-midnight: #0a0e14;        /* Background */
--color-vital-pulse: #00d4aa;     /* Primary accent (teal) */
--color-vital-warning: #ffb347;   /* Warning (amber) */
--color-vital-critical: #ff6b6b;  /* Critical (coral) */
--color-vital-info: #4ecdc4;      /* Info (cyan) */

/* Medical Imaging Colors */
--color-scan-ultrasound: #1e90ff; /* Blue */
--color-scan-xray: #00ced1;       /* Cyan */
--color-scan-thermal: #ff7f50;    /* Orange */
--color-scan-mri: #9370db;        /* Purple */
```

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `Button` | `components/ui/Button.tsx` | Primary, secondary, ghost variants |
| `Card` | `components/ui/Card.tsx` | Glass-morphism card with hover effects |
| `Badge` | `components/ui/Badge.tsx` | Severity badges (critical, warning, stable) |
| `Input` | `components/ui/Input.tsx` | Clinical input fields |
| `ConfidenceMeter` | `components/ui/ConfidenceMeter.tsx` | Diagnosis confidence visualization |
| `VitalSign` | `components/ui/VitalSign.tsx` | Vital sign display with trends |

### Dashboard Components

| Component | Purpose |
|-----------|---------|
| `AlertPanel` | Critical alerts with severity indicators |
| `DiagnosisPanel` | Differential diagnosis with confidence bars |
| `InsightsPanel` | AI-generated clinical insights |
| `EvidenceGraphPanel` | Reasoning chain visualization |
| `PatientList` | Patient roster with status badges |

---

## Architecture

```
frontend/
├── src/
│   ├── app/                     # Next.js App Router
│   │   ├── page.tsx             # Landing page
│   │   ├── layout.tsx           # Root layout
│   │   ├── globals.css          # Tailwind + custom CSS
│   │   ├── doctor/
│   │   │   ├── page.tsx         # Doctor dashboard
│   │   │   └── diagnosis/
│   │   │       └── page.tsx     # Diagnosis tool
│   │   └── patient/
│   │       └── page.tsx         # Patient portal
│   │
│   ├── components/
│   │   ├── ui/                  # Base UI components
│   │   ├── dashboard/           # Dashboard-specific components
│   │   └── shared/              # Navigation, sidebar
│   │
│   ├── lib/
│   │   ├── api.ts               # API client
│   │   └── utils.ts             # Utility functions
│   │
│   ├── hooks/                   # React hooks
│   │   ├── useAlerts.ts
│   │   ├── useDiagnosis.ts
│   │   └── useInsights.ts
│   │
│   └── types/                   # TypeScript definitions
│       └── index.ts
│
├── tailwind.config.ts           # Tailwind configuration
├── next.config.ts               # Next.js configuration
└── package.json
```

---

## CSS Classes

### Cards

```html
<!-- Glass card with blur effect -->
<div class="glass-card">Content</div>

<!-- Glass card with hover glow -->
<div class="glass-card-hover">Content</div>

<!-- Data panel with top gradient line -->
<div class="data-panel">Content</div>
```

### Badges

```html
<span class="badge-critical">CRITICAL</span>
<span class="badge-warning">WARNING</span>
<span class="badge-stable">STABLE</span>
<span class="badge-info">INFO</span>
```

### Buttons

```html
<button class="btn-primary">Primary Action</button>
<button class="btn-secondary">Secondary</button>
<button class="btn-ghost">Ghost</button>
```

### Alerts

```html
<div class="alert-card alert-critical">Critical alert</div>
<div class="alert-card alert-warning">Warning alert</div>
<div class="alert-card alert-info">Info alert</div>
```

### Effects

```html
<!-- ECG animation background -->
<div class="ecg-bg">...</div>

<!-- Grid overlay -->
<div class="grid-overlay">...</div>

<!-- Gradient text -->
<span class="text-gradient">Gradient</span>
```

---

## API Integration

The frontend uses mock data by default. To connect to the backend:

```typescript
// src/lib/api.ts
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const api = {
  search: (query: string) => fetch(`${API_BASE}/search`, {...}),
  diagnose: (symptoms: string) => fetch(`${API_BASE}/diagnose`, {...}),
  alerts: (patientId: string) => fetch(`${API_BASE}/alerts/${patientId}`, {...}),
};
```

---

## Development

### Type Checking

```bash
npm run lint
npx tsc --noEmit
```

### Build

```bash
npm run build
```

### Clean Install

```bash
rm -rf node_modules .next
npm install
npm run build
```

---

## Screenshots

### Doctor Dashboard
- Patient list with status badges
- Real-time alerts panel
- AI insights carousel
- Quick actions

### Diagnosis Tool
- Symptom input with suggestions
- Differential diagnosis cards
- Confidence meters
- Evidence graph visualization

### Patient Portal
- Health score gauge
- Vital signs with trends
- Medication list
- Appointment reminders

---

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 15 | React framework |
| React | 19 | UI library |
| Tailwind CSS | 4 | Styling |
| TypeScript | 5 | Type safety |
| Motion | 11 | Animations |
