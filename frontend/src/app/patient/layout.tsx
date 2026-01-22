"use client";

import { Navigation } from "@/components/shared";

export default function PatientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-midnight">
      <Navigation userRole="patient" userName="John Smith" />
      <main className="pt-24 pb-8 px-4 sm:px-6 lg:px-8 max-w-5xl mx-auto">
        {children}
      </main>
    </div>
  );
}
