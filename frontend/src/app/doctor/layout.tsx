"use client";

import { Navigation, Sidebar } from "@/components/shared";

export default function DoctorLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-midnight">
      <Navigation userRole="doctor" userName="Dr. Strange" />
      <Sidebar />
      <main className="pt-16 pl-64 min-h-screen transition-all duration-300">
        <div className="p-6 lg:p-8">{children}</div>
      </main>
    </div>
  );
}
