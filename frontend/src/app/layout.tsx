import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MediSync | Clinical Decision Support",
  description: "AI-powered clinical decision support system for healthcare professionals",
  keywords: ["healthcare", "AI", "clinical decision support", "medical", "diagnosis"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-midnight antialiased">
        {children}
      </body>
    </html>
  );
}
