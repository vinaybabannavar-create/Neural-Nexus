import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Neural Nexus | Self-Reflective C-RAG",
  description: "Advanced Self-Reflective Corrective RAG with Next.js, LiveKit Voice, and Confidence Escalation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased selection:bg-accent/20 selection:text-accent">
        {children}
      </body>
    </html>
  );
}
