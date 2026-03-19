"use client";

import type { Metadata } from "next";
import { TextToSignClient } from "@/components/features/TextToSignClient";
import FloatingLines from "@/components/ui/FloatingLines";

export default function TextToSignPage() {
  const waveColors = ["#3730a3", "#581c87", "#831843"];
  const bgColor = "#000000";

  return (
    <div className="relative min-h-screen">
      {/* Background */}
      <div className="fixed inset-0 z-0">
        <FloatingLines
          linesGradient={waveColors}
          backgroundColor={bgColor}
          brightnessMultiplier={1.0}
          enabledWaves={["top", "middle", "bottom"]}
          lineCount={5}
          lineDistance={5}
          bendRadius={3}
          bendStrength={-0.8}
          interactive={true}
          parallax={false}
          mixBlendMode="normal"
        />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        <h1 className="text-4xl md:text-5xl font-bold mb-8 text-center">
          Text to <span className="gradient-text font-cursive">Sign</span>{" "}
          Translation
        </h1>
        <TextToSignClient />
      </div>
    </div>
  );
}
