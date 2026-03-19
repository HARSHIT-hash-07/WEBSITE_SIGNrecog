"use client";

import { useEffect, useRef } from "react";

interface GradualBlurProps {
  target?: "parent" | "body";
  position?: "top" | "bottom";
  height?: string;
  strength?: number;
  divCount?: number;
  curve?: "linear" | "bezier";
  exponential?: boolean;
  opacity?: number;
}

export default function GradualBlur({
  target = "parent",
  position = "bottom",
  height = "7rem",
  strength = 1.5,
  divCount = 5,
  curve = "bezier",
  exponential = false,
  opacity = 0.8,
}: GradualBlurProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const layers = Array.from(container.children) as HTMLDivElement[];

    layers.forEach((layer, index) => {
      const progress = (index + 1) / divCount;
      let blurAmount: number;

      if (exponential) {
        blurAmount = Math.pow(progress, 2) * strength * 10;
      } else if (curve === "bezier") {
        // Cubic bezier approximation
        const t = progress;
        blurAmount =
          (3 * Math.pow(1 - t, 2) * t * 0.5 +
            3 * (1 - t) * Math.pow(t, 2) * 0.75 +
            Math.pow(t, 3)) *
          strength *
          10;
      } else {
        blurAmount = progress * strength * 10;
      }

      layer.style.backdropFilter = `blur(${blurAmount}px)`;
      (layer.style as any).WebkitBackdropFilter = `blur(${blurAmount}px)`;
    });
  }, [strength, divCount, curve, exponential]);

  const positionStyles = position === "bottom" ? { bottom: 0 } : { top: 0 };

  return (
    <div
      ref={containerRef}
      className="pointer-events-none absolute left-0 right-0 z-10"
      style={{
        ...positionStyles,
        height,
        opacity,
      }}
    >
      {Array.from({ length: divCount }).map((_, i) => (
        <div
          key={i}
          className="absolute inset-0 bg-background/20"
          style={{
            height: `${100 / divCount}%`,
            top: position === "bottom" ? `${(i * 100) / divCount}%` : undefined,
            bottom: position === "top" ? `${(i * 100) / divCount}%` : undefined,
          }}
        />
      ))}
    </div>
  );
}
