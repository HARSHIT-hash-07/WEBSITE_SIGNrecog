"use client";

import React, { useRef, useEffect } from "react";
import Link from "next/link";
import { gsap } from "gsap";
import "./PillNav.css"; // Use shared styles

interface PillButtonProps {
  href?: string;
  label: string;
  onClick?: () => void;
  className?: string;
  baseColor?: string;
  hoverColor?: string;
  textColor?: string;
  hoverTextColor?: string;
}

export default function PillButton({
  href,
  label,
  onClick,
  className = "",
  baseColor = "#000000",
  hoverColor = "#ffffff",
  textColor = "#ffffff",
  hoverTextColor = "#000000",
}: PillButtonProps) {
  const circleRef = useRef<HTMLSpanElement>(null);
  const labelRef = useRef<HTMLSpanElement>(null);
  const hoverLabelRef = useRef<HTMLSpanElement>(null);
  const tlRef = useRef<gsap.core.Timeline | null>(null);
  const activeTweenRef = useRef<gsap.core.Tween | null>(null);

  useEffect(() => {
    // ... (same animation effect logic) ...
    const circle = circleRef.current;
    const labelEl = labelRef.current;
    const hoverLabelEl = hoverLabelRef.current;

    if (!circle || !labelEl || !hoverLabelEl) return;

    const setupLayout = () => {
      if (!circle.parentElement) return;
      const rect = circle.parentElement.getBoundingClientRect();
      const { width: w, height: h } = rect;

      // Safety check for zero dimensions (e.g. if hidden)
      if (w === 0 || h === 0) return;

      const R = ((w * w) / 4 + h * h) / (2 * h);
      const D = Math.ceil(2 * R) + 2;

      const delta =
        Math.ceil(R - Math.sqrt(Math.max(0, R * R - (w * w) / 4))) + 2;

      circle.style.width = `${D}px`;
      circle.style.height = `${D}px`;
      circle.style.bottom = `-${delta}px`;

      const originY = D - delta;
      gsap.set(circle, {
        xPercent: -50,
        scale: 0,
        backgroundColor: hoverColor,
        transformOrigin: `50% ${originY}px`,
      });

      gsap.set(labelEl, { y: 0, color: textColor });
      gsap.set(hoverLabelEl, {
        y: h + 10,
        opacity: 0,
        color: hoverTextColor,
      });

      tlRef.current?.kill();
      const tl = gsap.timeline({ paused: true });

      tl.to(circle, { scale: 1.2, duration: 0.5, ease: "power3.easeOut" }, 0);
      tl.to(
        labelEl,
        { y: -(h + 10), duration: 0.5, ease: "power3.easeOut" },
        0,
      );
      tl.set(hoverLabelEl, { y: h + 20, opacity: 0 }, 0);
      tl.to(
        hoverLabelEl,
        { y: 0, opacity: 1, duration: 0.5, ease: "power3.easeOut" },
        0,
      );

      tlRef.current = tl;
    };

    setupLayout();

    // Add resize listener just in case
    window.addEventListener("resize", setupLayout);
    return () => window.removeEventListener("resize", setupLayout);
  }, [hoverColor, textColor, hoverTextColor]);

  const handleMouseEnter = () => {
    const tl = tlRef.current;
    if (!tl) return;
    activeTweenRef.current?.kill();
    activeTweenRef.current = tl.tweenTo(tl.duration(), {
      duration: 0.4,
      ease: "power3.easeOut",
      overwrite: "auto",
    });
  };

  const handleMouseLeave = () => {
    const tl = tlRef.current;
    if (!tl) return;
    activeTweenRef.current?.kill();
    activeTweenRef.current = tl.tweenTo(0, {
      duration: 0.3,
      ease: "power3.easeOut",
      overwrite: "auto",
    });
  };

  // Determine element type
  const Component = href ? Link : "button";

  // Safe props extraction
  const linkProps = href ? { href } : {};
  const buttonProps = !href ? { type: "button" as const, onClick } : {};

  return (
    // @ts-expect-error - Dynamic component props are tricky with TS
    <Component
      {...linkProps}
      {...buttonProps}
      className={`pill-button relative inline-flex items-center justify-center overflow-hidden rounded-full ${className}`}
      style={
        {
          background: baseColor,
          color: textColor,
          padding: "12px 32px",
          minWidth: "140px",
          border: "1px solid rgba(255,255,255,0.1)",
        } as React.CSSProperties
      }
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <span
        className="hover-circle absolute left-1/2 bottom-0 pointer-events-none z-0"
        ref={circleRef}
      />

      <span className="relative z-10 flex flex-col items-center justify-center h-[1.2em] overflow-visible">
        <span
          className="pill-button-text block whitespace-nowrap leading-none"
          ref={labelRef}
        >
          {label}
        </span>
        <span
          className="pill-button-text-hover absolute left-0 block whitespace-nowrap leading-none w-full text-center"
          ref={hoverLabelRef}
        >
          {label}
        </span>
      </span>
    </Component>
  );
}
