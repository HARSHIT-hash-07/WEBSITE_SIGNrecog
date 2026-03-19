"use client";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";

export function BackgroundBeams() {
  const [beams, setBeams] = useState<
    Array<{
      initialX: number;
      initialY: number;
      initialScale: number;
      animateX: number;
      animateY: number;
      duration: number;
      width: string;
      height: string;
    }>
  >([]);

  useEffect(() => {
    const newBeams = Array.from({ length: 6 }).map(() => ({
      initialX: Math.random() * window.innerWidth,
      initialY: Math.random() * window.innerHeight,
      initialScale: Math.random() * 0.5 + 0.5,
      animateX: Math.random() * window.innerWidth,
      animateY: Math.random() * window.innerHeight,
      duration: Math.random() * 20 + 20,
      width: Math.random() * 400 + 100 + "px",
      height: Math.random() * 400 + 100 + "px",
    }));
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setBeams(newBeams);
  }, []);

  if (beams.length === 0) return null;

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none -z-10">
      <div className="absolute inset-0 bg-linear-to-b from-transparent to-slate-50/50 dark:to-slate-900/50" />
      {beams.map((beam, i) => (
        <motion.div
          key={i}
          className="absolute bg-primary/10 dark:bg-primary/5 blur-3xl rounded-full"
          initial={{
            x: beam.initialX,
            y: beam.initialY,
            scale: beam.initialScale,
          }}
          animate={{
            x: beam.animateX,
            y: beam.animateY,
            rotate: 360,
          }}
          transition={{
            duration: beam.duration,
            repeat: Infinity,
            repeatType: "mirror",
            ease: "easeInOut",
          }}
          style={{
            width: beam.width,
            height: beam.height,
          }}
        />
      ))}
    </div>
  );
}
