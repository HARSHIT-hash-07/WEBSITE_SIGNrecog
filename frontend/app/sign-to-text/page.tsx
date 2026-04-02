"use client";

import { Button } from "@/components/ui/Button";
import { Camera, CameraOff, Video } from "lucide-react";
import { useState } from "react";
import { motion } from "framer-motion";
import FloatingLines from "@/components/ui/FloatingLines";

export default function SignToTextPage() {
  const [cameraActive, setCameraActive] = useState(false);

  const waveColors = ["#3730a3", "#581c87", "#831843"];
  const bgColor = "#000000";

  return (
    <div className="relative min-h-screen">
      {/* Background */}
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
        <h1 className="text-4xl md:text-5xl font-bold mb-8 text-center flex flex-col items-center gap-4">
          <div>
            <span className="gradient-text font-cursive">Sign</span> to Text
            Translation
          </div>
          <span className="text-xs font-mono px-3 py-1 bg-indigo-500/20 border border-indigo-500/30 rounded-full text-indigo-300 animate-pulse uppercase tracking-widest">
            Coming Soon
          </span>
        </h1>

        <div className="max-w-4xl mx-auto relative group">
          {/* Coming Soon Overlay */}
          <div className="absolute inset-0 z-30 flex items-center justify-center pointer-events-none">
            <div className="bg-black/60 backdrop-blur-md border border-white/10 px-8 py-4 rounded-2xl shadow-2xl transform rotate-[-5deg] group-hover:rotate-0 transition-transform duration-500">
              <span className="text-3xl font-black italic tracking-tighter gradient-text uppercase">
                Under Development
              </span>
            </div>
          </div>
          <div className="card min-h-125 flex flex-col items-center justify-center relative overflow-hidden border-indigo-500/30 shadow-2xl shadow-indigo-500/10">
            {/* Scanner UI Elements */}
            <div className="absolute inset-0 pointer-events-none">
              {/* Corner Brackets */}
              <div className="absolute top-4 left-4 w-16 h-16 border-t-2 border-l-2 border-indigo-500 rounded-tl-lg" />
              <div className="absolute top-4 right-4 w-16 h-16 border-t-2 border-r-2 border-indigo-500 rounded-tr-lg" />
              <div className="absolute bottom-4 left-4 w-16 h-16 border-b-2 border-l-2 border-indigo-500 rounded-bl-lg" />
              <div className="absolute bottom-4 right-4 w-16 h-16 border-b-2 border-r-2 border-indigo-500 rounded-br-lg" />

              {/* Grid Overlay */}
              <div className="absolute inset-0 bg-[linear-gradient(rgba(99,102,241,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(99,102,241,0.05)_1px,transparent_1px)] bg-size-[40px_40px]" />

              {/* Scanning Line Animation */}
              {cameraActive && (
                <motion.div
                  className="absolute left-0 right-0 h-0.5 bg-indigo-400 shadow-[0_0_20px_rgba(129,140,248,0.5)] z-20"
                  animate={{ top: ["10%", "90%"], opacity: [0, 1, 0] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                />
              )}
            </div>

            {/* Background Ambient Effect */}
            <div className="absolute inset-0 bg-linear-to-b from-indigo-500/5 to-transparent pointer-events-none" />

            {cameraActive ? (
              <div className="w-full h-full absolute inset-0 bg-black/50 backdrop-blur-sm flex flex-col items-center justify-center text-slate-400 z-10">
                <div className="animate-pulse flex flex-col items-center gap-4">
                  <Video className="w-16 h-16 opacity-50 text-indigo-500" />
                  <p className="font-mono text-indigo-400">
                    CONNECTING TO VIDEO FEED...
                  </p>
                  <p className="text-xs max-w-xs text-center font-mono opacity-70">
                    Initializing sign language detection model
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-6 z-10">
                <div className="w-32 h-32 rounded-full bg-linear-to-br from-indigo-500/20 to-violet-500/20 flex items-center justify-center backdrop-blur-sm border border-indigo-500/30">
                  <Camera className="w-16 h-16 text-indigo-400" />
                </div>
                <div className="text-center space-y-2">
                  <h2 className="text-xl font-semibold">
                    Camera Access Required
                  </h2>
                  <p className="text-muted dark:text-zinc-400 max-w-md">
                    Enable your camera to start translating sign language
                    gestures into text
                  </p>
                </div>
              </div>
            )}

            {/* Camera Controls */}
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20">
              <Button
                onClick={() => setCameraActive(!cameraActive)}
                className={
                  cameraActive
                    ? "bg-red-500 hover:bg-red-600 text-white"
                    : "btn-gradient"
                }
              >
                {cameraActive ? (
                  <>
                    <CameraOff className="w-5 h-5 mr-2" />
                    Stop Camera
                  </>
                ) : (
                  <>
                    <Camera className="w-5 h-5 mr-2" />
                    Start Camera
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Translation Output */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mt-8 card backdrop-blur-sm bg-card-bg/80"
          >
            <h3 className="text-lg font-semibold mb-4 gradient-text">
              Translation Output
            </h3>
            <div className="min-h-25 p-4 rounded-lg bg-input/50 border border-border font-mono text-sm">
              {cameraActive ? (
                <p className="text-muted dark:text-zinc-400 italic">
                  Waiting for sign language input...
                </p>
              ) : (
                <p className="text-muted dark:text-zinc-400 italic">
                  Start the camera to begin translation
                </p>
              )}
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
