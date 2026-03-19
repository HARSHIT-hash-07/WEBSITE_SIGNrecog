"use client";

import { Button } from "@/components/ui/Button";
import { ArrowRight, MessageSquareText, Brain, Video } from "lucide-react";
import Link from "next/link";
import { ScrollReveal } from "@/components/ui/ScrollReveal";
import FloatingLines from "@/components/ui/FloatingLines";
import GradualBlur from "@/components/ui/GradualBlur";
import TiltedCard from "@/components/ui/TiltedCard";
import { motion } from "motion/react";
import PillButton from "@/components/ui/PillButton";

export default function Home() {
  // Dark mode only - wave colors
  const waveColors = ["#3730a3", "#581c87", "#831843"];
  const bgColor = "#000000";

  return (
    <div className="min-h-screen bg-background relative">
      {/* Floating Lines Background - Dark Mode Only */}
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

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-20 pb-32 z-10">
        <div className="container relative mx-auto px-4">
          <motion.div
            initial="hidden"
            animate="visible"
            variants={{
              hidden: { opacity: 0 },
              visible: {
                opacity: 1,
                transition: {
                  staggerChildren: 0.2,
                  delayChildren: 0.1,
                },
              },
            }}
            className="flex flex-col items-center text-center max-w-5xl mx-auto"
          >
            {/* USP Badge - Moved to Top */}
            <motion.div
              variants={{
                hidden: { opacity: 0, y: -20 },
                visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
              }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/50 border border-zinc-800/50 backdrop-blur-md mb-8"
            >
              <span className="text-xs md:text-sm text-zinc-400 font-medium">
                Built for
              </span>
              <span className="text-xs md:text-sm font-semibold gradient-text">
                Accessibility & Inclusion
              </span>
            </motion.div>

            {/* Main Heading */}
            <motion.div
              variants={{
                hidden: { opacity: 0, y: 20, scale: 0.95 },
                visible: {
                  opacity: 1,
                  y: 0,
                  scale: 1,
                  transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] },
                },
              }}
            >
              <h1 className="text-6xl md:text-8xl font-medium tracking-tighter mb-8 leading-[1.05] text-white">
                Bridge Communication
                <br />
                with{" "}
                <span className="gradient-text font-cursive font-normal">
                  Sign Language
                </span>
              </h1>
            </motion.div>

            {/* Subheading */}
            <motion.div
              variants={{
                hidden: { opacity: 0, y: 20 },
                visible: {
                  opacity: 1,
                  y: 0,
                  transition: { duration: 0.8, ease: "easeOut" },
                },
              }}
            >
              <p className="text-xl md:text-2xl text-zinc-300 mb-10 max-w-3xl mx-auto leading-relaxed font-normal">
                Automate sign language translation, unlock real-time
                accessibility, and scale inclusive communication.
              </p>
            </motion.div>

            {/* CTA Buttons */}
            <motion.div
              variants={{
                hidden: { opacity: 0, y: 20 },
                visible: {
                  opacity: 1,
                  y: 0,
                  transition: { duration: 0.8, ease: "easeOut" },
                },
              }}
              className="flex flex-col sm:flex-row items-center justify-center gap-6"
            >
              <div className="scale-105">
                <PillButton
                  href="/text-to-sign"
                  label="Get Started"
                  className="font-medium shadow-2xl text-lg px-8 py-4"
                  baseColor="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)"
                  hoverColor="#ffffff"
                  textColor="#ffffff"
                  hoverTextColor="#000000"
                />
              </div>

              <Link href="/sign-to-text">
                <button className="px-8 py-4 text-lg font-medium text-zinc-400 hover:text-white transition-colors border border-zinc-800 hover:border-zinc-600 rounded-full">
                  Try Beta
                </button>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features Section with Slide Up Animation */}
      <section className="py-24 relative z-10">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {/* Feature 1 */}
            <ScrollReveal delay={0.1} direction="up">
              <TiltedCard
                containerHeight="300px"
                containerWidth="100%"
                imageHeight="300px"
                imageWidth="100%"
                rotateAmplitude={12}
                scaleOnHover={1.05}
                showMobileWarning={false}
                showTooltip={false}
                displayOverlayContent={false}
              >
                <>
                  <div className="w-16 h-16 mx-auto mb-6 flex items-center justify-center transition-transform">
                    <MessageSquareText className="w-10 h-10 text-indigo-500" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-center">
                    Text to Sign
                  </h3>
                  <p className="text-muted dark:text-zinc-400 leading-relaxed text-center">
                    Convert written text into accurate sign language animations
                    instantly
                  </p>
                </>
              </TiltedCard>
            </ScrollReveal>

            {/* Feature 2 */}
            <ScrollReveal delay={0.2} direction="up">
              <TiltedCard
                containerHeight="300px"
                containerWidth="100%"
                imageHeight="300px"
                imageWidth="100%"
                rotateAmplitude={12}
                scaleOnHover={1.05}
                showMobileWarning={false}
                showTooltip={false}
                displayOverlayContent={false}
              >
                <>
                  <div className="w-16 h-16 mx-auto mb-6 flex items-center justify-center transition-transform">
                    <Brain className="w-10 h-10 text-violet-500" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-center">
                    AI-Powered
                  </h3>
                  <p className="text-muted dark:text-zinc-400 leading-relaxed text-center">
                    Unlock data-driven translation with cutting-edge AI models
                  </p>
                </>
              </TiltedCard>
            </ScrollReveal>

            {/* Feature 3 */}
            <ScrollReveal delay={0.3} direction="up">
              <TiltedCard
                containerHeight="300px"
                containerWidth="100%"
                imageHeight="300px"
                imageWidth="100%"
                rotateAmplitude={12}
                scaleOnHover={1.05}
                showMobileWarning={false}
                showTooltip={false}
                displayOverlayContent={false}
              >
                <>
                  <div className="w-16 h-16 mx-auto mb-6 flex items-center justify-center transition-transform">
                    <Video className="w-10 h-10 text-pink-500" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-center">
                    Sign to Text
                  </h3>
                  <p className="text-muted dark:text-zinc-400 leading-relaxed text-center">
                    Translate sign language gestures into text using your camera
                  </p>
                </>
              </TiltedCard>
            </ScrollReveal>
          </div>
        </div>
      </section>

      {/* Mission Section with Gradual Blur */}
      <section className="relative py-24 bg-background z-10">
        <div className="container mx-auto px-4">
          <ScrollReveal direction="up">
            <div className="max-w-4xl mx-auto text-center pb-24">
              <p className="text-2xl font-medium tracking-tighter text-indigo-400 uppercase mb-6">
                Mission
              </p>
              <h2 className="text-4xl md:text-5xl font-light mb-6 leading-tight text-white">
                The New Era of{" "}
                <span className="gradient-text font-cursive font-normal">
                  AI-Powered Accessibility
                </span>
              </h2>
              <p className="text-xl text-zinc-300 font-normal leading-relaxed mb-8 max-w-3xl mx-auto">
                Our platform puts AI at the center of communication — helping
                bridge the gap between text and sign language, generate instant
                translations, and make accessibility seamless.
              </p>
              <p className="text-xl text-zinc-300 font-normal leading-relaxed mb-16 max-w-3xl mx-auto">
                With real-time translation and easy integration, you can build
                more inclusive experiences without limits.
              </p>

              {/* Stats Grid with Custom Styling */}
              <div className="grid md:grid-cols-3 gap-8">
                <ScrollReveal delay={0.1} direction="up">
                  <TiltedCard
                    containerHeight="180px"
                    containerWidth="100%"
                    imageHeight="100%"
                    imageWidth="100%"
                    rotateAmplitude={10}
                    scaleOnHover={1.05}
                    showMobileWarning={false}
                    showTooltip={false}
                    displayOverlayContent={false}
                  >
                    <>
                      <p className="text-sm font-medium text-muted mb-2">
                        Launch Date
                      </p>
                      <p className="text-3xl font-bold gradient-text">
                        Coming Soon
                      </p>
                    </>
                  </TiltedCard>
                </ScrollReveal>

                <ScrollReveal delay={0.2} direction="up">
                  <TiltedCard
                    containerHeight="180px"
                    containerWidth="100%"
                    imageHeight="100%"
                    imageWidth="100%"
                    rotateAmplitude={10}
                    scaleOnHover={1.05}
                    showMobileWarning={false}
                    showTooltip={false}
                    displayOverlayContent={false}
                  >
                    <>
                      <p className="text-sm font-medium text-muted mb-2">
                        Key Benefit
                      </p>
                      <p className="text-3xl font-bold gradient-text leading-tight">
                        Real-time Translation
                      </p>
                    </>
                  </TiltedCard>
                </ScrollReveal>

                <ScrollReveal delay={0.3} direction="up">
                  <TiltedCard
                    containerHeight="180px"
                    containerWidth="100%"
                    imageHeight="100%"
                    imageWidth="100%"
                    rotateAmplitude={10}
                    scaleOnHover={1.05}
                    showMobileWarning={false}
                    showTooltip={false}
                    displayOverlayContent={false}
                  >
                    <>
                      <p className="text-sm font-medium text-muted mb-2">
                        Built For
                      </p>
                      <p className="text-3xl font-bold gradient-text">
                        Everyone
                      </p>
                    </>
                  </TiltedCard>
                </ScrollReveal>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </section>
    </div>
  );
}
