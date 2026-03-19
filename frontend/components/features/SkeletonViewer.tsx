"use client";

import React, { useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import * as THREE from "three";

interface SkeletonViewerProps {
  skeletons: number[][][]; // [frame][joint][x,y,z]
  isPlaying: boolean;
}

function Skeleton({ frameData }: { frameData: number[][] }) {
  // Simple rendering of joints as spheres and bones as lines
  // Assuming 21 joints standard hand model (or body)
  // For now just rendering points

  if (!frameData || frameData.length === 0) return null;

  return (
    <group>
      {frameData.map((joint, idx) => (
        <mesh
          key={idx}
          position={new THREE.Vector3(joint[0], joint[1], joint[2])}
        >
          <sphereGeometry args={[0.05]} />
          <meshStandardMaterial color="#0ea5e9" />
        </mesh>
      ))}
      {/* Add connections later based on topology */}
    </group>
  );
}

function Scene({ skeletons, isPlaying }: SkeletonViewerProps) {
  const [frameIndex, setFrameIndex] = useState(0);
  const lastTimeRef = useRef(0);
  const FPS = 30;
  const interval = 1000 / FPS;

  useFrame((state) => {
    if (!isPlaying || skeletons.length === 0) return;

    const time = state.clock.getElapsedTime() * 1000;
    if (time - lastTimeRef.current > interval) {
      setFrameIndex((prev) => (prev + 1) % skeletons.length);
      lastTimeRef.current = time;
    }
  });

  const currentFrameData = skeletons[frameIndex] || [];

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <Skeleton frameData={currentFrameData} />
      <Grid infiniteGrid fadeDistance={20} fadeStrength={1.5} />
      <OrbitControls />
    </>
  );
}

export function SkeletonViewer({ skeletons, isPlaying }: SkeletonViewerProps) {
  return (
    <div className="w-full h-100 bg-slate-50 dark:bg-slate-900 rounded-card overflow-hidden border border-border">
      <Canvas camera={{ position: [0, 2, 5], fov: 50 }}>
        <Scene skeletons={skeletons} isPlaying={isPlaying} />
      </Canvas>
    </div>
  );
}
