"use client";

import React, { useRef, useState, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid, Center, Environment } from "@react-three/drei";
import * as THREE from "three";

// PHOENIX-14T Skeleton Connectivity (50 joints)
// [parent, child]
const SKELETON_CONNECTIONS = [
  [1, 0], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [1, 7],
  // Hand Left (8-28)
  [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [8, 13], [13, 14], [14, 15], [15, 16],
  [8, 17], [17, 18], [18, 19], [19, 20], [8, 21], [21, 22], [22, 23], [23, 24], [8, 25], [25, 26], [26, 27], [27, 28],
  // Hand Right (29-49)
  [4, 29], [29, 30], [30, 31], [31, 32], [32, 33], [29, 34], [34, 35], [35, 36], [36, 37],
  [29, 38], [38, 39], [39, 40], [40, 41], [29, 42], [42, 43], [43, 44], [44, 45], [29, 46], [46, 47], [47, 48], [48, 49]
];

interface SkeletonViewerProps {
  skeletons: number[][][]; // [frame][joint][x,y,z]
  isPlaying: boolean;
}

function Bone({ start, end, color }: { start: number[], end: number[], color: string }) {
  const points = useMemo(() => [
    new THREE.Vector3(start[0], start[1], start[2]),
    new THREE.Vector3(end[0], end[1], end[2])
  ], [start, end]);
  
  return (
    <line>
      <bufferGeometry attach="geometry" onUpdate={self => self.setFromPoints(points)} />
      <lineBasicMaterial attach="material" color={color} linewidth={2} />
    </line>
  );
}

function Skeleton({ frameData }: { frameData: number[][] }) {
  if (!frameData || frameData.length === 0) return null;

  return (
    <group>
      {/* Joints */}
      {frameData.map((joint, idx) => {
        const isHand = idx >= 8;
        return (
          <mesh key={`j-${idx}`} position={[joint[0], joint[1], joint[2]]}>
            <sphereGeometry args={[isHand ? 0.02 : 0.04]} />
            <meshStandardMaterial 
              color={isHand ? "#f472b6" : (idx === 0 ? "#fbbf24" : "#6366f1")} 
              emissive={isHand ? "#f472b6" : "#6366f1"}
              emissiveIntensity={0.5}
            />
          </mesh>
        );
      })}

      {/* Bones (Lines) */}
      {SKELETON_CONNECTIONS.map(([p, c], bIdx) => {
        if (!frameData[p] || !frameData[c]) return null;
        const isHand = c >= 8;
        return (
          <Bone 
            key={`b-${bIdx}`} 
            start={frameData[p]} 
            end={frameData[c]} 
            color={isHand ? "#ec4899" : "#4f46e5"} 
          />
        );
      })}
    </group>
  );
}

function Scene({ skeletons, isPlaying }: SkeletonViewerProps) {
  const [frameIndex, setFrameIndex] = useState(0);
  const lastTimeRef = useRef(0);
  const FPS = 25; // Matching common video frame rates
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
      <ambientLight intensity={1} />
      <pointLight position={[10, 10, 10]} intensity={2} />
      <pointLight position={[-10, -10, -10]} intensity={1} color="#4f46e5" />
      
      <Center top>
        <Skeleton frameData={currentFrameData} />
      </Center>
      
      <Grid 
        infiniteGrid 
        fadeDistance={30} 
        fadeStrength={1} 
        sectionSize={1}
        sectionColor="#1e293b"
        cellColor="#334155"
      />
      
      <OrbitControls makeDefault minDistance={1} maxDistance={10} />
      <Environment preset="city" />
    </>
  );
}

export function SkeletonViewer({ skeletons, isPlaying }: SkeletonViewerProps) {
  return (
    <div className="w-full h-full bg-slate-950 rounded-lg overflow-hidden relative">
      <Canvas camera={{ position: [0, 1, 3], fov: 45 }}>
        <Scene skeletons={skeletons} isPlaying={isPlaying} />
      </Canvas>
      <div className="absolute top-4 left-4 flex gap-2">
        <div className="flex items-center gap-2 px-2 py-1 rounded bg-black/50 border border-white/10 text-[10px] text-zinc-400">
          <div className="w-2 h-2 rounded-full bg-indigo-500" /> BODY
        </div>
        <div className="flex items-center gap-2 px-2 py-1 rounded bg-black/50 border border-white/10 text-[10px] text-zinc-400">
          <div className="w-2 h-2 rounded-full bg-pink-500" /> HANDS
        </div>
      </div>
    </div>
  );
}
