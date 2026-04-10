"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/Button";
import { Search, Loader2, Play, AlertCircle, Heart, ThumbsUp, ThumbsDown, Sparkles } from "lucide-react";
import { createClient } from "@/utils/supabase/client";
import { SkeletonViewer } from "./SkeletonViewer";

interface VideoEntry {
  name: string;
  video_url: string;
}

export function TextToSignClient() {
  const [inputText, setInputText] = useState("");
  const [mode, setMode] = useState<"search" | "generate">("search");
  const [loading, setLoading] = useState(false);
  const [generativeLoading, setGenerativeLoading] = useState(false);
  const [results, setResults] = useState<VideoEntry[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [skeletons, setSkeletons] = useState<number[][][] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processedText, setProcessedText] = useState("");
  const [user, setUser] = useState<import("@supabase/supabase-js").User | null>(null);

  const supabase = createClient();

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => setUser(user));
  }, []);

  const handleSearch = async () => {
    if (!inputText.trim()) return;

    if (mode === "generate") {
      await handleGenerate();
      return;
    }

    setLoading(true);
    setError(null);
    setHasSearched(true);
    setResults([]);
    setSelectedVideo(null);
    setSkeletons(null);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8001";
      const response = await fetch(`${baseUrl}/search?q=${encodeURIComponent(inputText)}`, {
        method: "GET",
      });

      if (!response.ok) {
        throw new Error("Failed to search videos.");
      }

      const data = await response.json();
      setResults(data.results || []);

      // Log search query async
      supabase.from("search_history").insert([{ query: inputText, user_id: user?.id || null }]).then();
    } catch (err) {
      setError(
        `Something went wrong. Please check if the sign-idd-api is running at ${process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8001"}`,
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!inputText.trim()) return;

    setGenerativeLoading(true);
    setError(null);
    setHasSearched(true);
    setSkeletons(null);
    setSelectedVideo(null);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8001";
      const response = await fetch(`${baseUrl}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Translation service error" }));
        throw new Error(errData.detail || "Failed to generate sign language.");
      }

      const data = await response.json();
      if (data.video_url) {
        setSelectedVideo(data.video_url);
        setSkeletons(null);
      } else {
        setSkeletons(data.skeletons);
      }
      setProcessedText(data.text_processed);
      
      // Log generation for history
      supabase.from("search_history").insert([{ query: `[GEN] ${inputText}`, user_id: user?.id || null }]).then();
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      setError(
        `Generative Error: ${errorMessage}. Ensure SignBridge AI Engine is running at ${process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8001"}`
      );
      console.error(err);
    } finally {
      setGenerativeLoading(false);
    }
  };

  const handleSelectVideo = (videoName: string) => {
    const baseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8001";
    setSelectedVideo(`${baseUrl}/video/${videoName}`);
  };
  return (
    <div className="flex flex-col lg:flex-row gap-8 h-[calc(100vh-200px)]" style={{ minHeight: '500px' }}>
      {/* Search Section */}
      <div className="flex-1 space-y-4">
        <div className="card h-full flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">Input Translation</h2>
            <div className="flex bg-zinc-900 rounded-lg p-1 border border-zinc-800">
              <button 
                onClick={() => setMode("search")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${mode === "search" ? "bg-indigo-600 text-white shadow-lg" : "text-zinc-500 hover:text-zinc-300"}`}
              >
                Search Videos
              </button>
              <button 
                onClick={() => setMode("generate")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 ${mode === "generate" ? "bg-purple-600 text-white shadow-lg" : "text-zinc-500 hover:text-zinc-300"}`}
              >
                <Sparkles className="w-3 h-3" />
                Live AI Bridge
              </button>
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-3 mb-6">
            <div className="relative flex-1">
              <input
                type="text"
                className="w-full bg-slate-50 dark:bg-slate-950 border border-input rounded-md px-4 py-3 text-base text-slate-900 dark:text-slate-50 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-ring/50"
                placeholder={mode === "search" ? "Search for topics..." : "Type anything to translate to sign..."}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              />
            </div>
            <Button
              onClick={handleSearch}
              isLoading={loading || generativeLoading}
              disabled={!inputText.trim()}
              className="py-3 px-6 h-auto"
            >
              {mode === "search" ? <Search className="w-4 h-4 mr-2" /> : <Sparkles className="w-4 h-4 mr-2" />}
              {mode === "search" ? "Search" : "Generate"}
            </Button>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 text-red-400 rounded-md text-sm flex items-start gap-3">
              <AlertCircle className="w-5 h-5 shrink-0" />
              <p>{error}</p>
            </div>
          )}

          {/* Results Pane */}
          <div className="flex-1 flex flex-col border border-zinc-800/50 rounded-lg overflow-hidden bg-zinc-900/30">
            <div className="p-4 border-b border-zinc-800/50 bg-zinc-900/50">
              <h3 className="text-sm font-medium text-zinc-300">
                {mode === "search" ? (
                  hasSearched ? (loading ? "Searching..." : `Results (${results.length})`) : "Common Topics"
                ) : (
                  hasSearched ? (generativeLoading ? "Generating Path..." : "AI Generated Stream") : "Dictionary Overview"
                )}
              </h3>
            </div>
            
            <div className="flex-1 overflow-y-auto p-2">
              {mode === "search" ? (
                loading ? (
                   <div className="flex justify-center items-center h-32">
                     <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
                   </div>
                ) : hasSearched && results.length === 0 ? (
                  <div className="flex flex-col items-center justify-center p-8 text-center text-zinc-500 h-full">
                    <Search className="w-8 h-8 mb-3 opacity-20" />
                    <p>No videos found matching &quot;{inputText}&quot;.</p>
                  </div>
                ) : hasSearched ? (
                  <div className="space-y-1">
                    {results.map((video) => (
                      <button
                        key={video.name}
                        onClick={() => handleSelectVideo(video.name)}
                        className={`w-full text-left px-4 py-3 rounded-md transition-colors text-sm flex items-center gap-3 ${
                          selectedVideo?.includes(video.name) 
                            ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30" 
                            : "hover:bg-zinc-800/50 text-zinc-300 border border-transparent"
                        }`}
                      >
                        <Play className={`w-4 h-4 shrink-0 ${selectedVideo?.includes(video.name) ? "text-indigo-400" : "text-zinc-500"}`} />
                        <span className="truncate">{video.name}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 flex flex-wrap gap-2">
                    {['tagesschau', 'heute', 'April 2010', 'Monday'].map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => { setInputText(suggestion); setTimeout(() => handleSearch(), 10); }}
                        className="px-3 py-1.5 rounded-full bg-zinc-800/50 border border-zinc-700/50 text-xs text-zinc-400 hover:text-white"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                )
              ) : (
                <div className="p-4 space-y-4">
                  {generativeLoading ? (
                    <div className="flex flex-col items-center justify-center h-32 space-y-3">
                      <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
                      <p className="text-zinc-500 text-sm">Diffusion sampling in progress...</p>
                    </div>
                  ) : hasSearched && skeletons ? (
                    <div className="bg-purple-500/5 border border-purple-500/20 p-4 rounded-lg">
                      <p className="text-xs text-purple-400 mb-1 font-mono uppercase">Output Sequence</p>
                      <p className="text-white text-lg font-medium tracking-tight">
                        {processedText || inputText}
                      </p>
                      <div className="mt-4 flex gap-2">
                         <div className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 text-[10px] boder border-purple-500/10">ACD_DIFFUSION</div>
                         <div className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-500 text-[10px]">{skeletons.length} frames</div>
                      </div>
                    </div>
                  ) : (
                    <div className="p-2 text-zinc-500 text-sm leading-relaxed">
                      <p className="mb-4">This mode uses the **Sign-IDD Diffusion Model** to generate 3D motion on the fly.</p>
                      <p className="font-semibold text-zinc-400 mb-2">Supported Vocabulary:</p>
                      <div className="flex flex-wrap gap-1.5">
                        {['Wetter', 'Regen', 'Sonne', 'Today', 'Tomorrow', 'Monday', 'North', 'South'].map(v => (
                          <span key={v} className="px-2 py-1 rounded bg-zinc-800 text-[10px]">{v}</span>
                        ))}
                        <span className="px-2 py-1 rounded bg-zinc-800 text-[10px] opacity-50">+ 1000 more...</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Video Player Section */}
      <div className="flex-1 space-y-4">
        <div className="card h-full flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">Sign Animation</h2>
            {selectedVideo && (
              <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-green-500/10 text-green-400 border border-green-500/20">
                Playing
              </span>
            )}
          </div>

          <div className="flex-1 relative rounded-lg overflow-hidden bg-black/40 border border-zinc-800/50 flex items-center justify-center shadow-inner">
            {skeletons ? (
              <SkeletonViewer skeletons={skeletons} isPlaying={true} />
            ) : selectedVideo ? (
              <video
                key={selectedVideo} // Forces video to reload when source changes
                controls
                autoPlay
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-contain bg-black"
              >
                <source src={selectedVideo} type="video/mp4" />
              </video>
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center text-zinc-500 max-w-sm mx-auto">
                <div className="w-16 h-16 rounded-full bg-zinc-900/50 flex items-center justify-center mb-4 border border-zinc-800">
                  {mode === "generate" ? <Sparkles className="w-6 h-6 opacity-50 text-purple-400" /> : <Play className="w-6 h-6 ml-1 opacity-50" />}
                </div>
                <p className="text-zinc-300 font-medium mb-2">
                  {mode === "generate" ? "Ready to Generate" : "No Video Selected"}
                </p>
                <p className="text-sm">
                  {mode === "generate" 
                    ? "Type your message and click 'Generate' to see the AI diffusion model create a 3D sign language stream."
                    : "Search for a topic and select a video from the results to play the sign language animation."}
                </p>
              </div>
            )}
          </div>

          {selectedVideo && (
            <div className="mt-4 flex items-center justify-between border border-zinc-800/50 bg-zinc-900/30 p-4 rounded-lg">
              <div className="flex items-center gap-4">
                <Button 
                  onClick={async () => {
                    if (!user) { alert("Please login to save favorites!"); return; }
                    let videoName = selectedVideo;
                    if (selectedVideo.includes("/video/")) videoName = selectedVideo.split("/video/")[1];
                    else if (selectedVideo.includes("/static/")) videoName = selectedVideo.split("/static/")[1];
                    else videoName = selectedVideo.split("/").pop() || selectedVideo;
                    
                    const { error } = await supabase.from("favorites").insert([{ video_name: videoName, user_id: user.id }]);
                    if (error) {
                      console.error("Supabase error:", error);
                      if (error.code === '23505') alert("Already in favorites!");
                      else alert("Error saving favorite");
                    } else {
                      alert("Saved to favorites!");
                    }
                  }}
                  className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700"
                >
                  <Heart className="w-4 h-4 mr-2" />
                  Save to Favorites
                </Button>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-zinc-500 mr-2 hidden sm:inline">Was this accurate?</span>
                <Button 
                  size="icon" 
                  className="bg-zinc-800 hover:bg-green-600/20 text-zinc-400 hover:text-green-400 border border-zinc-700"
                  onClick={() => {
                    let videoName = selectedVideo;
                    if (selectedVideo.includes("/video/")) videoName = selectedVideo.split("/video/")[1];
                    else if (selectedVideo.includes("/static/")) videoName = selectedVideo.split("/static/")[1];
                    else videoName = selectedVideo.split("/").pop() || selectedVideo;
                    
                    supabase.from("feedback").insert([{ video_name: videoName, user_id: user?.id || null, is_positive: true }]).then(({ error }) => {
                      if (error) console.error("Feedback error:", error);
                      else alert("Thanks for your feedback!");
                    });
                  }}
                >
                  <ThumbsUp className="w-4 h-4" />
                </Button>
                <Button 
                  size="icon" 
                  className="bg-zinc-800 hover:bg-red-600/20 text-zinc-400 hover:text-red-400 border border-zinc-700"
                  onClick={() => {
                    let videoName = selectedVideo;
                    if (selectedVideo.includes("/video/")) videoName = selectedVideo.split("/video/")[1];
                    else if (selectedVideo.includes("/static/")) videoName = selectedVideo.split("/static/")[1];
                    else videoName = selectedVideo.split("/").pop() || selectedVideo;
                    
                    supabase.from("feedback").insert([{ video_name: videoName, user_id: user?.id || null, is_positive: false }]).then(({ error }) => {
                      if (error) console.error("Feedback error:", error);
                      else alert("Thanks for your feedback!");
                    });
                  }}
                >
                  <ThumbsDown className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

