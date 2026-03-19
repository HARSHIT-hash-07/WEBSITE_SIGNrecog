"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Search, Loader2, Play, AlertCircle } from "lucide-react";

interface VideoEntry {
  name: string;
  video_url: string;
}

export function TextToSignClient() {
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<VideoEntry[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);
    setResults([]);
    setSelectedVideo(null);

    try {
      const response = await fetch(`http://127.0.0.1:8000/search?q=${encodeURIComponent(inputText)}`, {
        method: "GET",
      });

      if (!response.ok) {
        throw new Error("Failed to search videos.");
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      setError(
        "Something went wrong. Please check if the backend API is running at http://127.0.0.1:8000",
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectVideo = (videoName: string) => {
    setSelectedVideo(`http://127.0.0.1:8000/video/${videoName}`);
  };

  return (
    <div className="flex flex-col lg:flex-row gap-8">
      {/* Search Section */}
      <div className="flex-1 space-y-4">
        <div className="card h-full min-h-100 flex flex-col">
          <h2 className="text-xl font-semibold mb-4 text-white">Search Sign Videos</h2>
          
          <div className="flex flex-col sm:flex-row gap-3 mb-6">
            <div className="relative flex-1">
              <input
                type="text"
                className="w-full bg-slate-50 dark:bg-slate-950 border border-input rounded-md px-4 py-3 text-base text-slate-900 dark:text-slate-50 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-ring/50"
                placeholder="Search for a topic (e.g. 'tagesschau', 'April', '2010')..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              />
            </div>
            <Button
              onClick={handleSearch}
              isLoading={loading}
              disabled={!inputText.trim()}
              className="py-3 px-6 h-auto"
            >
              <Search className="w-4 h-4 mr-2" />
              Search
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
                {hasSearched ? (
                  loading ? "Searching..." : `Results (${results.length})`
                ) : (
                  "Suggestions"
                )}
              </h3>
            </div>
            
            <div className="flex-1 overflow-y-auto max-h-[400px] p-2">
              {loading ? (
                 <div className="flex justify-center items-center h-32">
                   <Loader2 className="w-6 h-6 animate-spin text-zinc-500" />
                 </div>
              ) : hasSearched && results.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-8 text-center text-zinc-500 h-full">
                  <Search className="w-8 h-8 mb-3 opacity-20" />
                  <p>No videos found matching "{inputText}".</p>
                  <p className="text-sm mt-1">Try a different date or keyword.</p>
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
                  {['tagesschau', 'heute', 'April 2010', 'January', 'Monday'].map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => {
                        setInputText(suggestion);
                        setTimeout(() => handleSearch(), 10);
                      }}
                      className="px-3 py-1.5 rounded-full bg-zinc-800/50 border border-zinc-700/50 text-xs text-zinc-400 hover:text-white hover:bg-zinc-700 hover:border-zinc-600 transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Video Player Section */}
      <div className="flex-1 space-y-4">
        <div className="card h-full min-h-100 flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">Sign Animation</h2>
            {selectedVideo && (
              <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-green-500/10 text-green-400 border border-green-500/20">
                Playing
              </span>
            )}
          </div>

          <div className="flex-1 min-h-[400px] relative rounded-lg overflow-hidden bg-black/40 border border-zinc-800/50 flex items-center justify-center shadow-inner">
            {selectedVideo ? (
              <video
                key={selectedVideo} // Forces video to reload when source changes
                src={selectedVideo}
                controls
                autoPlay
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center text-zinc-500 max-w-sm mx-auto">
                <div className="w-16 h-16 rounded-full bg-zinc-900/50 flex items-center justify-center mb-4 border border-zinc-800">
                  <Play className="w-6 h-6 ml-1 opacity-50" />
                </div>
                <p className="text-zinc-300 font-medium mb-2">No Video Selected</p>
                <p className="text-sm">
                  Search for a topic and select a video from the results to play the sign language animation.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

