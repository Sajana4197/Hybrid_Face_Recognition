import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  Camera,
  ShieldCheck,
  Wifi,
  WifiOff,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader,
  Info,
} from "lucide-react";
import clsx from "clsx";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5001";

// --- Main App Component ---
export default function App() {
  const videoRef = useRef(null);
  const [decision, setDecision] = useState("");
  const [personId, setPersonId] = useState("");
  const [resultData, setResultData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState("unknown");

  // NEW STATE: Manages the glow color ('none', 'red', or 'green')
  const [glowColor, setGlowColor] = useState("none");
  const glowTimerRef = useRef(null); // Ref to hold the timer ID

  // Initialize camera
  useEffect(() => {
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (e) {
        console.error("Camera error:", e);
      }
    })();
  }, []);

  // Backend health check
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const r = await fetch(`${API_BASE}/health`);
        setHealth(r.ok ? "ok" : "down");
      } catch {
        setHealth("down");
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // MODIFIED: Logic to handle both red and green glows
  useEffect(() => {
    // Clear any existing timer when a new decision is made
    if (glowTimerRef.current) {
      clearTimeout(glowTimerRef.current);
    }

    if (decision === "MATCH") {
      setGlowColor("red");
      new Audio("/match-sound.wav")
        .play()
        .catch((e) => console.error("Error playing sound:", e));
    } else if (decision === "NO_MATCH") {
      setGlowColor("green");
      new Audio("/no-match-sound.mp3")
        .play()
        .catch((e) => console.error("Error playing sound:", e));
    } else {
      setGlowColor("none"); // No glow for other states
      return; // Exit early if no glow is needed
    }

    // Set a 5-second timer to turn off the glow
    glowTimerRef.current = setTimeout(() => {
      setGlowColor("none");
    }, 5000);

    // Cleanup function to clear the timer if the component unmounts
    return () => {
      if (glowTimerRef.current) {
        clearTimeout(glowTimerRef.current);
      }
    };
  }, [decision, resultData]);

  async function captureAndMatch() {
    if (busy) return;

    // MODIFICATION: Stop any active glow when a new scan starts
    setGlowColor("none");
    if (glowTimerRef.current) {
      clearTimeout(glowTimerRef.current);
    }

    setBusy(true);
    setDecision("Capturing...");
    setResultData(null);

    const video = videoRef.current;
    const frames = [];
    try {
      for (let i = 0; i < 5; i++) {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise((resolve) =>
          canvas.toBlob(resolve, "image/jpeg", 0.9)
        );
        frames.push(blob);
        await new Promise((r) => setTimeout(r, 80));
      }

      const form = new FormData();
      frames.forEach((b, idx) => form.append("files", b, `frame${idx}.jpg`));
      const res = await axios.post(`${API_BASE}/match_multi`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = res.data;
      setResultData(data);
      setDecision(data.best_decision || data.decision || "UNKNOWN");
      setPersonId(data.best_person_id || "");
    } catch (e) {
      console.error(e);
      setDecision("ERROR");
      setResultData(null);
    } finally {
      setBusy(false);
    }
  }

  // Map state to CSS classes for clean rendering
  const glowClasses = {
    red: "animate-red-glow",
    green: "animate-green-glow",
  };

  return (
    <div className="h-screen bg-gray-900 text-gray-200 font-sans flex flex-col p-4 sm:p-6 lg:p-8">
      <div className="w-full max-w-6xl mx-auto flex flex-col flex-1">
        <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-8 h-8 text-cyan-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Border Control Field Client
            </h1>
          </div>
          <HealthStatus health={health} />
        </header>

        {/* MODIFICATION: Dynamically apply glow class based on state */}
        <main
          className={clsx(
            "flex-1 bg-gray-800/50 rounded-2xl shadow-2xl ring-1 ring-white/10 backdrop-blur-sm p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-hidden",
            glowClasses[glowColor] // This applies 'animate-red-glow' or 'animate-green-glow'
          )}
        >
          <div className="flex flex-col items-center justify-center gap-3">
            {/* Outer container for the animated gradient border */}
            <div
              className="relative w-auto max-h-[480px] aspect-[4/5] rounded-xl p-[4px] overflow-hidden shadow-lg 
               bg-gradient-to-r from-cyan-400 via-blue-700 to-red-600 
               bg-300% animate-gradient-move"
            >
              {/* Inner container that provides the dark background */}
              <div className="w-full h-full bg-gray-900 rounded-[10px] overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full h-full object-cover transform scale-x-[-1]"
                />
              </div>
            </div>
            <p className="text-sm text-gray-400">
              Live camera feed — ensure face is centered
            </p>
          </div>
          <div className="flex flex-col items-center gap-4">
            <button
              onClick={captureAndMatch}
              disabled={busy}
              className={clsx(
                "w-full max-w-sm flex items-center justify-center gap-3 px-6 py-4 rounded-lg font-semibold shadow-lg transition-all duration-300",
                "text-lg text-white transform hover:scale-105",
                busy
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
              )}
            >
              {busy ? <Loader className="animate-spin" /> : <Camera />}
              {busy ? "Processing..." : "Capture & Match"}
            </button>

            <AnimatePresence mode="wait">
              <AnimatePresence mode="wait">
                <DecisionDisplay
                  key={resultData ? resultData.best_person_id : decision}
                  decision={decision}
                  glow={glowColor} // <-- ADD THIS PROP
                />
              </AnimatePresence>
            </AnimatePresence>

            <AnimatePresence>
              {resultData && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5 }}
                  className="w-full max-w-sm"
                >
                  <ResultsSummary resultData={resultData} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </main>

        <footer className="text-xs text-gray-500 pt-6 text-center">
          © 2025 Border Security AI — Hybrid Face Recognition (NH + HDIC)
        </footer>
      </div>
    </div>
  );
}

// --- Sub-components (No changes below this line) ---

const HealthStatus = ({ health }) => {
  const isOk = health === "ok";
  return (
    <div
      className={clsx(
        "flex items-center gap-2 text-sm px-3 py-1.5 rounded-full font-medium",
        isOk ? "bg-green-500/20 text-green-300" : "bg-red-500/20 text-red-300"
      )}
      title="Backend health"
    >
      {isOk ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
      {isOk ? "Backend Connected" : "Backend Down"}
    </div>
  );
};

const DecisionDisplay = ({ decision, glow }) => {
  // <-- 1. Add 'glow' here
  const config = {
    MATCH: {
      icon: <CheckCircle2 size={28} />,
      style:
        "bg-red-500/20 border-red-500/50 text-red-200 items items-center justify-center",
      title: "MATCH FOUND",
    },
    NO_MATCH: {
      icon: <XCircle size={28} />,
      style:
        "bg-green-500/20 border-green-500/50 text-green-200 items-center justify-center",
      title: "NO MATCH FOUND",
    },
    NO_FACE: {
      icon: <AlertTriangle size={28} />,
      style:
        "bg-yellow-500/20 border-yellow-500/50 text-yellow-200 items-center justify-center",
      title: "No Face Detected",
    },
    ERROR: {
      icon: <XCircle size={28} />,
      style:
        "bg-gray-500/20 border-gray-500/50 text-gray-200 items-center justify-center",
      title: "Processing Error",
    },
    "Capturing...": {
      icon: <Loader size={28} className="animate-spin" />,
      style: "bg-sky-500/20 border-sky-500/50 text-sky-200",
      title: "Capturing Frames...",
    },
    default: {
      icon: <Info size={28} />,
      style: "bg-gray-700/60 border-gray-600/80 text-gray-300",
      title: "Ready for Facial Scan",
      text: "Click 'Capture & Match' to begin.",
    },
  };

  const current = config[decision] || config.default;

  const glowClasses = {
    red: "animate-red-glow",
    green: "animate-green-glow",
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.3 }}
      // 2. Modify clsx to add the glow class from the prop
      className={clsx(
        "w-full max-w-sm flex items-center gap-4 p-4 rounded-lg border",
        current.style,
        glowClasses[glow] // <-- ADD THIS LINE
      )}
    >
      <div className="flex-shrink-0">{current.icon}</div>
      <div>
        <h3 className="font-bold text-lg leading-tight">{current.title}</h3>
        {current.text && <p className="text-sm opacity-80">{current.text}</p>}
      </div>
    </motion.div>
  );
};

const ResultsSummary = ({ resultData }) => (
  <div className="grid grid-cols-2 gap-3 text-sm text-gray-300 bg-gray-900/50 p-4 rounded-lg ring-1 ring-white/10">
    <div>
      Frames processed: <b className="text-white">{resultData.frames}</b>
    </div>
    <div>
      Matches found: <b className="text-white">{resultData.match_frames}</b>
    </div>
    <div>
      Matching ID:{" "}
      <b className="ml-1 font-mono text-cyan-400">
        {resultData.decision === "MATCH" ? resultData.best_person_id : "N/A"}
      </b>
    </div>
    {resultData.best_score && (
      <div>
        Best Score:{" "}
        <b className="text-cyan-400">{resultData.best_score.toFixed(3)}</b>
      </div>
    )}
  </div>
);
