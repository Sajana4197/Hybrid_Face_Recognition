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

  const [glowColor, setGlowColor] = useState("none");
  const glowTimerRef = useRef(null);

  const [adminStatus, setAdminStatus] = useState(null);
  const pollRef = useRef(null);

  // Initialize camera
  useEffect(() => {
    async function initCamera() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter((d) => d.kind === "videoinput");
        if (videoDevices.length === 0)
          throw new Error("No camera devices found");

        const preferredDevice =
          videoDevices.find((d) => d.label.toLowerCase().includes("usb")) ||
          videoDevices[0];
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: preferredDevice.deviceId },
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        console.error("Camera initialization failed:", err);
        alert("Could not access camera: " + err.message);
      }
    }
    initCamera();
  }, []);

  // Health check
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

  // Glow control
  useEffect(() => {
    if (glowTimerRef.current) clearTimeout(glowTimerRef.current);

    if (decision === "MATCH") {
      setGlowColor("red");
      new Audio("/match-sound.wav").play().catch(() => {});
    } else if (decision === "NO_MATCH") {
      setGlowColor("green");
      new Audio("/no-match-sound.mp3").play().catch(() => {});
    } else if (decision === "ADMIN_CONFIRMED") {
      setGlowColor("green");
      new Audio("/approved.mp3").play().catch(() => {});
    } else if (decision === "ADMIN_REJECTED") {
      setGlowColor("red");
      new Audio("/rejected.mp3").play().catch(() => {});
    } else {
      setGlowColor("none");
      return;
    }

    glowTimerRef.current = setTimeout(() => setGlowColor("none"), 5000);
    return () => clearTimeout(glowTimerRef.current);
  }, [decision]);

  // Poll Admin decision
  async function pollAdminDecision(pid, ts) {
    if (!pid || !ts) return;
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(
          `${API_BASE}/check_status?person_id=${pid}&timestamp=${ts}`
        );
        const data = await res.json();
        if (data.status === "confirmed") {
          clearInterval(pollRef.current);
          setAdminStatus("confirmed");
          setDecision("ADMIN_CONFIRMED");
        } else if (data.status === "rejected") {
          clearInterval(pollRef.current);
          setAdminStatus("rejected");
          setDecision("ADMIN_REJECTED");
        }
      } catch (e) {
        console.warn("Polling error:", e);
      }
    }, 5000);
  }

  async function captureAndMatch() {
    if (busy) return;
    setGlowColor("none");
    if (glowTimerRef.current) clearTimeout(glowTimerRef.current);

    setBusy(true);
    setDecision("Capturing...");
    setResultData(null);
    setAdminStatus(null);

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

      // Start polling for admin confirmation if match
      if (data.decision === "MATCH" && data.best_person_id && data.timestamp) {
        pollAdminDecision(data.best_person_id, data.timestamp);
      }
    } catch (e) {
      console.error(e);
      setDecision("ERROR");
      setResultData(null);
    } finally {
      setBusy(false);
    }
  }

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

        <main
          className={clsx(
            "flex-1 bg-gray-800/50 rounded-2xl shadow-2xl ring-1 ring-white/10 backdrop-blur-sm p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-hidden",
            glowClasses[glowColor]
          )}
        >
          <div className="flex flex-col items-center justify-center gap-3">
            <div
              className="relative w-auto max-h-[480px] aspect-[4/5] rounded-xl p-[4px] overflow-hidden shadow-lg 
               bg-gradient-to-r from-cyan-400 via-blue-700 to-red-600 
               bg-300% animate-gradient-move"
            >
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
              <DecisionDisplay
                key={resultData ? resultData.best_person_id : decision}
                decision={decision}
                glow={glowColor}
                adminStatus={adminStatus}
              />
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

// --- Sub-components (unchanged, except new admin statuses added) ---

const HealthStatus = ({ health }) => {
  const isOk = health === "ok";
  return (
    <div
      className={clsx(
        "flex items-center gap-2 text-sm px-3 py-1.5 rounded-full font-medium",
        isOk ? "bg-green-500/20 text-green-300" : "bg-red-500/20 text-red-300"
      )}
    >
      {isOk ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
      {isOk ? "Backend Connected" : "Backend Down"}
    </div>
  );
};

const DecisionDisplay = ({ decision, glow, adminStatus }) => {
  const config = {
    MATCH: {
      icon: <CheckCircle2 size={28} />,
      style: "bg-red-500/20 border-red-500/50 text-red-200",
      title: "MATCH FOUND",
    },
    NO_MATCH: {
      icon: <XCircle size={28} />,
      style: "bg-green-500/20 border-green-500/50 text-green-200",
      title: "NO MATCH FOUND",
    },
    NO_FACE: {
      icon: <AlertTriangle size={28} />,
      style: "bg-yellow-500/20 border-yellow-500/50 text-yellow-200",
      title: "No Face Detected",
    },
    ERROR: {
      icon: <XCircle size={28} />,
      style: "bg-gray-500/20 border-gray-500/50 text-gray-200",
      title: "Processing Error",
    },
    ADMIN_CONFIRMED: {
      icon: <CheckCircle2 size={28} />,
      style: "bg-green-500/30 border-green-500/50 text-green-200",
      title: "ADMIN CONFIRMED MATCH",
    },
    ADMIN_REJECTED: {
      icon: <XCircle size={28} />,
      style: "bg-red-500/30 border-red-500/50 text-red-200",
      title: "ADMIN REJECTED MATCH",
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
  const glowClasses = { red: "animate-red-glow", green: "animate-green-glow" };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.3 }}
      className={clsx(
        "w-full max-w-sm flex items-center gap-4 p-4 rounded-lg border",
        current.style,
        glowClasses[glow]
      )}
    >
      <div className="flex-shrink-0">{current.icon}</div>
      <div>
        <h3 className="font-bold text-lg leading-tight">{current.title}</h3>
        {adminStatus && (
          <p className="text-sm opacity-80 mt-1">
            Admin status:{" "}
            <b
              className={
                adminStatus === "confirmed" ? "text-green-300" : "text-red-300"
              }
            >
              {adminStatus}
            </b>
          </p>
        )}
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
