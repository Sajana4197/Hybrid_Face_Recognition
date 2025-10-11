import React, { useEffect, useRef, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5001";

export default function App() {
  const videoRef = useRef(null);
  const [decision, setDecision] = useState("");
  const [personId, setPersonId] = useState("");
  const [scores, setScores] = useState(null);
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState("unknown");

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

  useEffect(() => {
    // ping backend health once
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/health`);
        setHealth(r.ok ? "ok" : "down");
      } catch {
        setHealth("down");
      }
    })();
  }, []);

  async function captureAndMatch() {
    if (busy) return;
    setBusy(true);
    try {
      const canvas = document.createElement("canvas");
      const video = videoRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.9)
      );

      const form = new FormData();
      form.append("file", blob, "frame.jpg");

      const res = await axios.post(`${API_BASE}/match`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = res.data || {};
      setDecision(data.decision || "ERROR");
      setPersonId(data.person_id || "");
      setScores(data.scores || {});
    } catch (e) {
      console.error(e);
      setDecision("ERROR");
      setPersonId("");
      setScores({});
    } finally {
      setBusy(false);
    }
  }

  const bannerClasses = {
    MATCH: "bg-green-600",
    NO_MATCH: "bg-red-600",
    NO_FACE: "bg-yellow-500",
    ERROR: "bg-gray-700",
    default: "bg-slate-500",
  };

  const decisionText =
    decision === ""
      ? "Ready"
      : decision === "MATCH"
      ? `✅ MATCH: ${personId} — Sfinal = ${
          scores?.Sfinal !== undefined ? scores.Sfinal.toFixed(3) : "N/A"
        }`
      : decision === "NO_MATCH"
      ? "❌ NO MATCH FOUND"
      : decision === "NO_FACE"
      ? "🟧 No Face Detected"
      : decision === "ERROR"
      ? "🟥 Error"
      : "Ready";

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col items-center py-8">
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-6">
          <h1 className="text-2xl font-bold tracking-wide">
            BORDER CONTROL — Field Client (Parallel NH + HDIC)
          </h1>
          <span
            className={`inline-flex items-center gap-2 text-sm px-3 py-1 rounded-full ${
              health === "ok" ? "bg-emerald-600" : "bg-rose-600"
            }`}
            title="Backend health"
          >
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                health === "ok" ? "bg-emerald-300" : "bg-rose-300"
              }`}
            />
            {health === "ok" ? "Backend Connected" : "Backend Down"}
          </span>
        </div>

        {/* Card */}
        <div className="bg-slate-800/70 rounded-2xl shadow-xl ring-1 ring-slate-700 p-5">
          {/* Video */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full rounded-lg border border-slate-700 shadow-md"
          />

          {/* Controls */}
          <div className="flex justify-center mt-5">
            <button
              onClick={captureAndMatch}
              disabled={busy}
              className={`px-6 py-3 rounded-lg font-semibold shadow-lg transition ${
                busy
                  ? "bg-slate-600 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              {busy ? "Processing…" : "Capture & Match"}
            </button>
          </div>

          {/* Decision Banner */}
          <div
            className={`mt-6 text-white text-center py-3 text-lg font-bold rounded-lg ${
              bannerClasses[decision] || bannerClasses.default
            }`}
          >
            {decisionText}
          </div>

          {/* Scores */}
          {scores && Object.keys(scores).length > 0 && (
            <div className="mt-4 text-sm bg-slate-900/60 rounded-lg p-4 border border-slate-700">
              <p className="font-semibold mb-2 text-slate-200">Score Details</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1">
                <div>
                  <b>d_NH</b>: {scores?.d_nh ?? "N/A"}
                </div>
                <div>
                  <b>d_HDiC</b>: {scores?.d_hdic ?? "N/A"}
                </div>
                <div>
                  <b>Snh</b>:{" "}
                  {scores?.Snh !== undefined ? scores.Snh.toFixed(3) : "N/A"}
                </div>
                <div>
                  <b>Shdic_norm</b>:{" "}
                  {scores?.Shdic_norm !== undefined
                    ? scores.Shdic_norm.toFixed(3)
                    : "N/A"}
                </div>
                <div className="sm:col-span-2">
                  <b>Sfinal</b>:{" "}
                  {scores?.Sfinal !== undefined
                    ? scores.Sfinal.toFixed(3)
                    : "N/A"}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <p className="text-xs text-slate-400 mt-6 text-center">
          © 2025 Border Security AI — Hybrid Face Recognition (NH + HDIC)
        </p>
      </div>
    </div>
  );
}
