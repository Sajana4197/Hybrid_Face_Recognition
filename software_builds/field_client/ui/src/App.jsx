import React, { useEffect, useRef, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5001";

export default function App() {
  const videoRef = useRef(null);
  const [decision, setDecision] = useState("");
  const [personId, setPersonId] = useState("");
  const [resultData, setResultData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState("unknown");

  // --- Initialize camera ---
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

  // --- Backend health check ---
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/health`);
        setHealth(r.ok ? "ok" : "down");
      } catch {
        setHealth("down");
      }
    })();
  }, []);

  // --- Capture and match multiple frames ---
  async function captureAndMatch() {
    if (busy) return;
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
        await new Promise((r) => setTimeout(r, 200)); // 0.2s gap
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

  // --- UI Styling ---
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
      ? `✅ MATCH: ${personId} — Best Sfinal = ${
          resultData?.best_score ? resultData.best_score.toFixed(3) : "N/A"
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
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full rounded-lg border border-slate-700 shadow-md"
          />

          {/* Button */}
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
              {busy ? "Processing…" : "Capture & Match (5 Frames)"}
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

          {/* Frame Summary */}
          {resultData && (
            <div className="mt-2 text-sm text-slate-300">
              <p>
                Matched frames: <b>{resultData.match_frames}</b> /{" "}
                {resultData.frames} →{" "}
                <b
                  className={
                    resultData.decision === "MATCH"
                      ? "text-emerald-400"
                      : "text-rose-400"
                  }
                >
                  {resultData.decision}
                </b>
              </p>
            </div>
          )}

          {/* Frame-by-frame Details */}
          {resultData?.frame_details?.length > 0 && (
            <div className="mt-5 bg-slate-900/60 rounded-lg p-4 border border-slate-700">
              <p className="font-semibold mb-2 text-slate-200">
                Frame-by-frame Results:
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse border border-slate-700">
                  <thead>
                    <tr className="bg-slate-700 text-slate-100">
                      <th className="p-2 border border-slate-600">Frame</th>
                      <th className="p-2 border border-slate-600">Decision</th>
                      <th className="p-2 border border-slate-600">Sfinal</th>
                      <th className="p-2 border border-slate-600">d_NH</th>
                      <th className="p-2 border border-slate-600">d_HDiC</th>
                      <th className="p-2 border border-slate-600">Person ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resultData.frame_details.map((f, i) => (
                      <tr
                        key={i}
                        className={`${
                          f.decision === "MATCH"
                            ? "bg-emerald-800/30"
                            : "bg-slate-800"
                        }`}
                      >
                        <td className="p-2 border border-slate-700 text-center">
                          {f.index}
                        </td>
                        <td className="p-2 border border-slate-700 text-center">
                          {f.decision}
                        </td>
                        <td className="p-2 border border-slate-700 text-center">
                          {f.Sfinal}
                        </td>
                        <td className="p-2 border border-slate-700 text-center">
                          {f.d_nh}
                        </td>
                        <td className="p-2 border border-slate-700 text-center">
                          {f.d_hdic}
                        </td>
                        <td className="p-2 border border-slate-700 text-center">
                          {f.person_id || "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
