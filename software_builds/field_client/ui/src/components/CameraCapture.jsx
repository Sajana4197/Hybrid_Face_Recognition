import React, { useEffect, useRef, useState } from "react";
import { postMatch, sendAlertToAdmin } from "../api";

export default function CameraCapture() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [status, setStatus] = useState("camera starting...");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setStatus("ready");
      } catch (e) {
        setStatus("camera error: " + e.message);
      }
    })();
    return () => {
      const s = videoRef.current?.srcObject;
      s && s.getTracks().forEach(t => t.stop());
    };
  }, []);

  async function captureAndMatch() {
    if (busy) return;
    setBusy(true);
    setResult(null);

    const w = videoRef.current.videoWidth;
    const h = videoRef.current.videoHeight;
    const cvs = canvasRef.current;
    cvs.width = w; cvs.height = h;
    const ctx = cvs.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, w, h);
    cvs.toBlob(async (blob) => {
      try {
        const res = await postMatch(blob);
        setResult(res);
        if (res.decision === "MATCH") {
          const ts = new Date().toISOString();
          await sendAlertToAdmin({
            blob,
            person_id: res.person_id,
            score: res.scores?.Sfinal ?? 0,
            timestamp: ts
          });
        }
      } catch (e) {
        setResult({ error: e.message });
      } finally {
        setBusy(false);
      }
    }, "image/jpeg", 0.95);
  }

  return (
    <div style={{display:"grid", gap:12}}>
      <video ref={videoRef} style={{width:"100%", maxWidth:640, border:"1px solid #ccc"}} playsInline muted/>
      <button onClick={captureAndMatch} disabled={busy || status!=="ready"}>
        {busy ? "Matching..." : "Capture & Match"}
      </button>
      <canvas ref={canvasRef} style={{display:"none"}}/>
      <pre style={{whiteSpace:"pre-wrap", background:"#111", color:"#0f0", padding:12}}>
        {result ? JSON.stringify(result, null, 2) : status}
      </pre>
    </div>
  );
}
