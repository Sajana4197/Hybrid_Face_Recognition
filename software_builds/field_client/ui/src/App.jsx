import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
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
  RefreshCw,
  ClipboardList,
  Menu,
  X,
  ChevronsLeft,
  ChevronsRight,
  UserCircle,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:5001";

export default function App() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [decision, setDecision] = useState("");
  const [resultData, setResultData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState("checking");
  const [glowColor, setGlowColor] = useState("none");
  const glowTimerRef = useRef(null);
  const [adminStatus, setAdminStatus] = useState(null);
  const pollRef = useRef(null);
  const [activeTab, setActiveTab] = useState("main");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const matchSoundRef = useRef(null);
  const noMatchSoundRef = useRef(null);

  // Ref for admin decision sound
  const verificationSoundRef = useRef(null);

  const [verificationItems, setVerificationItems] = useState([]);
  const [verificationLoading, setVerificationLoading] = useState(true);
  const [verificationError, setVerificationError] = useState(null);

  // --- ADDED: State to track *seen admin decisions* ---
  const [lastSeenDecisionTimestamp, setLastSeenDecisionTimestamp] =
    useState(null);
  // --------------------------------------------------

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
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        console.error("Camera initialization failed:", err);
      }
    }
    initCamera();
  }, []);

  useEffect(() => {
    if (
      activeTab === "main" &&
      videoRef.current &&
      streamRef.current &&
      videoRef.current.srcObject !== streamRef.current
    ) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [activeTab]);

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

  // --- MODIFIED: loadVerifications is now simple ---
  // It no longer plays sound or does complex state updates
  const loadVerifications = useCallback(async (manual = false) => {
    if (manual) setVerificationLoading(true);
    setVerificationError(null);
    try {
      const r = await fetch(`${API_BASE}/verifications`);
      if (!r.ok) throw new Error(`Failed: ${r.status}`);
      const d = await r.json();
      const sortedItems = (d.items || []).sort((a, b) =>
        b.timestamp.localeCompare(a.timestamp)
      );

      // Just set the state. The `useMemo` will handle the badge.
      setVerificationItems(sortedItems);
    } catch (err) {
      console.error("Failed to load verifications:", err);
      setVerificationError(err.message);
      setVerificationItems([]);
    } finally {
      if (manual) setVerificationLoading(false);
    }
  }, []);

  useEffect(() => {
    loadVerifications(true);
    const interval = setInterval(() => loadVerifications(false), 10000);
    return () => clearInterval(interval);
  }, [loadVerifications]);

  // --- MODIFIED: This now counts *new admin decisions* ---
  const newDecisionCount = useMemo(() => {
    return verificationItems.filter(
      (i) =>
        (i.status === "confirmed" || i.status === "rejected") && // 1. It's a decided item
        (!lastSeenDecisionTimestamp || // 2. We've never checked OR
          (i.decision_time && i.decision_time > lastSeenDecisionTimestamp)) // 3. The decision is new
    ).length;
  }, [verificationItems, lastSeenDecisionTimestamp]);
  // --------------------------------------------------------

  useEffect(() => {
    if (glowTimerRef.current) clearTimeout(glowTimerRef.current);
    if (["MATCH", "ADMIN_REJECTED"].includes(decision)) {
      setGlowColor("pulse-red");
    } else if (["NO_MATCH", "ADMIN_CONFIRMED"].includes(decision)) {
      setGlowColor("pulse-green");
    } else {
      setGlowColor("none");
      return;
    }
    glowTimerRef.current = setTimeout(() => setGlowColor("none"), 5000);
    return () => clearTimeout(glowTimerRef.current);
  }, [decision]);

  // --- MODIFIED: pollAdminDecision now plays the sound ---
  async function pollAdminDecision(pid, ts) {
    if (!pid || !ts) return;
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(
          `${API_BASE}/check_status?person_id=${pid}&timestamp=${ts}`
        );
        const data = await res.json();

        // Check if a decision has been made
        if (data.status === "confirmed" || data.status === "rejected") {
          clearInterval(pollRef.current); // Stop polling

          // --- Play sound on admin decision ---
          const audio = verificationSoundRef.current;
          if (audio) {
            audio.currentTime = 0;
            audio
              .play()
              .catch((e) =>
                console.warn("Admin decision sound play failed:", e)
              );
          }
          // ------------------------------------

          // Update the UI
          setAdminStatus(data.status);
          loadVerifications(false); // Refresh the list (this will trigger the badge)
        }
      } catch (e) {
        console.warn("Polling error:", e);
      }
    }, 5000);
  }

  async function captureAndMatch() {
    if (busy) return;

    // --- ADDED: Prime all audio sounds on user click ---
    if (matchSoundRef.current) matchSoundRef.current.load();
    if (noMatchSoundRef.current) noMatchSoundRef.current.load();
    if (verificationSoundRef.current) verificationSoundRef.current.load();
    // ---------------------------------------------------

    setGlowColor("none");
    if (glowTimerRef.current) clearTimeout(glowTimerRef.current);
    setBusy(true);
    setDecision("Capturing...");
    setResultData(null);
    setAdminStatus(null);
    const video = videoRef.current;
    if (!video) {
      setDecision("ERROR");
      setBusy(false);
      return;
    }
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
      const res = await fetch(`${API_BASE}/match_multi`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      setResultData(data);
      setDecision(data.best_decision || data.decision || "UNKNOWN");
      try {
        if (data.decision === "MATCH") {
          const audio = matchSoundRef.current;
          if (audio) {
            audio.currentTime = 0;
            audio
              .play()
              .catch((e) => console.warn("Match audio play failed:", e));
            setTimeout(() => audio.pause(), 5000);
          }
        } else if (data.decision === "NO_MATCH") {
          const audio = noMatchSoundRef.current;
          if (audio) {
            audio.currentTime = 0;
            audio
              .play()
              .catch((e) => console.warn("No-match audio play failed:", e));
            setTimeout(() => audio.pause(), 5000);
          }
        }
      } catch (err) {
        console.warn("Audio playback error:", err);
      }
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

  // --- MODIFIED: Badge now uses newDecisionCount ---
  const navItems = [
    { id: "main", label: "Face Recognition", icon: <Camera size={20} /> },
    {
      id: "verifications",
      label: "Manual Verifications",
      icon: <ClipboardList size={20} />,
      badge: newDecisionCount > 0 ? newDecisionCount : null,
    },
  ];

  // --- MODIFIED: Tab click now "sees" the latest *decision* ---
  const handleVerificationTabClick = () => {
    setActiveTab("verifications");
    setSidebarOpen(false);

    // Find the latest *decision time* among all decided items
    const latestDecisionItem = verificationItems
      .filter((i) => i.status === "confirmed" || i.status === "rejected")
      .sort((a, b) => {
        // Handle null/undefined decision_time
        if (!a.decision_time) return 1;
        if (!b.decision_time) return -1;
        return b.decision_time.localeCompare(a.decision_time);
      })[0]; // Get newest

    if (latestDecisionItem && latestDecisionItem.decision_time) {
      // Set the "last seen" to this timestamp
      setLastSeenDecisionTimestamp(latestDecisionItem.decision_time);
    }
  };

  const healthStatus = {
    checking: {
      text: "Connecting...",
      color: "bg-sky-600/80 ring-sky-500/50",
    },
    ok: {
      text: "Backend Connected",
      color: "bg-green-600/80 ring-green-500/50",
    },
    down: {
      text: "Backend Down",
      color: "bg-red-600/80 ring-red-500/50",
    },
  };

  return (
    <div className="flex h-screen bg-black text-slate-200 overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-gray-900 via-black to-gray-900"></div>
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80%] h-[80%] bg-gradient-radial from-cyan-500/10 to-transparent blur-3xl"></div>
      <aside
        className={`fixed sm:relative z-20 flex h-full flex-col bg-slate-900/60 backdrop-blur-xl border-r border-cyan-500/10 transition-all duration-300 ease-in-out
          ${isCollapsed ? "sm:w-20" : "sm:w-64"}
          ${
            sidebarOpen
              ? "translate-x-0 w-64"
              : "-translate-x-full sm:translate-x-0"
          }
        `}
      >
        <div
          className={`flex items-center border-b border-slate-700/80 ${
            isCollapsed ? "justify-center" : "px-4"
          } h-20`}
        >
          {!isCollapsed && (
            <h1 className="text-lg font-bold text-cyan-400 whitespace-nowrap tracking-widest">
              HFR CLIENT
            </h1>
          )}
          {isCollapsed && (
            <h1 className="text-lg font-bold text-cyan-400">HFR</h1>
          )}
          <button
            onClick={() => setSidebarOpen(false)}
            className="sm:hidden absolute right-4 text-slate-400 hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        <nav className="flex-grow p-2 space-y-4">
          <div className="space-y-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  if (item.id === "verifications") {
                    handleVerificationTabClick();
                  } else {
                    setActiveTab(item.id);
                    setSidebarOpen(false);
                  }
                }}
                className={`relative w-full flex items-center gap-3 rounded-lg transition-all duration-300 ${
                  isCollapsed ? "justify-center px-2 py-3" : "px-4 py-3"
                }
                  ${
                    activeTab === item.id
                      ? "bg-gradient-to-r from-cyan-500/25 via-cyan-600/25 to-cyan-400/20 border border-cyan-500/40 text-cyan-300 shadow-inner shadow-cyan-700/30 animate-glow"
                      : "text-slate-400 hover:text-cyan-300 hover:bg-slate-700/60 border border-transparent"
                  }
                `}
              >
                {activeTab === item.id && (
                  <span className="absolute left-0 top-0 bottom-0 w-1 bg-cyan-400 rounded-r-md shadow-[0_0_10px_rgba(34,211,238,0.6)]"></span>
                )}
                {item.icon}
                {!isCollapsed && (
                  <span className="font-medium text-sm whitespace-nowrap">
                    {item.label}
                  </span>
                )}
                {!isCollapsed && item.badge && (
                  <span className="ml-auto bg-rose-600 text-white text-xs font-bold px-1.5 py-0.5 rounded-full animate-pulse">
                    {item.badge}
                  </span>
                )}
                {isCollapsed && item.badge && (
                  <span className="absolute top-2 right-2 w-3 h-3 bg-rose-600 rounded-full border-2 border-slate-900 animate-pulse"></span>
                )}
              </button>
            ))}
          </div>
        </nav>

        <div className="border-t border-slate-700/80 p-2">
          <div className="flex items-center justify-between p-2 rounded-md hover:bg-slate-700/50 transition-colors">
            <div
              className={`flex items-center gap-3 ${
                isCollapsed ? "justify-center w-full" : ""
              }`}
            >
              <UserCircle size={24} className="text-slate-400 flex-shrink-0" />
              {!isCollapsed && (
                <span className="text-sm font-medium text-slate-300 whitespace-nowrap">
                  Field Officer
                </span>
              )}
            </div>
            {!isCollapsed && (
              <button
                onClick={() => setIsCollapsed(true)}
                className="hidden sm:block text-slate-400 hover:text-white"
              >
                <ChevronsLeft size={20} />
              </button>
            )}
            {isCollapsed && (
              <button
                onClick={() => setIsCollapsed(false)}
                className="hidden sm:block text-slate-400 hover:text-white absolute right-2"
              >
                <ChevronsRight size={20} />
              </button>
            )}
          </div>
        </div>
      </aside>

      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm sm:hidden z-10 animate-fadeIn"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <main className="relative flex-1 p-4 sm:p-8 overflow-y-auto">
        <header className="flex items-center justify-between mb-8 h-12">
          <button
            className="sm:hidden text-slate-300 hover:text-white"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={22} />
          </button>
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-8 h-8 text-cyan-400" />
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
              Border Control Field Client
            </h1>
          </div>
          <span
            className={`flex-shrink-0 px-4 py-1.5 rounded-full text-sm font-medium ring-1 ${healthStatus[health].color}`}
          >
            {healthStatus[health].text}
          </span>
        </header>

        <div key={activeTab}>
          {activeTab === "main" && (
            <div className="animate-fadeIn">
              <main
                className={`bg-gray-800/50 rounded-2xl shadow-2xl ring-1 ring-white/10 backdrop-blur-sm p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-hidden transition-shadow duration-500 ${
                  glowColor === "pulse-red"
                    ? "animate-pulse-red"
                    : glowColor === "pulse-green"
                    ? "animate-pulse-green"
                    : ""
                }`}
              >
                <div className="flex flex-col items-center justify-center gap-3">
                  <div className="relative w-auto max-h-[480px] aspect-[4/5] rounded-xl p-1 overflow-hidden shadow-lg bg-gradient-to-r from-cyan-400 via-blue-700 to-red-600">
                    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover transform scale-x-[-1]"
                      />
                    </div>
                  </div>
                  <p className="text-sm text-gray-400">
                    Live camera feed. Ensure face is <strong>Centered</strong>
                  </p>
                </div>
                <div className="flex flex-col items-center justify-center gap-4">
                  <button
                    onClick={captureAndMatch}
                    disabled={busy}
                    className={`w-full max-w-sm flex items-center justify-center gap-3 px-6 py-4 rounded-lg font-semibold shadow-lg transition-all duration-300 text-lg text-white transform hover:scale-105 ${
                      busy
                        ? "bg-gray-600 cursor-not-allowed"
                        : "bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700"
                    }`}
                  >
                    {busy ? <Loader className="animate-spin" /> : <Camera />}
                    {busy ? "Processing..." : "Capture & Match"}
                  </button>
                  <DecisionDisplay
                    decision={decision}
                    adminStatus={adminStatus}
                  />
                  {resultData && <ResultsSummary resultData={resultData} />}
                  <audio
                    ref={matchSoundRef}
                    id="matchSound"
                    src="/match-sound.wav"
                    preload="auto"
                  />
                  <audio
                    ref={noMatchSoundRef}
                    id="noMatchSound"
                    src="/no-match-sound.mp3"
                    preload="auto"
                  />
                  <audio
                    ref={verificationSoundRef}
                    id="verificationSound"
                    src="../public/new-verification.mp3"
                    preload="auto"
                  />
                </div>
              </main>
            </div>
          )}

          {activeTab === "verifications" && (
            <div className="animate-fadeIn">
              <VerificationPanel
                items={verificationItems}
                isLoading={verificationLoading}
                error={verificationError}
                loadVerifications={() => loadVerifications(true)}
              />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function VerificationPanel({ items, isLoading, error, loadVerifications }) {
  const formatTimestamp = (ts) => {
    try {
      const isoString = ts
        .replace("_", "T")
        .replace(/-(\d{2})-(\d{2})$/, ":$1:$2");
      const date = new Date(isoString);
      if (isNaN(date.getTime())) return ts;
      return date.toLocaleString();
    } catch {
      return ts;
    }
  };

  return (
    <div className="bg-gray-800/50 p-4 sm:p-6 rounded-2xl ring-1 ring-white/10 backdrop-blur-sm">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-white">
          Manual Verification Queue
        </h2>
        <button
          onClick={loadVerifications}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors disabled:bg-gray-600"
        >
          {isLoading ? (
            <Loader className="w-5 h-5 animate-spin" />
          ) : (
            <RefreshCw className="w-5 h-5" />
          )}
          {isLoading ? "Loading..." : "Refresh"}
        </button>
      </div>
      <div>
        {error ? (
          <div className="flex flex-col justify-center items-center text-center h-64 bg-gray-900/40 rounded-lg p-4 ring-2 ring-red-500/50">
            <AlertTriangle className="w-12 h-12 text-red-400 mb-4" />
            <p className="text-red-300 text-lg font-semibold">
              Error Loading Verifications
            </p>
            <p className="text-red-400 mt-1">{error}</p>
          </div>
        ) : isLoading && items.length === 0 ? (
          <div className="flex justify-center items-center h-64">
            <Loader className="w-12 h-12 animate-spin text-blue-400" />
          </div>
        ) : items.length === 0 ? (
          <div className="flex justify-center items-center h-64 bg-gray-900/40 rounded-lg">
            <p className="text-gray-400 text-lg">
              No items in verification queue.
            </p>
          </div>
        ) : (
          <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {items.map((i) => (
              <div
                key={i.person_id + i.timestamp}
                className={`bg-gray-900/70 rounded-lg p-3 border shadow-lg transition-all hover:scale-105 ${
                  i.status === "confirmed"
                    ? "ring-2 ring-red-400/60 border-red-500/50"
                    : i.status === "rejected"
                    ? "ring-2 ring-green-400/60 border-green-500/50"
                    : "ring-2 ring-yellow-300/60 border-yellow-500/50"
                }`}
              >
                <img
                  src={`${API_BASE}/verifications/image/${i.image}`}
                  alt="Verification"
                  className="w-full h-auto aspect-square object-cover rounded-md mb-3 border border-gray-600"
                  onError={(e) =>
                    (e.target.src =
                      "https://placehold.co/400x400/334155/94a3b8?text=Error")
                  }
                />
                <p className="font-mono text-sm break-all mb-1">
                  <b>ID:</b> {i.person_id}
                </p>
                <p className="text-sm mb-1">
                  <b>Score:</b>{" "}
                  <span className="font-medium text-cyan-300">
                    {i.score.toFixed(4)}
                  </span>
                </p>
                <p className="text-sm mb-1">
                  <b>Status:</b>{" "}
                  <span
                    className={`font-semibold ${
                      i.status === "pending"
                        ? "text-yellow-400"
                        : i.status === "confirmed"
                        ? "text-red-400"
                        : "text-green-400"
                    }`}
                  >
                    {i.status.charAt(0).toUpperCase() + i.status.slice(1)}
                  </span>
                </p>
                {i.decision_time && (
                  <p className="text-xs text-gray-400 mt-1">
                    <b>Decided:</b> {formatTimestamp(i.decision_time)}
                  </p>
                )}
                <p className="text-xs text-gray-400 mt-2">
                  {formatTimestamp(i.timestamp)}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const DecisionDisplay = ({ decision, adminStatus }) => {
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
    "Capturing...": {
      icon: <Loader size={28} className="animate-spin" />,
      style:
        "bg-sky-500/20 border-sky-500/50 text-sky-200 items-center justify-center",
      title: "Capturing...",
    },
    default: {
      icon: <Info size={28} />,
      style:
        "bg-gray-700/60 border-gray-600/80 text-gray-300 items-center justify-center",
      title: "Ready",
      text: "Click to begin",
    },
  };
  const current = config[decision] || config.default;
  return (
    <div
      className={`w-full max-w-sm flex items-center justify-center gap-4 p-4 rounded-lg border ${current.style}`}
    >
      <div className="flex-shrink-0">{current.icon}</div>
      <div>
        <h3 className="font-bold text-lg leading-tight">{current.title}</h3>
        {current.text && <p className="text-sm opacity-80">{current.text}</p>}
      </div>
    </div>
  );
};

const ResultsSummary = ({ resultData }) => (
  <div className="w-full max-w-sm grid grid-cols-2 gap-3 text-sm text-gray-300 bg-gray-900/50 p-4 rounded-lg ring-1 ring-white/10">
    <div>
      Frames: <b className="text-white">{resultData.frames}</b>
    </div>
    <div>
      Matches: <b className="text-white">{resultData.match_frames}</b>
    </div>
    <div>
      ID:{" "}
      <b className="ml-1 font-mono text-cyan-400">
        {resultData.decision === "MATCH" ? resultData.best_person_id : "N/A"}
      </b>
    </div>
    {resultData.best_score && (
      <div>
        Score:{" "}
        <b className="text-cyan-400">{resultData.best_score.toFixed(3)}</b>
      </div>
    )}
  </div>
);

/* -------------------------- Global Styles & Animations -------------------------- */
const style = document.createElement("style");
style.innerHTML = `
@keyframes pulseGlow {
  0%, 100% {
    box-shadow: 0 0 8px rgba(34, 211, 238, 0.4), inset 0 0 6px rgba(34, 211, 238, 0.2);
  }
  50% {
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.7), inset 0 0 10px rgba(34, 211, 238, 0.4);
  }
}
.animate-glow {
  animation: pulseGlow 2.5s ease-in-out infinite;
}
@keyframes fadeIn { 
  from { opacity: 0; transform: translateY(10px); } 
  to { opacity: 1; transform: translateY(0); } 
}
.animate-fadeIn { 
  animation: fadeIn 0.5s ease-in-out; 
}
@keyframes pulse-red {
  0%, 100% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.6); }
  50% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
}
.animate-pulse-red {
  animation: pulse-red 1s ease-in-out 5;
}
@keyframes pulse-green {
  0%, 100% { box-shadow: 0 0 30px rgba(34, 197, 94, 0.6); }
  50% { box-shadow: 0 0 15px rgba(34, 197, 94, 0.3); }
}
.animate-pulse-green {
  animation: pulse-green 1s ease-in-out 5;
}
`;
document.head.appendChild(style);
