import React, {
  useEffect,
  useState,
  useCallback,
  useMemo,
  useRef,
} from "react";
import {
  Menu,
  X,
  UserPlus,
  UserMinus,
  Search,
  Loader,
  RefreshCw,
  ChevronsLeft,
  ChevronsRight,
  UserCircle,
  Upload,
  CheckCircle2,
  XCircle,
  Clock,
  AlertCircle,
} from "lucide-react";

// The backend API URL is retrieved from environment variables, with a fallback for local development.
const API = import.meta.env.VITE_ADMIN_API || "http://127.0.0.1:5002";

/* --------------------------- Reusable UI Components --------------------------- */

const Card = ({ title, children, actions = null }) => (
  <div className="bg-gradient-to-br from-slate-800/40 via-slate-800/30 to-slate-900/40 backdrop-blur-2xl rounded-2xl p-6 sm:p-8 ring-1 ring-white/10 border border-white/5 shadow-2xl shadow-black/50 animate-fadeIn hover:ring-white/20 transition-all duration-500">
    <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/10">
      <h2 className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent tracking-tight">
        {title}
      </h2>
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </div>
    <div>{children}</div>
  </div>
);

const Button = ({
  children,
  onClick,
  variant = "primary",
  type = "button",
  disabled = false,
  className = "",
}) => {
  const baseStyles =
    "px-5 py-2.5 rounded-xl font-semibold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-transparent disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transform hover:scale-[1.02] active:scale-[0.98]";
  const variants = {
    primary:
      "bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white focus:ring-emerald-400 shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50",
    secondary:
      "bg-slate-700/50 hover:bg-slate-600/50 text-slate-200 focus:ring-slate-400 backdrop-blur-xl border border-white/10 hover:border-white/20",
    danger:
      "bg-gradient-to-r from-rose-500 to-pink-500 hover:from-rose-600 hover:to-pink-600 text-white focus:ring-rose-400 shadow-lg shadow-rose-500/30 hover:shadow-rose-500/50",
  };
  return (
    <button
      type={type}
      onClick={onClick}
      className={`${baseStyles} ${variants[variant]} ${className}`}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

const Input = ({ label, id, ...props }) => (
  <div className="group">
    <label
      htmlFor={id}
      className="block text-sm font-semibold text-slate-300 mb-2 tracking-wide group-focus-within:text-emerald-400 transition-colors"
    >
      {label}
    </label>
    <input
      id={id}
      {...props}
      className="w-full bg-slate-800/40 backdrop-blur-xl rounded-xl px-4 py-3 text-slate-100 ring-1 ring-white/10 border border-white/5 focus:ring-2 focus:ring-emerald-400/50 focus:border-emerald-400/50 focus:outline-none transition-all duration-300 placeholder:text-slate-500"
    />
  </div>
);

const Spinner = () => (
  <div className="flex justify-center p-8">
    <div className="relative">
      <Loader className="animate-spin text-emerald-400" size={32} />
      <div className="absolute inset-0 animate-ping">
        <Loader className="text-emerald-400/30" size={32} />
      </div>
    </div>
  </div>
);

/* -------------------------------- Main App -------------------------------- */
export default function App() {
  const [activeTab, setActiveTab] = useState("enroll");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [health, setHealth] = useState("checking");
  const [manualAlerts, setManualAlerts] = useState([]);
  const [alertsLoading, setAlertsLoading] = useState(true);
  const [lastSeenPendingTimestamp, setLastSeenPendingTimestamp] =
    useState(null);
  const alertSoundRef = useRef(null);
  const lastAlertedTimestampRef = useRef(null);
  const audioPrimedRef = useRef(false);

  // --- Health Check ---
  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => setHealth(r.ok ? "ok" : "down"))
      .catch(() => setHealth("down"));
  }, []);

  // --- Audio priming ---
  useEffect(() => {
    const primeAudio = () => {
      if (alertSoundRef.current && !audioPrimedRef.current) {
        const audio = alertSoundRef.current;
        console.log("Attempting to prime audio via user interaction...");
        try {
          audio.volume = 0;
          const playPromise = audio.play();

          if (playPromise !== undefined) {
            playPromise
              .then(() => {
                audio.pause();
                audio.currentTime = 0;
                audio.volume = 1;
                audioPrimedRef.current = true;
                console.log(
                  "✅ Audio successfully primed by user interaction."
                );

                document.removeEventListener("click", primeAudio, true);
                document.removeEventListener("touchstart", primeAudio, true);
              })
              .catch((error) => {
                console.warn("Audio priming play() failed:", error);
              });
          } else {
            console.warn(
              "Audio element might not be ready for priming yet (playPromise undefined)."
            );
          }
        } catch (err) {
          console.error("Error during audio priming attempt:", err);
        }
      } else if (audioPrimedRef.current) {
        document.removeEventListener("click", primeAudio, true);
        document.removeEventListener("touchstart", primeAudio, true);
      }
    };

    document.addEventListener("click", primeAudio, true);
    document.addEventListener("touchstart", primeAudio, true);

    return () => {
      document.removeEventListener("click", primeAudio, true);
      document.removeEventListener("touchstart", primeAudio, true);
    };
  }, []);

  // --- Polling function ---
  const loadAlerts = useCallback(async () => {
    try {
      const r = await fetch(`${API}/manual_check/list`);
      const d = await r.json();
      const sortedAlerts = (d.alerts || []).sort((a, b) =>
        b.timestamp.localeCompare(a.timestamp)
      );
      setManualAlerts(sortedAlerts);
    } catch (err) {
      console.error("Failed to load alerts:", err);
    }
  }, []);

  // --- Start polling on mount ---
  useEffect(() => {
    async function initialLoad() {
      await loadAlerts();
      setAlertsLoading(false);
    }
    initialLoad();
    const interval = setInterval(loadAlerts, 5000);
    return () => clearInterval(interval);
  }, [loadAlerts]);

  // --- Play sound when alerts change ---
  useEffect(() => {
    const newestPending = manualAlerts.find((a) => a.status === "pending");

    if (newestPending) {
      const newTimestamp = newestPending.timestamp;
      const lastTimestamp = lastAlertedTimestampRef.current || "";

      if (newTimestamp > lastTimestamp) {
        const audio = alertSoundRef.current;

        if (audio && audioPrimedRef.current) {
          audio.currentTime = 0;
          audio.play().catch((e) => {
            console.warn(`Alert sound play failed for ${newTimestamp}:`, e);
          });
        } else if (!audioPrimedRef.current) {
          console.log(
            `Audio not yet primed for alert ${newTimestamp}. Click/touch page.`
          );
        }

        lastAlertedTimestampRef.current = newTimestamp;
      }
    }
  }, [manualAlerts]);

  // --- Handle decision ---
  async function handleManualDecision(a, decision) {
    const form = new FormData();
    form.append("person_id", a.person_id);
    form.append("timestamp", a.timestamp);
    form.append("decision", decision);
    await fetch(`${API}/manual_check/decision`, { method: "POST", body: form });
    await loadAlerts();
  }

  // --- Calculate new pending count ---
  const newPendingCount = useMemo(
    () =>
      manualAlerts.filter(
        (a) =>
          a.status === "pending" &&
          (!lastSeenPendingTimestamp || a.timestamp > lastSeenPendingTimestamp)
      ).length,
    [manualAlerts, lastSeenPendingTimestamp]
  );

  const navGroups = [
    {
      title: "Main",
      items: [
        { id: "enroll", label: "Enroll", icon: <UserPlus size={20} /> },
        { id: "watchlist", label: "Watchlist", icon: <UserMinus size={20} /> },
        {
          id: "manual",
          label: "Manual Check",
          icon: <Search size={20} />,
          badge: newPendingCount > 0 ? newPendingCount : null,
        },
      ],
    },
  ];

  const healthStatus = {
    checking: {
      text: "Connecting...",
      color: "bg-gradient-to-r from-sky-500 to-blue-500",
      icon: <Clock size={14} />,
    },
    ok: {
      text: "System Online",
      color: "bg-gradient-to-r from-emerald-500 to-teal-500",
      icon: <CheckCircle2 size={14} />,
    },
    down: {
      text: "System Offline",
      color: "bg-gradient-to-r from-rose-500 to-pink-500",
      icon: <XCircle size={14} />,
    },
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden relative">
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-950 to-black"></div>
      <div className="absolute top-0 left-0 w-full h-full">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      <audio ref={alertSoundRef} src="/new-verification.mp3" preload="auto" />

      {/* Sidebar */}
      <aside
        className={`fixed sm:relative z-20 flex h-full flex-col bg-slate-900/80 backdrop-blur-2xl border-r border-white/10 transition-all duration-300 ease-in-out
          ${isCollapsed ? "sm:w-20" : "sm:w-72"}
          ${
            sidebarOpen
              ? "translate-x-0 w-72"
              : "-translate-x-full sm:translate-x-0"
          }
        `}
      >
        {/* Logo */}
        <div
          className={`flex items-center border-b border-white/10 bg-gradient-to-r from-emerald-500/10 to-teal-500/10 ${
            isCollapsed ? "justify-center" : "px-6"
          } h-20`}
        >
          {!isCollapsed && (
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                <span className="text-white font-bold text-xl">FR</span>
              </div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent tracking-tight">
                HFR PANEL
              </h1>
            </div>
          )}
          {isCollapsed && (
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/30">
              <span className="text-white font-bold text-xl">FR</span>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(false)}
            className="sm:hidden absolute right-4 text-slate-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-grow p-3 space-y-6 overflow-y-auto">
          {navGroups.map((group) => (
            <div key={group.title}>
              {!isCollapsed && (
                <h3 className="px-4 py-2 text-xs uppercase text-slate-500 font-bold tracking-widest">
                  {group.title}
                </h3>
              )}
              <div className="space-y-2">
                {group.items.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      if (item.id === "manual") {
                        const latestPending = manualAlerts.find(
                          (a) => a.status === "pending"
                        );
                        if (latestPending) {
                          setLastSeenPendingTimestamp(latestPending.timestamp);
                        }
                      }
                      setActiveTab(item.id);
                      setSidebarOpen(false);
                    }}
                    className={`relative w-full flex items-center gap-3 rounded-xl transition-all duration-300 ${
                      isCollapsed ? "justify-center px-3 py-3" : "px-4 py-3"
                    }
                      ${
                        activeTab === item.id
                          ? "bg-gradient-to-r from-emerald-500/20 to-teal-500/20 text-emerald-300 shadow-lg shadow-emerald-500/20 ring-1 ring-emerald-500/30"
                          : "text-slate-400 hover:text-emerald-300 hover:bg-white/5"
                      }
                    `}
                  >
                    {activeTab === item.id && (
                      <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-gradient-to-b from-emerald-400 to-teal-400 rounded-r-full shadow-lg shadow-emerald-400/50"></span>
                    )}
                    <span className={activeTab === item.id ? "scale-110" : ""}>
                      {item.icon}
                    </span>
                    {!isCollapsed && (
                      <span className="font-semibold text-sm whitespace-nowrap">
                        {item.label}
                      </span>
                    )}
                    {!isCollapsed && item.badge && (
                      <span className="ml-auto bg-gradient-to-r from-rose-500 to-pink-500 text-white text-xs font-bold px-2 py-1 rounded-full shadow-lg shadow-rose-500/30 animate-pulse">
                        {item.badge}
                      </span>
                    )}
                    {isCollapsed && item.badge && (
                      <span className="absolute top-2 right-2 w-3 h-3 bg-gradient-to-r from-rose-500 to-pink-500 rounded-full border-2 border-slate-900 shadow-lg shadow-rose-500/50 animate-pulse"></span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </nav>

        {/* User section */}
        <div className="border-t border-white/10 p-3 bg-gradient-to-r from-slate-900/50 to-slate-800/50">
          <div className="flex items-center justify-between p-3 rounded-xl hover:bg-white/5 transition-all duration-300">
            <div
              className={`flex items-center gap-3 ${
                isCollapsed ? "justify-center w-full" : ""
              }`}
            >
              <div className="w-9 h-9 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                <UserCircle size={20} className="text-white" />
              </div>
              {!isCollapsed && (
                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-slate-200">
                    Admin User
                  </span>
                  <span className="text-xs text-slate-500">Administrator</span>
                </div>
              )}
            </div>
            {!isCollapsed && (
              <button
                onClick={() => setIsCollapsed(true)}
                className="hidden sm:block text-slate-400 hover:text-white transition-colors p-1 hover:bg-white/10 rounded-lg"
              >
                <ChevronsLeft size={18} />
              </button>
            )}
            {isCollapsed && (
              <button
                onClick={() => setIsCollapsed(false)}
                className="hidden sm:block text-slate-400 hover:text-white transition-colors p-1 hover:bg-white/10 rounded-lg absolute right-2"
              >
                <ChevronsRight size={18} />
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm sm:hidden z-10 animate-fadeIn"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <main className="relative flex-1 p-4 sm:p-8 overflow-y-auto">
        <header className="flex items-center justify-between mb-8 h-12">
          <div className="flex items-center gap-4">
            <button
              className="sm:hidden text-slate-300 hover:text-white p-2 hover:bg-white/10 rounded-lg transition-all"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu size={22} />
            </button>
            <h1 className="hidden sm:block text-2xl font-bold bg-gradient-to-r from-slate-200 to-slate-400 bg-clip-text text-transparent tracking-tight">
              ADMINISTRATION CONSOLE
            </h1>
          </div>
          <div
            className={`px-4 py-2 rounded-xl text-sm font-semibold ${healthStatus[health].color} text-white shadow-lg flex items-center gap-2`}
          >
            {healthStatus[health].icon}
            <span>{healthStatus[health].text}</span>
          </div>
        </header>
        <div key={activeTab}>
          {activeTab === "enroll" && <EnrollView />}
          {activeTab === "watchlist" && <WatchlistView />}
          {activeTab === "manual" && (
            <ManualCheckView
              alerts={manualAlerts}
              loading={alertsLoading}
              onReload={async () => {
                setAlertsLoading(true);
                await loadAlerts();
                setAlertsLoading(false);
              }}
              onDecide={handleManualDecision}
            />
          )}
        </div>
      </main>
    </div>
  );
}

/* ----------------------------- Feature Views ------------------------------ */

function ConfirmationModal({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = "Confirm",
  cancelText = "Cancel",
  type = "danger",
}) {
  if (!isOpen) return null;

  const typeStyles = {
    danger: {
      button:
        "bg-rose-600/90 hover:bg-rose-700 text-white focus:ring-rose-500 shadow-lg shadow-rose-600/20 hover:shadow-rose-600/40",
      icon: "⚠️",
      iconBg: "bg-rose-500/20",
      iconRing: "ring-rose-500/50",
    },
    warning: {
      button:
        "bg-amber-600/90 hover:bg-amber-700 text-white focus:ring-amber-500 shadow-lg shadow-amber-600/20 hover:shadow-amber-600/40",
      icon: "⚡",
      iconBg: "bg-amber-500/20",
      iconRing: "ring-amber-500/50",
    },
    info: {
      button:
        "bg-cyan-600/90 hover:bg-cyan-700 text-white focus:ring-cyan-500 shadow-lg shadow-cyan-600/20 hover:shadow-cyan-600/40",
      icon: "ℹ️",
      iconBg: "bg-cyan-500/20",
      iconRing: "ring-cyan-500/50",
    },
  };

  const style = typeStyles[type];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center animate-fadeIn">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      ></div>

      {/* Modal */}
      <div className="relative bg-slate-800/95 backdrop-blur-xl rounded-2xl p-6 max-w-md w-full mx-4 ring-1 ring-slate-700/50 border-t border-slate-700/80 shadow-2xl shadow-black/40 animate-scaleIn">
        {/* Icon */}
        <div className="flex justify-center mb-4">
          <div
            className={`p-3 ${style.iconBg} rounded-full ring-2 ${style.iconRing}`}
          >
            <span className="text-3xl">{style.icon}</span>
          </div>
        </div>

        {/* Title */}
        <h3 className="text-xl font-bold text-slate-100 text-center mb-2">
          {title}
        </h3>

        {/* Message */}
        <p className="text-slate-400 text-center mb-6">{message}</p>

        {/* Buttons */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2.5 rounded-lg font-semibold transition-all duration-300 bg-slate-700/80 hover:bg-slate-700 text-slate-200 focus:ring-2 focus:ring-slate-500 focus:outline-none"
          >
            {cancelText}
          </button>
          <button
            onClick={() => {
              onConfirm();
              onClose();
            }}
            className={`flex-1 px-4 py-2.5 rounded-lg font-semibold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black ${style.button}`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}

function EnrollView() {
  const [pid, setPid] = useState("");
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]);
  const [msg, setMsg] = useState({ text: "", type: "info" });
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [confirmClear, setConfirmClear] = useState({ isOpen: false });

  async function submit(e) {
    e.preventDefault();
    if (!pid || files.length === 0) {
      setMsg({
        text: "Person ID and at least one folder/image are required.",
        type: "error",
      });
      return;
    }
    setIsLoading(true);
    setMsg({ text: "", type: "info" });
    const form = new FormData();
    form.append("person_id", pid);
    form.append("name", name);
    [...files].forEach((f) => form.append("files", f));
    try {
      const r = await fetch(`${API}/enroll`, { method: "POST", body: form });
      const data = await r.json();
      if (r.ok) {
        setMsg({
          text: `Success: Enrolled ${data.person_id} with ${data.added_images} images.`,
          type: "success",
        });
        setPid("");
        setName("");
        setFiles([]);
        const fileInput = document.getElementById("file-upload");
        if (fileInput) {
          fileInput.value = "";
        }
      } else {
        setMsg({
          text: `Error: ${data.detail || "Enrollment failed"}`,
          type: "error",
        });
      }
    } catch (e) {
      setMsg({ text: `Network Error: ${e.message}`, type: "error" });
    } finally {
      setIsLoading(false);
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) {
      setFiles(e.dataTransfer.files);
    }
  };

  function openClearConfirm() {
    if (pid || name || files.length > 0) {
      setConfirmClear({ isOpen: true });
    }
  }

  function closeClearConfirm() {
    setConfirmClear({ isOpen: false });
  }

  function confirmClearForm() {
    setPid("");
    setName("");
    setFiles([]);
    setMsg({ text: "", type: "info" });
    const fileInput = document.getElementById("file-upload");
    if (fileInput) {
      fileInput.value = "";
    }
  }

  const messageColors = {
    info: "text-slate-400",
    success: "text-emerald-400",
    error: "text-rose-400",
  };

  return (
    <>
      <div className="space-y-4">
        {/* Compact Hero Section */}
        <div className="relative overflow-hidden bg-gradient-to-br from-emerald-900/40 via-slate-900/40 to-cyan-900/40 backdrop-blur-xl rounded-xl p-5 border border-emerald-500/20 shadow-xl">
          <div className="absolute top-0 right-0 w-48 h-48 bg-emerald-500/10 rounded-full blur-3xl animate-pulse"></div>

          <div className="relative z-10 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-emerald-500/20 rounded-lg ring-1 ring-emerald-500/50 shadow-lg shadow-emerald-500/30">
                <UserPlus size={24} className="text-emerald-400" />
              </div>
              <div>
                <h2 className="text-2xl font-bold bg-gradient-to-r from-emerald-400 via-cyan-300 to-emerald-500 bg-clip-text text-transparent">
                  Enroll New Subject
                </h2>
                <p className="text-slate-400 text-xs mt-0.5">
                  Add new persons to the facial recognition system
                </p>
              </div>
            </div>

            {/* Clear Form Button */}
            {(pid || name || files.length > 0) && (
              <Button
                onClick={openClearConfirm}
                variant="secondary"
                className="px-3 py-2 text-xs flex-shrink-0"
              >
                <X size={14} />
                <span className="ml-1.5">Clear Form</span>
              </Button>
            )}
          </div>
        </div>

        {/* Compact Form Card */}
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-5 ring-1 ring-slate-700/50 border-t border-slate-700/80 shadow-xl animate-fadeIn">
          <form onSubmit={submit} className="space-y-4">
            {/* Compact Input Fields */}
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="group">
                <label
                  htmlFor="pid"
                  className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-1.5 tracking-wider"
                >
                  <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></span>
                  Person ID
                </label>
                <div className="relative">
                  <input
                    id="pid"
                    placeholder="e.g., n000123"
                    value={pid}
                    onChange={(e) => setPid(e.target.value)}
                    required
                    className="w-full bg-slate-900/60 rounded-lg p-3 pl-10 text-slate-100 ring-1 ring-slate-700/50 focus:ring-2 focus:ring-emerald-500 focus:outline-none transition-all duration-300 group-hover:ring-emerald-500/30 font-mono"
                  />
                  <UserCircle
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-emerald-400 transition-colors"
                    size={18}
                  />
                </div>
              </div>

              <div className="group">
                <label
                  htmlFor="name"
                  className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-1.5 tracking-wider"
                >
                  <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse delay-300"></span>
                  Name (Optional)
                </label>
                <div className="relative">
                  <input
                    id="name"
                    placeholder="e.g., John Doe"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full bg-slate-900/60 rounded-lg p-3 pl-10 text-slate-100 ring-1 ring-slate-700/50 focus:ring-2 focus:ring-cyan-500 focus:outline-none transition-all duration-300 group-hover:ring-cyan-500/30"
                  />
                  <UserCircle
                    className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-cyan-400 transition-colors"
                    size={18}
                  />
                </div>
              </div>
            </div>

            {/* Compact Drag & Drop Upload Area */}
            <div>
              <label
                htmlFor="file-upload"
                className="flex items-center gap-2 text-xs font-medium text-slate-300 mb-2 tracking-wider"
              >
                <span className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-pulse delay-500"></span>
                Upload Images from Folder
              </label>

              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-6 transition-all duration-300 ${
                  isDragging
                    ? "border-emerald-500 bg-emerald-500/10 scale-105"
                    : files.length > 0
                    ? "border-emerald-500/50 bg-emerald-500/5"
                    : "border-slate-600 hover:border-slate-500 bg-slate-900/40"
                }`}
              >
                <input
                  id="file-upload"
                  type="file"
                  className="hidden"
                  webkitdirectory=""
                  multiple
                  onChange={(e) => setFiles(e.target.files)}
                />

                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex flex-col items-center gap-3"
                >
                  <div
                    className={`p-3 rounded-full transition-all duration-300 ${
                      files.length > 0
                        ? "bg-emerald-500/20 ring-2 ring-emerald-500/30"
                        : "bg-slate-700/50"
                    }`}
                  >
                    <Upload
                      size={24}
                      className={`transition-colors ${
                        files.length > 0 ? "text-emerald-400" : "text-slate-400"
                      }`}
                    />
                  </div>

                  <div className="text-center">
                    {files.length > 0 ? (
                      <div className="space-y-1">
                        <p className="text-emerald-400 font-semibold">
                          ✓ {files.length}{" "}
                          {files.length === 1 ? "file" : "files"} selected
                        </p>
                        <p className="text-slate-400 text-xs">
                          Click to change or drag new files
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-1">
                        <p className="text-slate-300 font-semibold">
                          Drag & Drop folder here
                        </p>
                        <p className="text-slate-400 text-xs">
                          or click to browse
                        </p>
                      </div>
                    )}
                  </div>
                </label>

                {isDragging && (
                  <div className="absolute inset-0 bg-emerald-500/20 rounded-xl flex items-center justify-center">
                    <p className="text-emerald-300 font-bold">
                      Drop files here!
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Compact Submit Button */}
            <div className="space-y-3">
              <Button
                type="submit"
                variant="primary"
                disabled={isLoading}
                className="w-full sm:w-auto py-3 px-6 relative overflow-hidden group"
              >
                <span className="absolute inset-0 bg-gradient-to-r from-emerald-600 via-cyan-500 to-emerald-600 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></span>
                <span className="relative flex items-center gap-2">
                  {isLoading && <Loader size={18} className="animate-spin" />}
                  {isLoading ? "Enrolling..." : "Upload & Enroll"}
                </span>
              </Button>

              {/* Compact Message Display */}
              {msg.text && (
                <div
                  className={`flex items-center gap-2 p-3 rounded-lg text-sm ${
                    msg.type === "success"
                      ? "bg-emerald-500/10 border border-emerald-500/30"
                      : msg.type === "error"
                      ? "bg-rose-500/10 border border-rose-500/30"
                      : "bg-slate-700/30 border border-slate-600/30"
                  } animate-fadeIn`}
                >
                  <span
                    className={msg.type === "success" ? "animate-bounce" : ""}
                  >
                    {msg.type === "success"
                      ? "✓"
                      : msg.type === "error"
                      ? "⚠"
                      : "ℹ"}
                  </span>
                  <p className={`${messageColors[msg.type]} font-medium`}>
                    {msg.text}
                  </p>
                </div>
              )}
            </div>
          </form>
        </div>
      </div>

      {/* Clear Form Confirmation Modal */}
      <ConfirmationModal
        isOpen={confirmClear.isOpen}
        onClose={closeClearConfirm}
        onConfirm={confirmClearForm}
        title="Clear Form"
        message="Are you sure you want to clear all form data? Any unsaved information will be lost."
        confirmText="Clear"
        cancelText="Cancel"
        type="warning"
      />
    </>
  );
}

function WatchlistView() {
  const [rows, setRows] = useState([]);
  const [msg, setMsg] = useState({ text: "", type: "info" });
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [deletingId, setDeletingId] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState({
    isOpen: false,
    personId: null,
    personName: null,
  });

  async function refresh() {
    setIsLoading(true);
    setMsg({ text: "", type: "info" });
    try {
      const r = await fetch(`${API}/list`);
      const data = await r.json();
      setRows(data.persons || []);
    } catch (e) {
      setMsg({ text: `Error: ${e.message}`, type: "error" });
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  function openDeleteConfirm(person) {
    setDeleteConfirm({
      isOpen: true,
      personId: person.person_id,
      personName: person.name,
    });
  }

  function closeDeleteConfirm() {
    setDeleteConfirm({ isOpen: false, personId: null, personName: null });
  }

  async function confirmDelete() {
    const pid = deleteConfirm.personId;
    setDeletingId(pid);

    const r = await fetch(`${API}/delete/${pid}`, { method: "DELETE" });
    if (r.ok) {
      setMsg({ text: `Success: Person ${pid} deleted.`, type: "success" });
      refresh();
    } else {
      const d = await r.json();
      setMsg({ text: `Error: ${d.detail || "Delete failed"}`, type: "error" });
    }
    setDeletingId(null);
  }

  const filteredRows = rows.filter((r) => {
    const query = searchQuery.toLowerCase();
    return (
      r.person_id.toLowerCase().includes(query) ||
      (r.name && r.name.toLowerCase().includes(query))
    );
  });

  return (
    <>
      <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl ring-1 ring-slate-700/50 border-t border-slate-700/80 shadow-xl overflow-hidden flex flex-col h-full max-h-[calc(100vh-9rem)]">
        {/* Header with Title, Search Bar, and Reload */}
        <div className="relative overflow-hidden bg-gradient-to-br from-rose-900/30 via-slate-900/30 to-purple-900/30 p-5 border-b border-slate-700/50 flex-shrink-0">
          <div className="absolute top-0 right-0 w-32 h-32 bg-rose-500/10 rounded-full blur-3xl animate-pulse"></div>

          <div className="relative z-10">
            <div className="flex justify-between items-center gap-3">
              <div className="flex items-center gap-2 flex-shrink-0">
                <div className="p-1.5 bg-rose-500/20 rounded-lg ring-1 ring-rose-500/50">
                  <UserMinus size={24} className="text-rose-400" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-rose-400 via-purple-300 to-rose-500 bg-clip-text text-transparent whitespace-nowrap">
                    System Watchlist
                  </h2>
                  <p className="text-slate-500 text-[10px] whitespace-nowrap">
                    {rows.length} {rows.length === 1 ? "person" : "persons"}{" "}
                    enrolled
                  </p>
                </div>
              </div>

              <div className="relative group flex-1 max-w-md">
                <Search
                  className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 group-focus-within:text-rose-400 transition-colors"
                  size={16}
                />
                <input
                  type="text"
                  placeholder="Search by Person ID or Name..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-slate-900/60 rounded-lg pl-9 pr-9 py-2 text-slate-100 text-sm ring-1 ring-slate-700/50 focus:ring-2 focus:ring-rose-500 focus:outline-none transition-all duration-300 placeholder:text-slate-500"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery("")}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-rose-400 transition-colors"
                  >
                    <X size={16} />
                  </button>
                )}
              </div>

              <Button
                onClick={refresh}
                variant="secondary"
                disabled={isLoading}
                className="px-3 py-2 text-xs flex-shrink-0"
              >
                <RefreshCw
                  size={14}
                  className={isLoading ? "animate-spin" : ""}
                />
                <span className="ml-1.5">Reload</span>
              </Button>
            </div>

            {searchQuery && (
              <div className="mt-2 flex items-center gap-1.5 text-[10px]">
                <div className="w-1 h-1 bg-rose-400 rounded-full animate-pulse"></div>
                <p className="text-slate-400">
                  Found{" "}
                  <span className="text-rose-400 font-semibold">
                    {filteredRows.length}
                  </span>{" "}
                  of {rows.length}
                </p>
              </div>
            )}
          </div>
        </div>

        {msg.text && (
          <div
            className={`mx-4 mt-3 mb-2 flex items-center gap-2 p-2 rounded-lg text-xs flex-shrink-0 ${
              msg.type === "success"
                ? "bg-emerald-500/10 border border-emerald-500/30"
                : msg.type === "error"
                ? "bg-rose-500/10 border border-rose-500/30"
                : "bg-slate-700/30 border border-slate-600/30"
            }`}
          >
            <span className={msg.type === "success" ? "animate-bounce" : ""}>
              {msg.type === "success" ? "✓" : msg.type === "error" ? "⚠" : "ℹ"}
            </span>
            <p
              className={`${
                msg.type === "success"
                  ? "text-emerald-400"
                  : msg.type === "error"
                  ? "text-rose-400"
                  : "text-slate-400"
              } font-medium`}
            >
              {msg.text}
            </p>
          </div>
        )}

        <div className="overflow-auto flex-1">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-full gap-2">
              <Loader className="animate-spin text-rose-400" size={28} />
              <p className="text-slate-400 text-xs">Loading watchlist...</p>
            </div>
          ) : filteredRows.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-2">
              <div className="p-3 bg-slate-700/30 rounded-full">
                <Search size={32} className="text-slate-500" />
              </div>
              <p className="text-slate-400 text-sm">
                {searchQuery
                  ? `No subjects found matching "${searchQuery}"`
                  : "No subjects enrolled in the watchlist."}
              </p>
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-slate-900/60 border-b border-slate-700/50">
                <tr>
                  <th className="text-left p-3 font-semibold text-slate-300 w-1/4">
                    <div className="flex items-center gap-2">
                      <span>🆔</span>
                      <span>Person ID</span>
                    </div>
                  </th>
                  <th className="text-left p-3 font-semibold text-slate-300 w-1/4">
                    <div className="flex items-center gap-2">
                      <span>👤</span>
                      <span>Name</span>
                    </div>
                  </th>
                  <th className="text-left p-3 font-semibold text-slate-300 w-1/6">
                    <div className="flex items-center gap-2">
                      <span>📊</span>
                      <span>NH</span>
                    </div>
                  </th>
                  <th className="text-left p-3 font-semibold text-slate-300 w-1/6">
                    <div className="flex items-center gap-2">
                      <span>📈</span>
                      <span>HDIC</span>
                    </div>
                  </th>
                  <th className="text-left p-3 font-semibold text-slate-300 w-1/6">
                    <div className="flex items-center gap-2">
                      <span>⚡</span>
                      <span>Actions</span>
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/50">
                {filteredRows.map((r, index) => (
                  <tr
                    key={r.person_id}
                    className="group hover:bg-slate-700/30 transition-all duration-200"
                  >
                    <td className="p-3 w-1/4">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-gradient-to-br from-rose-500/20 to-purple-500/20 rounded-lg flex items-center justify-center ring-1 ring-rose-500/30 group-hover:ring-rose-500/50 transition-all flex-shrink-0">
                          <span className="text-rose-400 font-bold text-xs">
                            {r.person_id.slice(0, 2).toUpperCase()}
                          </span>
                        </div>
                        <span className="font-mono text-slate-200 text-sm">
                          {r.person_id}
                        </span>
                      </div>
                    </td>
                    <td className="p-3 text-slate-300 w-1/4">
                      {r.name || "—"}
                    </td>
                    <td className="p-3 w-1/6">
                      <span className="inline-flex items-center px-2.5 py-1 bg-emerald-500/10 rounded-md ring-1 ring-emerald-500/30 text-emerald-400 font-semibold text-sm">
                        {r.nh_count}
                      </span>
                    </td>
                    <td className="p-3 w-1/6">
                      <span className="inline-flex items-center px-2.5 py-1 bg-cyan-500/10 rounded-md ring-1 ring-cyan-500/30 text-cyan-400 font-semibold text-sm">
                        {r.hdic_count}
                      </span>
                    </td>
                    <td className="p-3 w-1/6">
                      <Button
                        onClick={() => openDeleteConfirm(r)}
                        variant="danger"
                        disabled={deletingId === r.person_id}
                        className="px-3 py-1.5 text-xs"
                      >
                        {deletingId === r.person_id ? (
                          <span className="flex items-center gap-1.5">
                            <Loader size={12} className="animate-spin" />
                            Deleting
                          </span>
                        ) : (
                          <span className="flex items-center gap-1.5">
                            <X size={12} />
                            Delete
                          </span>
                        )}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {!isLoading && filteredRows.length > 0 && (
          <div className="grid grid-cols-3 gap-3 p-3 border-t border-slate-700/50 bg-slate-900/40 flex-shrink-0">
            <div className="flex items-center gap-2 p-2 bg-emerald-500/5 rounded-lg border border-emerald-500/20">
              <UserCircle
                className="text-emerald-400 flex-shrink-0"
                size={18}
              />
              <div>
                <p className="text-emerald-400 text-base font-bold leading-none">
                  {filteredRows.length}
                </p>
                <p className="text-slate-400 text-[10px] leading-none mt-1">
                  Persons
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 p-2 bg-cyan-500/5 rounded-lg border border-cyan-500/20">
              <div className="text-cyan-400 flex-shrink-0 text-lg">📊</div>
              <div>
                <p className="text-cyan-400 text-base font-bold leading-none">
                  {filteredRows.reduce((sum, r) => sum + r.nh_count, 0)}
                </p>
                <p className="text-slate-400 text-[10px] leading-none mt-1">
                  NH Images
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 p-2 bg-purple-500/5 rounded-lg border border-purple-500/20">
              <div className="text-purple-400 flex-shrink-0 text-lg">📈</div>
              <div>
                <p className="text-purple-400 text-base font-bold leading-none">
                  {filteredRows.reduce((sum, r) => sum + r.hdic_count, 0)}
                </p>
                <p className="text-slate-400 text-[10px] leading-none mt-1">
                  HDIC Images
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteConfirm.isOpen}
        onClose={closeDeleteConfirm}
        onConfirm={confirmDelete}
        title="Delete Person"
        message={`Are you sure you want to delete ${deleteConfirm.personId}${
          deleteConfirm.personName ? ` (${deleteConfirm.personName})` : ""
        }? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        type="danger"
      />
    </>
  );
}

function ManualCheckView({ alerts, loading, onReload, onDecide }) {
  async function decide(a, decision) {
    await onDecide(a, decision);
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl ring-1 ring-slate-700/50 border-t border-slate-700/80 shadow-xl overflow-hidden flex flex-col h-full max-h-[calc(100vh-9rem)]">
      {/* Header */}
      <div className="relative overflow-hidden bg-gradient-to-br from-cyan-900/30 via-slate-900/30 to-blue-900/30 p-5 border-b border-slate-700/50 flex-shrink-0">
        <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/10 rounded-full blur-3xl animate-pulse"></div>

        <div className="relative z-10 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-cyan-500/20 rounded-lg ring-1 ring-cyan-500/50">
              <Search size={24} className="text-cyan-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 via-blue-300 to-cyan-500 bg-clip-text text-transparent">
                Manual Verification Queue
              </h2>
              <p className="text-slate-500 text-[10px]">
                {alerts.length} {alerts.length === 1 ? "alert" : "alerts"} in
                queue
              </p>
            </div>
          </div>

          <Button
            onClick={onReload}
            variant="secondary"
            disabled={loading}
            className="px-3 py-2 text-xs flex-shrink-0"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            <span className="ml-1.5">Reload</span>
          </Button>
        </div>
      </div>

      {/* Content Area - Scrollable with Snap and Scroll Padding */}
      <div className="overflow-auto flex-1 p-4 scroll-smooth snap-y snap-mandatory scroll-pt-3">
        {loading && alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <Loader className="animate-spin text-cyan-400" size={32} />
            <p className="text-slate-400 text-sm">Loading alerts...</p>
          </div>
        ) : alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <div className="p-4 bg-slate-700/30 rounded-full">
              <Search size={40} className="text-slate-500" />
            </div>
            <p className="text-slate-400 text-base font-medium">
              No alerts in the verification queue.
            </p>
            <p className="text-slate-500 text-sm">
              All alerts have been processed
            </p>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {alerts.map((a, i) => (
              <div
                key={i}
                className="group bg-slate-900/60 backdrop-blur-sm rounded-xl border border-slate-700/50 hover:border-cyan-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/10 overflow-hidden flex flex-col snap-start"
              >
                {/* Image */}
                <div className="relative overflow-hidden aspect-[4/3]">
                  <img
                    src={`${API}${a.file_path}`}
                    alt="capture"
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>

                  {/* Status Badge on Image */}
                  <div className="absolute top-2 right-2">
                    <span
                      className={`text-[10px] font-bold px-2 py-1 rounded-full backdrop-blur-sm ${
                        a.status === "pending"
                          ? "bg-amber-500/30 text-amber-300 ring-1 ring-amber-400/50"
                          : a.status === "confirmed"
                          ? "bg-emerald-500/30 text-emerald-300 ring-1 ring-emerald-400/50"
                          : "bg-rose-500/30 text-rose-300 ring-1 ring-rose-400/50"
                      }`}
                    >
                      {a.status.toUpperCase()}
                    </span>
                  </div>

                  {/* Matched Person ID Badge on Image */}
                  <div className="absolute bottom-2 left-2 right-2">
                    <div className="flex items-center gap-1.5 bg-black/60 backdrop-blur-md px-2 py-1.5 rounded-lg border border-white/20">
                      <span className="text-[9px] text-slate-400 font-semibold uppercase tracking-wider">
                        Match:
                      </span>
                      <span className="font-mono text-xs font-bold text-emerald-400">
                        {a.person_id}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Details */}
                <div className="p-3 flex-1 flex flex-col">
                  <div className="space-y-2 flex-1">
                    {/* Matched Person ID - Prominent Display */}
                    <div className="bg-gradient-to-r from-cyan-500/10 to-emerald-500/10 rounded-lg p-2 border border-cyan-500/20">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                          Matched Person
                        </span>
                        <div className="flex items-center gap-1.5">
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                          <span className="font-mono text-sm font-bold text-emerald-400">
                            {a.person_id}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Timestamp */}
                    <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                      <span className="opacity-50">🕐</span>
                      <span>{a.timestamp}</span>
                    </div>

                    {/* Decision Time (if decided) */}
                    {a.status !== "pending" && a.decision_time && (
                      <div className="text-[10px] text-slate-500 bg-slate-800/50 px-2 py-1 rounded">
                        <span className="font-semibold">Decided:</span>{" "}
                        {a.decision_time}
                      </div>
                    )}
                  </div>

                  {/* Action Buttons */}
                  {a.status === "pending" ? (
                    <div className="flex gap-2 mt-3">
                      <Button
                        onClick={() => decide(a, "confirm")}
                        variant="primary"
                        className="flex-1 text-xs py-2 px-2"
                      >
                        <span className="flex items-center justify-center gap-1">
                          <span className="text-base">✓</span>
                          <span>Confirm</span>
                        </span>
                      </Button>
                      <Button
                        onClick={() => decide(a, "reject")}
                        variant="danger"
                        className="flex-1 text-xs py-2 px-2"
                      >
                        <span className="flex items-center justify-center gap-1">
                          <X size={12} />
                          <span>Reject</span>
                        </span>
                      </Button>
                    </div>
                  ) : (
                    <div
                      className={`mt-3 text-center text-xs font-semibold py-2 px-2 rounded-lg ${
                        a.status === "confirmed"
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                          : "bg-rose-500/10 text-rose-400 border border-rose-500/30"
                      }`}
                    >
                      {a.status === "confirmed" ? "✓ Confirmed" : "✗ Rejected"}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Stats Footer */}
      {!loading && alerts.length > 0 && (
        <div className="grid grid-cols-3 gap-3 p-3 border-t border-slate-700/50 bg-slate-900/40 flex-shrink-0">
          <div className="flex items-center gap-2 p-2 bg-amber-500/5 rounded-lg border border-amber-500/20">
            <div className="text-amber-400 flex-shrink-0 text-lg">⏳</div>
            <div>
              <p className="text-amber-400 text-base font-bold leading-none">
                {alerts.filter((a) => a.status === "pending").length}
              </p>
              <p className="text-slate-400 text-[10px] leading-none mt-1">
                Pending
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 bg-emerald-500/5 rounded-lg border border-emerald-500/20">
            <div className="text-emerald-400 flex-shrink-0 text-lg">✓</div>
            <div>
              <p className="text-emerald-400 text-base font-bold leading-none">
                {alerts.filter((a) => a.status === "confirmed").length}
              </p>
              <p className="text-slate-400 text-[10px] leading-none mt-1">
                Confirmed
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 bg-rose-500/5 rounded-lg border border-rose-500/20">
            <div className="text-rose-400 flex-shrink-0 text-lg">✗</div>
            <div>
              <p className="text-rose-400 text-base font-bold leading-none">
                {alerts.filter((a) => a.status === "rejected").length}
              </p>
              <p className="text-slate-400 text-[10px] leading-none mt-1">
                Rejected
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
/* -------------------------- Global Styles & Animations -------------------------- */
const style = document.createElement("style");
style.innerHTML = `
@keyframes fadeIn { 
  from { opacity: 0; transform: translateY(20px); } 
  to { opacity: 1; transform: translateY(0); } 
}
.animate-fadeIn { 
  animation: fadeIn 0.6s ease-out; 
}
.delay-1000 {
  animation-delay: 1s;
}
@keyframes shimmer {
  0% { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
}
`;
document.head.appendChild(style);
