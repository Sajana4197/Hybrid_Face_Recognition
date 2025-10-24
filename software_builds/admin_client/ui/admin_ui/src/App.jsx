import React, { useEffect, useState } from "react";
import {
  Menu,
  X,
  UserPlus,
  UserMinus,
  Search,
  BarChart2,
  Loader,
  RefreshCw,
  ChevronsLeft,
  ChevronsRight,
  UserCircle,
  Upload,
} from "lucide-react";

// The backend API URL is retrieved from environment variables, with a fallback for local development.
const API = import.meta.env.VITE_ADMIN_API || "http://127.0.0.1:5002";

/* --------------------------- Reusable UI Components --------------------------- */

const Card = ({ title, children, actions = null }) => (
  <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-5 sm:p-6 ring-1 ring-slate-700/50 border-t border-slate-700/80 shadow-2xl shadow-black/40 animate-fadeIn">
    <div className="flex items-center justify-between mb-4 border-b border-slate-700/80 pb-3">
      <h2 className="text-xl font-semibold text-slate-100 tracking-wide">
        {title}
      </h2>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
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
    "px-4 py-2 rounded-md font-semibold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2";
  const variants = {
    primary:
      "bg-emerald-600/90 hover:bg-emerald-700 text-white focus:ring-emerald-500 shadow-lg shadow-emerald-600/20 hover:shadow-emerald-600/40",
    secondary:
      "bg-slate-700/80 hover:bg-slate-700 text-slate-200 focus:ring-slate-500",
    danger:
      "bg-rose-600/90 hover:bg-rose-800 text-white focus:ring-rose-500 shadow-lg shadow-rose-600/20 hover:shadow-rose-600/40",
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
  <div>
    <label
      htmlFor={id}
      className="block text-sm font-medium text-slate-300 mb-1 tracking-wider"
    >
      {label}
    </label>
    <input
      id={id}
      {...props}
      className="w-full bg-slate-800/60 rounded-md p-2 text-slate-100 ring-1 ring-slate-700/50 focus:ring-2 focus:ring-emerald-500 focus:outline-none transition-all duration-300"
    />
  </div>
);

const Spinner = () => (
  <div className="flex justify-center p-4">
    <Loader className="animate-spin text-emerald-400" />
  </div>
);

const FileInput = (props) => (
  <input
    type="file"
    {...props}
    className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-blue-500/90 hover:file:bg-blue-600 file:text-white transition cursor-pointer"
  />
);

/* -------------------------------- Main App -------------------------------- */
export default function App() {
  const [activeTab, setActiveTab] = useState("enroll");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [health, setHealth] = useState("checking");

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => setHealth(r.ok ? "ok" : "down"))
      .catch(() => setHealth("down"));
  }, []);

  const navGroups = [
    {
      title: "Main",
      items: [
        { id: "enroll", label: "Enroll", icon: <UserPlus size={20} /> },
        { id: "watchlist", label: "Watchlist", icon: <UserMinus size={20} /> },
        { id: "manual", label: "Manual Check", icon: <Search size={20} /> },
      ],
    },
    {
      title: "System",
      items: [
        { id: "config", label: "Configuration", icon: <BarChart2 size={20} /> },
      ],
    },
  ];

  const healthStatus = {
    checking: { text: "Connecting...", color: "bg-sky-600/80 ring-sky-500/50" },
    ok: {
      text: "System Online",
      color: "bg-emerald-600/80 ring-emerald-500/50",
    },
    down: { text: "System Offline", color: "bg-rose-600/80 ring-rose-500/50" },
  };

  return (
    <div className="flex h-screen bg-black text-slate-200 overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-slate-900 via-black to-slate-900"></div>
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80%] h-[80%] bg-gradient-radial from-emerald-500/10 to-transparent blur-3xl"></div>

      <aside
        className={`fixed sm:relative z-20 flex h-full flex-col bg-slate-900/60 backdrop-blur-xl border-r border-emerald-500/10 transition-all duration-300 ease-in-out
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
            <h1 className="text-lg font-bold text-emerald-400 whitespace-nowrap tracking-widest">
              HFR PANEL
            </h1>
          )}
          {isCollapsed && (
            <h1 className="text-lg font-bold text-emerald-400">FR</h1>
          )}
          <button
            onClick={() => setSidebarOpen(false)}
            className="sm:hidden absolute right-4 text-slate-400 hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        <nav className="flex-grow p-2 space-y-4">
          {navGroups.map((group) => (
            <div key={group.title}>
              {!isCollapsed && (
                <h3 className="px-4 py-1 text-xs uppercase text-slate-500 font-semibold tracking-wider">
                  {group.title}
                </h3>
              )}
              <div className="space-y-1">
                {group.items.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActiveTab(item.id);
                      setSidebarOpen(false);
                    }}
                    className={`relative w-full flex items-center gap-3 rounded-lg transition-all duration-300 ${
                      isCollapsed ? "justify-center px-2 py-3" : "px-4 py-3"
                    }
                      ${
                        activeTab === item.id
                          ? "bg-gradient-to-r from-emerald-500/25 via-emerald-600/25 to-emerald-400/20 border border-emerald-500/40 text-emerald-300 shadow-inner shadow-emerald-700/30 animate-glow"
                          : "text-slate-400 hover:text-emerald-300 hover:bg-slate-700/60 border border-transparent"
                      }
                    `}
                  >
                    {activeTab === item.id && (
                      <span className="absolute left-0 top-0 bottom-0 w-1 bg-emerald-400 rounded-r-md shadow-[0_0_10px_rgba(16,185,129,0.6)]"></span>
                    )}
                    {item.icon}
                    {!isCollapsed && (
                      <span className="font-medium text-sm whitespace-nowrap">
                        {item.label}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          ))}
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
                  Admin User
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
          <h1 className="hidden sm:block text-2xl font-bold tracking-wider text-slate-300">
            ADMINISTRATION CONSOLE
          </h1>
          <span
            className={`px-4 py-1.5 rounded-full text-sm font-medium ring-1 ${healthStatus[health].color}`}
          >
            {healthStatus[health].text}
          </span>
        </header>
        <div key={activeTab}>
          {activeTab === "enroll" && <EnrollView />}
          {activeTab === "watchlist" && <WatchlistView />}
          {activeTab === "manual" && <ManualCheckView />}
          {activeTab === "config" && <ConfigView />}
        </div>
      </main>
    </div>
  );
}

/* ----------------------------- Feature Views (with Watchlist modified) ------------------------------ */
function EnrollView() {
  const [pid, setPid] = useState("");
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]);
  const [msg, setMsg] = useState({ text: "", type: "info" });
  const [isLoading, setIsLoading] = useState(false);

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
        document.getElementById("file-upload").value = "";
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

  const messageColors = {
    info: "text-slate-400",
    success: "text-emerald-400",
    error: "text-rose-400",
  };

  return (
    <Card title="Enroll New Subject">
      <form onSubmit={submit}>
        <div className="grid sm:grid-cols-2 gap-4">
          <Input
            id="pid"
            label="Person ID"
            placeholder="e.g., n000123"
            value={pid}
            onChange={(e) => setPid(e.target.value)}
            required
          />
          <Input
            id="name"
            label="Name (Optional)"
            placeholder="e.g., John Doe"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        <div className="mt-4">
          <label className="block text-sm font-medium text-slate-300 mb-2 tracking-wider">
            Upload Images from Folder
          </label>

          {/* The `webkitdirectory` attribute is added here */}
          <input
            id="file-upload"
            type="file"
            className="hidden"
            webkitdirectory="true"
            onChange={(e) => setFiles(e.target.files)}
          />

          <label
            htmlFor="file-upload"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md font-semibold transition-all duration-300 cursor-pointer bg-blue-500/80 hover:bg-blue-700 text-slate-200"
          >
            <Upload size={16} />
            Choose Folder...
          </label>

          {files.length > 0 && (
            <span className="ml-4 text-sm text-slate-400">
              {files.length} {files.length === 1 ? "file" : "files"} selected
            </span>
          )}
        </div>

        <div className="mt-6 flex items-center gap-4">
          <Button type="submit" variant="primary" disabled={isLoading}>
            {isLoading && <Loader size={16} className="animate-spin" />}
            {isLoading ? "Enrolling..." : "Upload & Enroll"}
          </Button>
          {msg.text && (
            <p className={`text-sm ${messageColors[msg.type]}`}>{msg.text}</p>
          )}
        </div>
      </form>
    </Card>
  );
}

function WatchlistView() {
  const [rows, setRows] = useState([]);
  const [msg, setMsg] = useState({ text: "", type: "info" });
  const [isLoading, setIsLoading] = useState(true);

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

  async function del(pid) {
    if (
      !confirm(
        `Are you sure you want to delete ${pid}? This action cannot be undone.`
      )
    )
      return;
    const r = await fetch(`${API}/delete/${pid}`, { method: "DELETE" });
    if (r.ok) {
      setMsg({ text: `Success: Person ${pid} deleted.`, type: "success" });
      refresh();
    } else {
      const d = await r.json();
      setMsg({ text: `Error: ${d.detail || "Delete failed"}`, type: "error" });
    }
  }

  return (
    <Card
      title="System Watchlist"
      actions={
        <Button onClick={refresh} variant="secondary" disabled={isLoading}>
          <RefreshCw size={16} className={isLoading ? "animate-spin" : ""} />{" "}
          Reload
        </Button>
      }
    >
      <div className="overflow-auto rounded-lg border border-slate-700/80 min-h-[10rem]">
        {isLoading ? (
          <Spinner />
        ) : rows.length === 0 ? (
          <p className="text-center text-slate-400 p-6">
            No subjects enrolled in the watchlist.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-slate-700/50">
              <tr>
                {["Person ID", "Name", "NH Count", "HDIC Count", "Actions"].map(
                  (h) => (
                    <th
                      key={h}
                      className="text-left p-2 font-semibold tracking-wider"
                    >
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/80">
              {rows.map((r) => (
                <tr
                  key={r.person_id}
                  className="odd:bg-slate-900/80 even:bg-slate-800/40"
                >
                  <td className="p-2 font-mono">{r.person_id}</td>
                  <td className="p-2">{r.name || "N/A"}</td>
                  <td className="p-2">{r.nh_count}</td>
                  <td className="p-2">{r.hdic_count}</td>
                  <td className="p-2">
                    <Button
                      onClick={() => del(r.person_id)}
                      variant="danger"
                      className="px-2 py-1 text-xs"
                    >
                      Delete
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </Card>
  );
}

function ManualCheckView() {
  const API = import.meta.env.VITE_ADMIN_API || "http://127.0.0.1:5002";
  const [alerts, setAlerts] = React.useState([]);
  const [loading, setLoading] = React.useState(false);

  async function loadAlerts() {
    setLoading(true);
    try {
      const r = await fetch(`${API}/manual_check/list`);
      const d = await r.json();
      setAlerts(d.alerts || []);
    } finally {
      setLoading(false);
    }
  }

  async function decide(a, decision) {
    const form = new FormData();
    form.append("person_id", a.person_id);
    form.append("timestamp", a.timestamp);
    form.append("decision", decision);
    await fetch(`${API}/manual_check/decision`, { method: "POST", body: form });
    loadAlerts();
  }

  React.useEffect(() => {
    loadAlerts();
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-semibold">Manual Verification</h2>
        <button
          onClick={loadAlerts}
          className="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600"
        >
          Reload
        </button>
      </div>

      {loading && <p className="text-sm text-slate-400">Loading…</p>}

      {alerts.length === 0 ? (
        <p className="text-slate-400 text-sm">No alerts.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {alerts.map((a, i) => (
            <div
              key={i}
              className="bg-slate-800 p-4 rounded-lg border border-slate-700"
            >
              <img
                src={`${API}${a.file_path}`}
                alt="capture"
                className="w-full h-48 object-cover rounded mb-3 border border-slate-600"
              />
              <div className="text-sm mb-3">
                <div>
                  <b>Person ID:</b> {a.person_id}
                </div>
                <div>
                  <b>Similarity:</b>{" "}
                  {a.similarity?.toFixed?.(3) ?? a.similarity}
                </div>
                <div>
                  <b>Time:</b> {a.timestamp}
                </div>
                <div>
                  <b>Status:</b>{" "}
                  <span
                    className={
                      a.status === "pending"
                        ? "text-amber-300"
                        : a.status === "confirmed"
                        ? "text-emerald-300"
                        : "text-rose-300"
                    }
                  >
                    {a.status}
                  </span>
                </div>
              </div>

              {a.status === "pending" ? (
                <div className="flex gap-2">
                  <button
                    onClick={() => decide(a, "confirm")}
                    className="flex-1 bg-emerald-600 hover:bg-emerald-700 px-3 py-1 rounded text-sm"
                  >
                    Confirm ✅
                  </button>
                  <button
                    onClick={() => decide(a, "reject")}
                    className="flex-1 bg-rose-600 hover:bg-rose-700 px-3 py-1 rounded text-sm"
                  >
                    Reject ❌
                  </button>
                </div>
              ) : (
                <div className="text-xs text-slate-400">
                  <b>Decided:</b> {a.decision_time || "-"}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ConfigView() {
  const [cfg, setCfg] = useState({});
  const [msg, setMsg] = useState({ text: "", type: "info" });
  const [isLoading, setIsLoading] = useState(true);
  useEffect(() => {
    fetch(`${API}/config`)
      .then((r) => r.json())
      .then(setCfg)
      .catch(() => {})
      .finally(() => setIsLoading(false));
  }, []);
  async function save() {
    setIsLoading(true);
    setMsg({ text: "", type: "info" });
    try {
      const r = await fetch(`${API}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      const d = await r.json();
      setMsg(
        r.ok
          ? { text: "Success: Configuration saved!", type: "success" }
          : { text: `Error: ${d.detail || "Save failed"}`, type: "error" }
      );
    } catch (e) {
      setMsg({ text: `Error: ${e.message}`, type: "error" });
    } finally {
      setIsLoading(false);
    }
  }
  const ConfigInput = ({ label, prop, step = 0.01 }) => (
    <Input
      label={label}
      id={prop}
      type="number"
      step={step}
      value={cfg[prop] || ""}
      onChange={(e) => setCfg({ ...cfg, [prop]: Number(e.target.value) })}
    />
  );
  const messageColors = {
    info: "text-slate-400",
    success: "text-emerald-400",
    error: "text-rose-400",
  };
  return (
    <Card title="System Thresholds & Configuration">
      {isLoading ? (
        <Spinner />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-4">
            <ConfigInput label="Tnh (NeuralHash)" prop="Tnh" step={1} />
            <ConfigInput label="Thdic (HDIC)" prop="Thdic" step={1} />
            <ConfigInput label="Fused Threshold (Sfinal)" prop="fused_th" />
            <ConfigInput label="Weight NH (w_nh)" prop="w_nh" />
            <ConfigInput label="Weight HDIC (w_hdic)" prop="w_hdic" />
          </div>
          <div className="mt-6 flex items-center gap-4">
            <Button onClick={save} variant="primary" disabled={isLoading}>
              {isLoading && <Loader size={16} className="animate-spin" />} Save
              Configuration
            </Button>
            {msg.text && (
              <p className={`text-sm ${messageColors[msg.type]}`}>{msg.text}</p>
            )}
          </div>
        </>
      )}
    </Card>
  );
}

/* -------------------------- Global Styles & Animations -------------------------- */
const style = document.createElement("style");
style.innerHTML = `
@keyframes pulseGlow {
  0%, 100% {
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.4), inset 0 0 6px rgba(16, 185, 129, 0.2);
  }
  50% {
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.7), inset 0 0 10px rgba(16, 185, 129, 0.4);
  }
}
.animate-glow {
  animation: pulseGlow 2.5s ease-in-out infinite;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.animate-fadeIn { animation: fadeIn 0.5s ease-in-out; }`;
document.head.appendChild(style);
