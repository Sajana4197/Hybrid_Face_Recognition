import React, { useEffect, useState, useCallback, useMemo, useRef } from "react";
import {
  Menu, X, UserPlus, UserMinus, Search, Loader, RefreshCw,
  ChevronsLeft, ChevronsRight, UserCircle, Upload
} from "lucide-react";

const API = import.meta.env.VITE_ADMIN_API || "http://127.0.0.1:5002";

/* ---------------- Reusable UI ---------------- */
const Card = ({ title, children, actions = null }) => (
  <div className="bg-slate-800/50 backdrop-blur-lg rounded-xl p-5 sm:p-6 ring-1 ring-slate-700/50 border-t border-slate-700/80 shadow-2xl shadow-black/40 animate-fadeIn">
    <div className="flex items-center justify-between mb-4 border-b border-slate-700/80 pb-3">
      <h2 className="text-xl font-semibold text-slate-100 tracking-wide">{title}</h2>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
    <div>{children}</div>
  </div>
);

const Button = ({ children, onClick, variant="primary", type="button", disabled=false, className="" }) => {
  const base = "px-4 py-2 rounded-md font-semibold transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2";
  const variants = {
    primary: "bg-emerald-600/90 hover:bg-emerald-700 text-white focus:ring-emerald-500 shadow-lg shadow-emerald-600/20 hover:shadow-emerald-600/40",
    secondary: "bg-slate-700/80 hover:bg-slate-700 text-slate-200 focus:ring-slate-500",
    danger: "bg-rose-600/90 hover:bg-rose-800 text-white focus:ring-rose-500 shadow-lg shadow-rose-600/20 hover:shadow-rose-600/40",
  };
  return (
    <button type={type} onClick={onClick} className={`${base} ${variants[variant]} ${className}`} disabled={disabled}>
      {children}
    </button>
  );
};

const Input = ({ label, id, ...props }) => (
  <div>
    <label htmlFor={id} className="block text-sm font-medium text-slate-300 mb-1 tracking-wider">
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

/* ---------------- Main App ---------------- */
export default function App() {
  const [activeTab, setActiveTab] = useState("enroll");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [health, setHealth] = useState("checking");

  const [manualAlerts, setManualAlerts] = useState([]);
  const [alertsLoading, setAlertsLoading] = useState(true);

  const alertSoundRef = useRef(null);
  const lastAlertedTsRef = useRef(null);
  const audioPrimedRef = useRef(false);

  // Health
  useEffect(() => {
    fetch(`${API}/health`).then(r => setHealth(r.ok ? "ok":"down")).catch(()=>setHealth("down"));
  }, []);

  // Prime audio once
  useEffect(() => {
    const primeAudio = () => {
      if (alertSoundRef.current && !audioPrimedRef.current) {
        const a = alertSoundRef.current;
        a.volume = 0;
        const p = a.play();
        if (p) {
          p.then(() => { a.pause(); a.currentTime = 0; a.volume = 1; audioPrimedRef.current = true; })
           .catch(()=>{});
        }
      }
    };
    document.addEventListener("click", primeAudio, { capture: true, once: true });
    document.addEventListener("touchstart", primeAudio, { capture: true, once: true });
    return () => {
      document.removeEventListener("click", primeAudio, { capture: true });
      document.removeEventListener("touchstart", primeAudio, { capture: true });
    };
  }, []);

  const loadAlerts = useCallback(async () => {
    try {
      const r = await fetch(`${API}/manual_check/list`);
      const d = await r.json();
      const sorted = (d.alerts || []).sort((a,b)=> b.timestamp.localeCompare(a.timestamp));
      setManualAlerts(sorted);
    } catch(e) {
      console.warn("alerts load failed", e);
    }
  }, []);

  useEffect(() => {
    (async ()=>{ await loadAlerts(); setAlertsLoading(false); })();
    const t = setInterval(loadAlerts, 5000);
    return () => clearInterval(t);
  }, [loadAlerts]);

  // Play sound on new pending
  useEffect(() => {
    const newest = manualAlerts.find(a => a.status === "pending");
    if (!newest) return;
    if (lastAlertedTsRef.current && newest.timestamp <= lastAlertedTsRef.current) return;

    if (audioPrimedRef.current && alertSoundRef.current) {
      const a = alertSoundRef.current;
      a.currentTime = 0;
      a.play().catch(()=>{});
    }
    lastAlertedTsRef.current = newest.timestamp;
  }, [manualAlerts]);

  async function handleDecision(a, decision) {
    const fd = new FormData();
    fd.append("person_id", a.person_id);
    fd.append("timestamp", a.timestamp);
    fd.append("decision", decision);
    await fetch(`${API}/manual_check/decision`, { method: "POST", body: fd });
    await loadAlerts();
  }

  const [lastSeenPendingTs, setLastSeenPendingTs] = useState(null);
  const newPendingCount = useMemo(
    () => manualAlerts.filter(a => a.status === "pending" && (!lastSeenPendingTs || a.timestamp > lastSeenPendingTs)).length,
    [manualAlerts, lastSeenPendingTs]
  );

  const navGroups = [{
    title: "Main",
    items: [
      { id:"enroll", label:"Enroll", icon:<UserPlus size={20}/> },
      { id:"watchlist", label:"Watchlist", icon:<UserMinus size={20}/> },
      { id:"manual", label:"Manual Check", icon:<Search size={20}/>, badge: newPendingCount || null },
    ]
  }];

  const healthStyle = {
    checking: "bg-sky-600/80 ring-sky-500/50",
    ok: "bg-emerald-600/80 ring-emerald-500/50",
    down: "bg-rose-600/80 ring-rose-500/50",
  };
  const healthText = { checking:"Connecting...", ok:"System Online", down:"System Offline" };

  return (
    <div className="flex h-screen bg-black text-slate-200 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-black to-slate-900"></div>
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[80%] h-[80%] bg-gradient-radial from-emerald-500/10 to-transparent blur-3xl"></div>

      <audio ref={alertSoundRef} src="/new-verification.mp3" preload="auto" />

      <aside className={`fixed sm:relative z-20 flex h-full flex-col bg-slate-900/60 backdrop-blur-xl border-r border-emerald-500/10 transition-all duration-300 ease-in-out
        ${isCollapsed ? "sm:w-20" : "sm:w-64"} ${sidebarOpen ? "translate-x-0 w-64" : "-translate-x-full sm:translate-x-0"}`}>
        <div className={`flex items-center border-b border-slate-700/80 ${isCollapsed ? "justify-center" : "px-4"} h-20`}>
          {!isCollapsed ? <h1 className="text-lg font-bold text-emerald-400 tracking-widest">HFR PANEL</h1> : <h1 className="text-lg font-bold text-emerald-400">FR</h1>}
          <button onClick={()=>setSidebarOpen(false)} className="sm:hidden absolute right-4 text-slate-400 hover:text-white"><X size={20}/></button>
        </div>

        <nav className="flex-grow p-2 space-y-4">
          {navGroups.map(g=>(
            <div key={g.title}>
              {!isCollapsed && <h3 className="px-4 py-1 text-xs uppercase text-slate-500 font-semibold tracking-wider">{g.title}</h3>}
              <div className="space-y-1">
                {g.items.map(item=>(
                  <button key={item.id}
                    onClick={()=>{
                      if (item.id === "manual") {
                        const latest = manualAlerts.find(a=>a.status==="pending");
                        if (latest) setLastSeenPendingTs(latest.timestamp);
                      }
                      setActiveTab(item.id); setSidebarOpen(false);
                    }}
                    className={`relative w-full flex items-center gap-3 rounded-lg transition-all duration-300 ${isCollapsed ? "justify-center px-2 py-3" : "px-4 py-3"}
                      ${activeTab===item.id ? "bg-gradient-to-r from-emerald-500/25 via-emerald-600/25 to-emerald-400/20 border border-emerald-500/40 text-emerald-300 shadow-inner shadow-emerald-700/30 animate-glow" : "text-slate-400 hover:text-emerald-300 hover:bg-slate-700/60 border border-transparent"}`}>
                    {activeTab===item.id && <span className="absolute left-0 top-0 bottom-0 w-1 bg-emerald-400 rounded-r-md shadow-[0_0_10px_rgba(16,185,129,0.6)]"></span>}
                    {item.icon}
                    {!isCollapsed && <span className="font-medium text-sm whitespace-nowrap">{item.label}</span>}
                    {!isCollapsed && item.badge && <span className="ml-auto bg-rose-600 text-white text-xs font-bold px-1.5 py-0.5 rounded-full animate-pulse">{item.badge}</span>}
                    {isCollapsed && item.badge && <span className="absolute top-2 right-2 w-3 h-3 bg-rose-600 rounded-full border-2 border-slate-900 animate-pulse"></span>}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </nav>

        <div className="border-t border-slate-700/80 p-2">
          <div className="flex items-center justify-between p-2 rounded-md hover:bg-slate-700/50 transition-colors">
            <div className={`flex items-center gap-3 ${isCollapsed ? "justify-center w-full":""}`}>
              <UserCircle size={24} className="text-slate-400 flex-shrink-0" />
              {!isCollapsed && <span className="text-sm font-medium text-slate-300 whitespace-nowrap">Admin User</span>}
            </div>
            {!isCollapsed
              ? <Button onClick={()=>setIsCollapsed(true)} variant="secondary" className="hidden sm:block"><ChevronsLeft size={20}/></Button>
              : <Button onClick={()=>setIsCollapsed(false)} variant="secondary" className="hidden sm:block absolute right-2"><ChevronsRight size={20}/></Button>}
          </div>
        </div>
      </aside>

      {sidebarOpen && <div className="fixed inset-0 bg-black/60 backdrop-blur-sm sm:hidden z-10 animate-fadeIn" onClick={()=>setSidebarOpen(false)}/>}

      <main className="relative flex-1 p-4 sm:p-8 overflow-y-auto">
        <header className="flex items-center justify-between mb-8 h-12">
          <button className="sm:hidden text-slate-300 hover:text-white" onClick={()=>setSidebarOpen(true)}><Menu size={22}/></button>
          <h1 className="hidden sm:block text-2xl font-bold tracking-wider text-slate-300">ADMINISTRATION CONSOLE</h1>
          <span className={`px-4 py-1.5 rounded-full text-sm font-medium ring-1 ${{
            checking:"bg-sky-600/80 ring-sky-500/50",
            ok:"bg-emerald-600/80 ring-emerald-500/50",
            down:"bg-rose-600/80 ring-rose-500/50"
          }[health]}`}>{({checking:"Connecting...", ok:"System Online", down:"System Offline"})[health]}</span>
        </header>

        <div key={activeTab}>
          {activeTab === "enroll" && <EnrollView />}
          {activeTab === "watchlist" && <WatchlistView />}
          {activeTab === "manual" &&
            <ManualCheckView
              alerts={manualAlerts}
              loading={alertsLoading}
              onReload={async ()=>{ setAlertsLoading(true); await loadAlerts(); setAlertsLoading(false); }}
              onDecide={handleDecision}
            />}
        </div>
      </main>
    </div>
  );
}

/* ---------------- Views ---------------- */
function EnrollView() {
  const [pid,setPid] = useState("");
  const [name,setName] = useState("");
  const [files,setFiles] = useState([]);
  const [msg,setMsg] = useState({text:"",type:"info"});
  const [busy,setBusy] = useState(false);

  async function submit(e){
    e.preventDefault();
    if(!pid || files.length===0){
      setMsg({text:"Person ID and at least one folder/image are required.", type:"error"}); return;
    }
    setBusy(true); setMsg({text:"",type:"info"});
    const fd = new FormData();
    fd.append("person_id", pid);
    fd.append("name", name);
    [...files].forEach(f => fd.append("files", f));
    try{
      const r = await fetch(`${API}/enroll`, { method:"POST", body:fd });
      const d = await r.json();
      if(r.ok){
        setMsg({text:`Enrolled ${d.person_id} with ${d.added_images} images.`, type:"success"});
        setPid(""); setName(""); setFiles([]);
        const inp = document.getElementById("file-upload");
        if (inp) inp.value = "";
      } else {
        setMsg({text:`Error: ${d.detail || "Enrollment failed"}`, type:"error"});
      }
    }catch(err){ setMsg({text:`Network Error: ${err.message}`, type:"error"}); }
    finally{ setBusy(false); }
  }

  const colors = { info:"text-slate-400", success:"text-emerald-400", error:"text-rose-400" };

  return (
    <Card title="Enroll New Subject">
      <form onSubmit={submit}>
        <div className="grid sm:grid-cols-2 gap-4">
          <Input id="pid" label="Person ID" placeholder="e.g., n000123" value={pid} onChange={e=>setPid(e.target.value)} required />
          <Input id="name" label="Name (Optional)" placeholder="e.g., John Doe" value={name} onChange={e=>setName(e.target.value)} />
        </div>

        <div className="mt-4">
          <label className="block text-sm font-medium text-slate-300 mb-2 tracking-wider">Upload Images from Folder</label>
          <input id="file-upload" type="file" className="hidden" webkitdirectory="true" multiple onChange={e=>setFiles(e.target.files)} />
          <label htmlFor="file-upload" className="inline-flex items-center gap-2 px-4 py-2 rounded-md font-semibold transition-all duration-300 cursor-pointer bg-blue-500/80 hover:bg-blue-700 text-slate-200">
            <Upload size={16}/> Choose Folder…
          </label>
          {files.length>0 && <span className="ml-4 text-sm text-slate-400">{files.length} file(s) selected</span>}
        </div>

        <div className="mt-6 flex items-center gap-4">
          <Button type="submit" variant="primary" disabled={busy}>
            {busy && <Loader size={16} className="animate-spin" />} {busy ? "Enrolling..." : "Upload & Enroll"}
          </Button>
          {msg.text && <p className={`text-sm ${colors[msg.type]}`}>{msg.text}</p>}
        </div>
      </form>
    </Card>
  );
}

function WatchlistView(){
  const [rows,setRows] = useState([]);
  const [msg,setMsg] = useState({text:"",type:"info"});
  const [busy,setBusy] = useState(true);

  async function refresh(){
    setBusy(true); setMsg({text:"",type:"info"});
    try{
      const r = await fetch(`${API}/list`);
      const d = await r.json();
      setRows(d.persons || []);
    }catch(e){ setMsg({text:`Error: ${e.message}`, type:"error"}); }
    finally{ setBusy(false); }
  }
  useEffect(()=>{ refresh(); },[]);

  async function del(pid){
    if(!confirm(`Delete ${pid}? This cannot be undone.`)) return;
    const r = await fetch(`${API}/delete/${pid}`, {method:"DELETE"});
    if(r.ok){ setMsg({text:`Deleted ${pid}`, type:"success"}); refresh(); }
    else { const d = await r.json(); setMsg({text:`Error: ${d.detail || "Delete failed"}`, type:"error"}); }
  }

  return (
    <Card title="System Watchlist" actions={
      <Button onClick={refresh} variant="secondary" disabled={busy}>
        <RefreshCw size={16} className={busy?"animate-spin":""}/> Reload
      </Button>
    }>
      <div className="overflow-auto rounded-lg border border-slate-700/80 min-h-[10rem]">
        {busy ? <Spinner/> : rows.length===0 ? (
          <p className="text-center text-slate-400 p-6">No subjects enrolled.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-slate-700/50">
              <tr>
                {["Person ID","Name","NH Count","HDIC Count","Actions"].map(h=>(
                  <th key={h} className="text-left p-2 font-semibold tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/80">
              {rows.map(r=>(
                <tr key={r.person_id} className="odd:bg-slate-900/80 even:bg-slate-800/40">
                  <td className="p-2 font-mono">{r.person_id}</td>
                  <td className="p-2">{r.name || "N/A"}</td>
                  <td className="p-2">{r.nh_count}</td>
                  <td className="p-2">{r.hdic_count}</td>
                  <td className="p-2"><Button onClick={()=>del(r.person_id)} variant="danger" className="px-2 py-1 text-xs">Delete</Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {msg.text && <p className={`mt-3 text-sm ${msg.type==="error"?"text-rose-400":"text-emerald-400"}`}>{msg.text}</p>}
    </Card>
  );
}

function ManualCheckView({ alerts, loading, onReload, onDecide }){
  async function decide(a, decision){ await onDecide(a, decision); }
  return (
    <Card title="Manual Verification Queue" actions={
      <Button onClick={onReload} variant="secondary" disabled={loading}>
        <RefreshCw size={16} className={loading?"animate-spin":""}/> Reload
      </Button>
    }>
      {loading && alerts.length===0 ? <Spinner/> :
      alerts.length===0 ? (
        <p className="text-center text-slate-400 p-6">No alerts in the queue.</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {alerts.map((a,i)=>(
            <div key={i} className="bg-slate-900/80 p-4 rounded-lg border border-slate-700/80">
              <img src={`${API}${a.file_path}`} alt="capture" className="w-full h-48 object-cover rounded mb-3 border border-slate-600" />
              <div className="text-sm mb-3 space-y-1">
                <div><b>Person ID:</b> <span className="font-mono">{a.person_id}</span></div>
                <div><b>Similarity:</b> <span className="font-medium text-emerald-300">{a.similarity?.toFixed?.(3) ?? a.similarity}</span></div>
                <div><b>Time:</b> <span className="text-slate-400">{a.timestamp}</span></div>
                <div><b>Status:</b> <span className={`font-semibold ${a.status==="pending"?"text-amber-300":a.status==="confirmed"?"text-emerald-300":"text-rose-300"}`}>{a.status}</span></div>
              </div>
              {a.status==="pending" ? (
                <div className="flex gap-2">
                  <Button onClick={()=>decide(a,"confirm")} variant="primary" className="flex-1 text-sm">Confirm ✅</Button>
                  <Button onClick={()=>decide(a,"reject")} variant="danger" className="flex-1 text-sm">Reject ❌</Button>
                </div>
              ) : (
                <div className="text-xs text-slate-400"><b>Decided:</b> {a.decision_time || "-"}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
