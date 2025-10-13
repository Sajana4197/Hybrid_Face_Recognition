import React, { useEffect, useState, useRef } from "react";
const API = import.meta.env.VITE_ADMIN_API || "http://127.0.0.1:5002";

function TabBtn({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-lg font-medium ${
        active ? "bg-emerald-600" : "bg-slate-700 hover:bg-slate-600"
      }`}
    >
      {children}
    </button>
  );
}

export default function App() {
  const [tab, setTab] = useState("enroll");
  const [health, setHealth] = useState("down");

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => setHealth(r.ok ? "ok" : "down"))
      .catch(() => setHealth("down"));
  }, []);

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-5xl mx-auto">
        <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-6">
          <h1 className="text-2xl font-bold">
            Admin Console — Hybrid FR (NH + HDIC)
          </h1>
          <span
            className={`px-3 py-1 rounded-full text-sm ${
              health === "ok" ? "bg-emerald-600" : "bg-rose-600"
            }`}
          >
            {health === "ok" ? "Backend Connected" : "Backend Down"}
          </span>
        </header>

        <nav className="flex gap-2 mb-4">
          <TabBtn active={tab === "enroll"} onClick={() => setTab("enroll")}>
            Enroll
          </TabBtn>
          <TabBtn
            active={tab === "watchlist"}
            onClick={() => setTab("watchlist")}
          >
            Watchlist
          </TabBtn>
          <TabBtn active={tab === "config"} onClick={() => setTab("config")}>
            Config
          </TabBtn>
        </nav>

        <div className="bg-slate-800/70 rounded-2xl p-5 ring-1 ring-slate-700">
          {tab === "enroll" && <EnrollView />}
          {tab === "watchlist" && <WatchlistView />}
          {tab === "config" && <ConfigView />}
        </div>
      </div>
    </div>
  );
}

function EnrollView() {
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [pid, setPid] = useState("");
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]);
  const [msg, setMsg] = useState("");

  async function submit() {
    if (!pid || files.length === 0) {
      setMsg("Provide person_id and at least one image.");
      return;
    }

    const validFiles = [...files].filter((f) => f.type.startsWith("image/"));
    const form = new FormData();
    form.append("person_id", pid);
    form.append("name", name);
    validFiles.forEach((f) => form.append("files", f));

    setBusy(true);
    setProgress(0);
    setMsg("Enrolling...");

    try {
      // Create manual upload progress tracking
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API}/enroll`);

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          setProgress(percent);
        }
      };

      xhr.onload = () => {
        setBusy(false);
        if (xhr.status === 200) {
          const data = JSON.parse(xhr.responseText);
          setMsg(
            `✅ Enrolled ${data.person_id} — Added ${data.added_images}/${data.total}, Failed ${data.failed_images}`
          );
        } else {
          setMsg(`❌ Enrollment failed (${xhr.status})`);
        }
      };

      xhr.onerror = () => {
        setBusy(false);
        setMsg("❌ Network or server error during enrollment.");
      };

      xhr.send(form);
    } catch (e) {
      setBusy(false);
      setMsg(`❌ Error: ${e}`);
    }
  }

  return (
    <div>
      <h2 className="text-xl font-semibold mb-3">Enroll Person</h2>
      <div className="grid sm:grid-cols-2 gap-3">
        <input
          className="bg-slate-700 rounded p-2"
          placeholder="Person ID (e.g., n000123)"
          value={pid}
          onChange={(e) => setPid(e.target.value)}
        />
        <input
          className="bg-slate-700 rounded p-2"
          placeholder="Name (optional)"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </div>
      <div className="mt-3">
        <input
          type="file"
          webkitdirectory="true"
          directory="true"
          onChange={(e) => setFiles(e.target.files)}
          className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700"
        />
        <p className="text-xs text-slate-400 mt-1">
          Select a folder containing multiple face images.
        </p>
      </div>
      <div className="mt-4">
        <button
          onClick={submit}
          className="px-5 py-2 rounded bg-blue-600 hover:bg-blue-700 font-medium"
        >
          Upload & Enroll
        </button>
      </div>
      <div className="mt-4 space-y-2">
        {busy && (
          <div className="w-full bg-slate-700 rounded-full h-3">
            <div
              className="bg-emerald-500 h-3 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}
        {msg && <p className="text-sm">{msg}</p>}
      </div>
    </div>
  );
}

function WatchlistView() {
  const [rows, setRows] = useState([]);
  const [detail, setDetail] = useState(null);
  const [msg, setMsg] = useState("");

  async function refresh() {
    setMsg("");
    try {
      const r = await fetch(`${API}/list`);
      const data = await r.json();
      setRows(data.persons || []);
    } catch (e) {
      setMsg(`❌ ${e}`);
    }
  }
  useEffect(() => {
    refresh();
  }, []);

  async function view(pid) {
    const r = await fetch(`${API}/view/${pid}`);
    const data = await r.json();
    setDetail(data);
  }
  async function del(pid) {
    if (!confirm(`Delete ${pid}?`)) return;
    const r = await fetch(`${API}/delete/${pid}`, { method: "DELETE" });
    if (r.ok) {
      setMsg("✅ deleted");
      refresh();
      setDetail(null);
    } else {
      const d = await r.json();
      setMsg(`❌ ${d.detail || "Delete failed"}`);
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-semibold">Watchlist</h2>
        <button
          onClick={refresh}
          className="px-3 py-1 rounded bg-slate-700 hover:bg-slate-600"
        >
          Reload
        </button>
      </div>
      <div className="overflow-auto rounded border border-slate-700">
        <table className="w-full text-sm">
          <thead className="bg-slate-700">
            <tr>
              <th className="text-left p-2">Person ID</th>
              <th className="text-left p-2">Name</th>
              <th className="text-left p-2">NH Count</th>
              <th className="text-left p-2">HDIC Count</th>
              <th className="text-left p-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.person_id}
                className="odd:bg-slate-800 even:bg-slate-900"
              >
                <td className="p-2">{r.person_id}</td>
                <td className="p-2">{r.name}</td>
                <td className="p-2">{r.nh_count}</td>
                <td className="p-2">{r.hdic_count}</td>
                <td className="p-2 flex gap-2">
                  <button
                    onClick={() => view(r.person_id)}
                    className="px-2 py-1 rounded bg-emerald-600"
                  >
                    View
                  </button>
                  <button
                    onClick={() => del(r.person_id)}
                    className="px-2 py-1 rounded bg-rose-600"
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {detail && (
        <div className="mt-4 bg-slate-900/60 rounded p-3 border border-slate-700">
          <div className="font-semibold mb-1">Details</div>
          <div>
            <b>PID:</b> {detail.person_id} | <b>Name:</b> {detail.name || "-"}
          </div>
          <div>
            <b>NH Hashes:</b> {detail.nh_hash_count}
          </div>
          <div>
            <b>HDIC Prototypes:</b> {detail.hdic_proto_keys?.length || 0}
          </div>
        </div>
      )}
      {msg && <p className="mt-3 text-sm">{msg}</p>}
    </div>
  );
}

function ConfigView() {
  const [cfg, setCfg] = useState({
    Tnh: 30,
    Thdic: 3100,
    fused_th: 0.7,
    w_nh: 0.5,
    w_hdic: 0.5,
  });
  const [msg, setMsg] = useState("");

  useEffect(() => {
    fetch(`${API}/config`)
      .then((r) => r.json())
      .then(setCfg)
      .catch(() => {});
  }, []);

  async function save() {
    setMsg("");
    try {
      const r = await fetch(`${API}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      const d = await r.json();
      setMsg(r.ok ? "✅ Saved" : `❌ ${d.detail || "Save failed"}`);
    } catch (e) {
      setMsg(`❌ ${e}`);
    }
  }

  const Input = ({ label, prop, step = 0.01 }) => (
    <label className="block mb-2">
      <span className="text-sm text-slate-300">{label}</span>
      <input
        type="number"
        step={step}
        value={cfg[prop]}
        onChange={(e) => setCfg({ ...cfg, [prop]: Number(e.target.value) })}
        className="mt-1 w-full bg-slate-700 rounded p-2"
      />
    </label>
  );

  return (
    <div>
      <h2 className="text-xl font-semibold mb-3">System Thresholds</h2>
      <div className="grid sm:grid-cols-2 gap-3">
        <Input label="Tnh (NeuralHash Hamming threshold)" prop="Tnh" step={1} />
        <Input label="Thdic (HDIC Hamming threshold)" prop="Thdic" step={1} />
        <Input label="Fused threshold (Sfinal)" prop="fused_th" />
        <Input label="w_nh (weight NH)" prop="w_nh" />
        <Input label="w_hdic (weight HDIC)" prop="w_hdic" />
      </div>
      <div className="mt-3">
        <button
          onClick={save}
          className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-700"
        >
          Save
        </button>
      </div>
      {msg && <p className="mt-3 text-sm">{msg}</p>}
    </div>
  );
}
