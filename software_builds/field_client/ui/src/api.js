export const FIELD_API = import.meta.env.VITE_FIELD_API || "http://localhost:8081";
export const ADMIN_API = import.meta.env.VITE_ADMIN_API || "http://localhost:8082";

// ---- Field
export async function postMatch(blob) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  const r = await fetch(`${FIELD_API}/match`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
export async function health() {
  try {
    const r = await fetch(`${FIELD_API}/health`);
    return r.ok ? "ok" : "down";
  } catch {
    return "down";
  }
}

// ---- Admin (manual checks)
export async function sendAlertToAdmin({ blob, person_id, score, timestamp }) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  fd.append("person_id", person_id);
  fd.append("score", String(score ?? 0));
  fd.append("timestamp", timestamp);
  const r = await fetch(`${ADMIN_API}/manual_check/receive`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchAlerts() {
  const r = await fetch(`${ADMIN_API}/manual_check/list`);
  if (!r.ok) throw new Error(await r.text());
  return r.json(); // {alerts:[...]}
}

export async function imageUrlForAlert(file) {
  return `${ADMIN_API}/uploads/${file}`;
}
