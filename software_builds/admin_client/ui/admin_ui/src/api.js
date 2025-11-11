export const ADMIN_API = import.meta.env.VITE_ADMIN_API || "http://localhost:8082";

export async function fetchPersons() {
  const r = await fetch(`${ADMIN_API}/list`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
export async function fetchAlerts() {
  const r = await fetch(`${ADMIN_API}/manual_check/list`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
export async function decideAlert({ person_id, timestamp, decision }) {
  const fd = new FormData();
  fd.append("person_id", person_id);
  fd.append("timestamp", timestamp);
  fd.append("decision", decision);
  const r = await fetch(`${ADMIN_API}/manual_check/decision`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
