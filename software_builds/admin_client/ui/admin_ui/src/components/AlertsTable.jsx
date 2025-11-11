import React from "react";
import { decideAlert } from "../api";

export default function AlertsTable({ alerts, refresh }) {
  async function action(a, decision) {
    await decideAlert({ person_id: a.person_id, timestamp: a.timestamp, decision });
    refresh();
  }
  return (
    <table border="1" cellPadding="6" style={{borderCollapse:"collapse", width:"100%"}}>
      <thead>
        <tr><th>Time</th><th>Person ID</th><th>Score</th><th>Status</th><th>Image</th><th>Action</th></tr>
      </thead>
      <tbody>
        {alerts.map((a,i) => (
          <tr key={i}>
            <td>{a.timestamp}</td>
            <td>{a.person_id}</td>
            <td>{(a.score ?? 0).toFixed(3)}</td>
            <td>{a.status}</td>
            <td>
              <img alt="" src={`${import.meta.env.VITE_ADMIN_API || "http://localhost:8082"}/uploads/${a.file}`} style={{height:64}}/>
            </td>
            <td>
              <button onClick={() => action(a, "confirm")} disabled={a.status!=="pending"}>Confirm</button>{" "}
              <button onClick={() => action(a, "reject")}  disabled={a.status!=="pending"}>Reject</button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
