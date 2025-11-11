import React from "react";
export default function PersonsTable({ persons }) {
  return (
    <table border="1" cellPadding="6" style={{borderCollapse:"collapse", width:"100%"}}>
      <thead><tr><th>Person ID</th><th>Name</th><th>NH rows</th><th>HDIC clusters</th></tr></thead>
      <tbody>
        {persons.map(p => (
          <tr key={p.person_id}>
            <td>{p.person_id}</td>
            <td>{p.name}</td>
            <td>{p.nh_count}</td>
            <td>{p.hdic_count}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
