import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

const METRICS = ["f1", "recall", "pr_auc", "mcc"];

export default function BeforeAfterChart({ beforeAfter, modelKey, height = 300 }) {
  const comp = beforeAfter?.[modelKey];
  if (!comp || !comp.before || !comp.after) {
    return <p className="text-sm text-slate-400 py-8 text-center">No before/after comparison available for this model.</p>;
  }

  const data = METRICS.map((m) => ({
    metric: m.replace("_", " ").toUpperCase(),
    Before: comp.before[m] != null ? Number(comp.before[m].toFixed(3)) : null,
    After: comp.after[m] != null ? Number(comp.after[m].toFixed(3)) : null,
  })).filter((d) => d.Before !== null && d.After !== null);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
        <XAxis dataKey="metric" tick={{ fontSize: 12, fill: "#475569" }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 12, fill: "#475569" }} />
        <Tooltip formatter={(v) => v.toFixed(3)} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Before" fill="#cbd5e1" radius={[6, 6, 0, 0]} maxBarSize={40} />
        <Bar dataKey="After" fill="#4f46e5" radius={[6, 6, 0, 0]} maxBarSize={40} />
      </BarChart>
    </ResponsiveContainer>
  );
}
