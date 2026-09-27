import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList,
} from "recharts";

const ML_COLOR = "#4f46e5";
const DL_COLOR = "#0d9488";

export default function BarChartMetric({ data, metric, height = 320 }) {
  const rows = data
    .filter((r) => !r.error && r[metric] !== null && r[metric] !== undefined)
    .map((r) => ({ name: r.model_label, value: Number(r[metric].toFixed(3)), family: r.family }))
    .sort((a, b) => b.value - a.value);

  if (rows.length === 0) {
    return <p className="text-sm text-slate-400 py-8 text-center">No data available for this metric.</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
        <XAxis dataKey="name" angle={-30} textAnchor="end" interval={0} height={60}
               tick={{ fontSize: 12, fill: "#475569" }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 12, fill: "#475569" }} />
        <Tooltip formatter={(v) => v.toFixed(3)} />
        <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={56}>
          {rows.map((r, i) => (
            <Cell key={i} fill={r.family === "DL" ? DL_COLOR : ML_COLOR} />
          ))}
          <LabelList dataKey="value" position="top" style={{ fontSize: 11, fill: "#334155" }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
