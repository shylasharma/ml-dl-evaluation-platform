import React from "react";
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Legend, Tooltip,
} from "recharts";

const METRICS = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc"];
const COLORS = ["#4f46e5", "#0d9488", "#d97706", "#e11d48", "#0284c7", "#7c3aed"];

export default function RadarCompare({ results }) {
  const usable = results.filter((r) => !r.error);
  if (usable.length === 0) {
    return <p className="text-sm text-slate-400 py-8 text-center">No completed models to compare.</p>;
  }

  const data = METRICS.map((metric) => {
    const row = { metric: metric.replace("_", " ").toUpperCase() };
    usable.forEach((r) => {
      // Normalize MCC (range -1..1) into 0..1 for a fair radar comparison
      const raw = r[metric];
      row[r.model_label] = metric === "mcc" && raw !== null ? (raw + 1) / 2 : raw;
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={380}>
      <RadarChart data={data} outerRadius={130}>
        <PolarGrid stroke="#e2e8f0" />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "#475569" }} />
        <PolarRadiusAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
        {usable.map((r, i) => (
          <Radar
            key={r.model_key}
            name={r.model_label}
            dataKey={r.model_label}
            stroke={COLORS[i % COLORS.length]}
            fill={COLORS[i % COLORS.length]}
            fillOpacity={0.12}
            strokeWidth={2}
          />
        ))}
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
