import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

export default function PrCurve({ curves, height = 300 }) {
  // curves: [{ label, points: [[recall, precision], ...], color }]
  const usable = curves.filter((c) => c.points && c.points.length > 0);
  if (usable.length === 0) {
    return <p className="text-sm text-slate-400 py-8 text-center">Precision-Recall curve unavailable for the selected model(s).</p>;
  }

  const maxLen = Math.max(...usable.map((c) => c.points.length));
  const merged = Array.from({ length: maxLen }, (_, i) => {
    const row = { idx: i };
    usable.forEach((c) => {
      const pt = c.points[Math.min(i, c.points.length - 1)];
      row[`${c.label}_x`] = pt[0];
      row[c.label] = pt[1];
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" dataKey={`${usable[0].label}_x`} data={merged} domain={[0, 1]}
               tick={{ fontSize: 11 }} label={{ value: "Recall", position: "insideBottom", offset: -5, fontSize: 11 }} />
        <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }}
               label={{ value: "Precision", angle: -90, position: "insideLeft", fontSize: 11 }} />
        <Tooltip formatter={(v) => v.toFixed(3)} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {usable.map((c) => (
          <Line key={c.label} data={merged} type="monotone" dataKey={c.label}
                stroke={c.color} strokeWidth={2} dot={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
