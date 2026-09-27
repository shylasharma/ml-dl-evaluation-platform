import React from "react";

export default function KpiCard({ label, value, sub, tone = "default" }) {
  const toneClasses = {
    default: "text-ink",
    primary: "text-primary-600",
    amber: "text-accent-amber",
    rose: "text-accent-rose",
    teal: "text-accent-teal",
  };
  return (
    <div className="card p-4 flex flex-col gap-1 min-w-[150px]">
      <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</span>
      <span className={`text-2xl font-bold ${toneClasses[tone] || toneClasses.default}`}>{value}</span>
      {sub && <span className="text-xs text-slate-400">{sub}</span>}
    </div>
  );
}
