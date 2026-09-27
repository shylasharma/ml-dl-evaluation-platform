import React from "react";
import { Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import api from "../api/client.js";

const FORMATS = [
  { key: "pdf", label: "PDF Report", icon: "📄", desc: "Full formatted report with conclusion, insights, and comparison table." },
  { key: "excel", label: "Excel Workbook", icon: "📊", desc: "Results table plus a summary sheet." },
  { key: "csv", label: "CSV", icon: "📑", desc: "Raw comparison table for further analysis." },
  { key: "json", label: "JSON", icon: "🧩", desc: "Full machine-readable experiment record for reproducibility." },
];

export default function Reports() {
  const { activeExperiment } = useApp();

  if (!activeExperiment || activeExperiment.status !== "completed") {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">📄</div>
        <p className="font-semibold">No experiment selected</p>
        <p className="text-sm text-slate-500 mt-1">Run or open an experiment to export its report.</p>
        <Link to="/experiments/new" className="btn-primary mt-5 inline-flex">Run New Experiment →</Link>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="card p-6">
        <p className="font-semibold text-lg">Export Experiment #{activeExperiment.id}</p>
        <p className="text-sm text-slate-500 mt-1">
          {activeExperiment.dataset_name} · {activeExperiment.results.results.length} model(s) ·{" "}
          {activeExperiment.results.imbalance_method_label}
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {FORMATS.map((f) => (
          <a
            key={f.key}
            href={api.exportUrl(activeExperiment.id, f.key)}
            className="card p-5 hover:border-primary-300 transition-colors"
          >
            <div className="text-3xl mb-2">{f.icon}</div>
            <p className="font-semibold">{f.label}</p>
            <p className="text-xs text-slate-500 mt-1">{f.desc}</p>
            <p className="text-sm text-primary-600 font-medium mt-3">Download →</p>
          </a>
        ))}
      </div>

      <div className="card p-5">
        <p className="font-semibold mb-2">What's included</p>
        <ul className="text-sm text-slate-600 list-disc list-inside space-y-1">
          <li>Dataset details & class distribution</li>
          <li>Imbalance-handling method used</li>
          <li>Full model comparison table (all metrics)</li>
          <li>Automatically generated insights</li>
          <li>Automatically generated conclusion</li>
        </ul>
      </div>
    </div>
  );
}
