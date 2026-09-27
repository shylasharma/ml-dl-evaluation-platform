import React, { useState } from "react";
import { Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import MetricSelector from "../components/MetricSelector.jsx";
import BarChartMetric from "../components/charts/BarChartMetric.jsx";
import RadarCompare from "../components/charts/RadarCompare.jsx";

export default function Compare() {
  const { activeExperiment } = useApp();
  const [metric, setMetric] = useState("f1");
  const [selected, setSelected] = useState(null); // null = all

  if (!activeExperiment || activeExperiment.status !== "completed") {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">⚖️</div>
        <p className="font-semibold">No experiment results to compare yet</p>
        <Link to="/experiments/new" className="btn-primary mt-5 inline-flex">Run New Experiment →</Link>
      </div>
    );
  }

  const results = activeExperiment.results.results;
  const chosenKeys = selected || results.map((r) => r.model_key);
  const chosen = results.filter((r) => chosenKeys.includes(r.model_key));

  const toggle = (key) => {
    const base = selected || results.map((r) => r.model_key);
    setSelected(base.includes(key) ? base.filter((k) => k !== key) : [...base, key]);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="card p-5">
        <p className="font-semibold mb-3">Select Models to Compare</p>
        <div className="flex flex-wrap gap-2">
          {results.map((r) => (
            <button
              key={r.model_key}
              onClick={() => toggle(r.model_key)}
              className={`chip ${chosenKeys.includes(r.model_key) ? "chip-active" : ""}`}
            >
              {r.family === "DL" ? "🧬" : "📈"} {r.model_label}
            </button>
          ))}
        </div>
      </div>

      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <p className="font-semibold">Metric Comparison</p>
          <MetricSelector value={metric} onChange={setMetric} compact />
        </div>
        <BarChartMetric data={chosen} metric={metric} />
      </div>

      <div className="card p-5">
        <p className="font-semibold mb-3">Multi-Metric Radar</p>
        <RadarCompare results={chosen} />
      </div>

      <div className="card p-5 overflow-x-auto scroll-thin">
        <p className="font-semibold mb-3">Side-by-Side Table</p>
        <table className="text-sm w-full border-collapse min-w-[720px]">
          <thead>
            <tr className="text-left text-xs text-slate-500 border-b border-border">
              <th className="py-2 pr-4">Model</th>
              {["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "g_mean"].map((k) => (
                <th key={k} className="py-2 pr-4">{k.replace("_", " ")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {chosen.filter((r) => !r.error).map((row) => (
              <tr key={row.model_key} className="border-b border-border/60">
                <td className="py-2 pr-4 font-medium">
                  <Link to={`/models/${row.model_key}?experiment=${activeExperiment.id}`} className="text-primary-600">{row.model_label}</Link>
                </td>
                {["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc", "balanced_accuracy", "g_mean"].map((k) => (
                  <td key={k} className="py-2 pr-4">{row[k] != null ? row[k].toFixed(3) : "—"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
