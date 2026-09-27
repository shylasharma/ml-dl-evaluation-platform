import React from "react";
import { Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";

export default function Insights() {
  const { activeExperiment } = useApp();

  if (!activeExperiment || activeExperiment.status !== "completed") {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">💡</div>
        <p className="font-semibold">No insights yet</p>
        <p className="text-sm text-slate-500 mt-1">Run an experiment first to generate data-driven insights.</p>
        <Link to="/experiments/new" className="btn-primary mt-5 inline-flex">Run New Experiment →</Link>
      </div>
    );
  }

  const { insights, conclusion } = activeExperiment;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="card p-6">
        <p className="font-semibold text-lg mb-1">💡 Automatic Insights</p>
        <p className="text-sm text-slate-500 mb-4">
          Every statement below is generated directly from this experiment's results — nothing is pre-written.
        </p>
        <ul className="space-y-3">
          {insights?.map((insight, i) => (
            <li key={i} className="flex items-start gap-3 text-sm">
              <span className="mt-0.5 w-6 h-6 shrink-0 rounded-full bg-primary-50 text-primary-600 grid place-items-center text-xs font-bold">
                {i + 1}
              </span>
              <span className="text-slate-700">{insight}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="card p-6 bg-gradient-to-br from-slate-900 to-slate-800 text-white">
        <p className="font-semibold text-lg mb-2">📄 Automatic Conclusion</p>
        <p className="text-sm text-slate-200 leading-relaxed">{conclusion}</p>
      </div>

      <div className="flex justify-end gap-3">
        <Link to="/reports" className="btn-primary">Export Full Report →</Link>
      </div>
    </div>
  );
}
