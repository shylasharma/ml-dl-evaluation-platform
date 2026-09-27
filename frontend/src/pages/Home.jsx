import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";

const STEPS = [
  { icon: "📂", title: "Upload", desc: "Drop a CSV. Rows, features, and class imbalance are detected automatically.", to: "/dataset" },
  { icon: "🧠", title: "Select", desc: "Select one, multiple, or all ML/DL models.", to: "/models" },
  { icon: "⚖️", title: "Balance", desc: "SMOTE, ADASYN, class weighting, or none — compare their effect.", to: "/experiments/new" },
  { icon: "🧪", title: "Analyze", desc: "One click trains and evaluates every selected configuration.", to: "/experiments/new" },
  { icon: "📊", title: "Explore", desc: "Interactive dashboard, comparisons, and a data-driven conclusion.", to: "/dashboard" },
];

export default function Home() {
  const { dataset, experimentHistory, notify } = useApp();
  const navigate = useNavigate();

  const startQuick = () => {
    if (!dataset) {
      notify("Upload or select a dataset first — Quick Analysis needs data to run on.", "info");
      navigate("/dataset");
      return;
    }
    navigate("/experiments/new?mode=quick");
  };

  const startAdvanced = () => {
    if (!dataset) {
      notify("Upload or select a dataset first.", "info");
      navigate("/dataset");
      return;
    }
    navigate("/experiments/new");
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div className="card p-8 bg-gradient-to-br from-primary-600 to-primary-700 text-white border-none">
        <p className="text-sm font-medium text-primary-100 uppercase tracking-wide">Research Platform</p>
        <h2 className="text-3xl font-bold mt-2">Evaluate & Compare ML/DL Models on Imbalanced Data</h2>
        <p className="text-primary-100 mt-3 max-w-2xl">
          Upload a dataset, choose your models and balancing technique, and get a rigorous,
          data-driven comparison — accuracy alone won't cut it here.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-6 max-w-2xl">
          <button
            onClick={startQuick}
            className="text-left rounded-xl2 bg-white/95 hover:bg-white text-ink p-5 transition-colors"
          >
            <div className="text-2xl mb-1">🚀</div>
            <p className="font-semibold">Quick Analysis</p>
            <p className="text-xs text-slate-500 mt-1">
              Run a recommended analysis with minimal configuration — models and balancing pre-selected.
            </p>
          </button>
          <button
            onClick={startAdvanced}
            className="text-left rounded-xl2 bg-white/10 hover:bg-white/20 border border-white/30 text-white p-5 transition-colors"
          >
            <div className="text-2xl mb-1">⚙️</div>
            <p className="font-semibold">Advanced Analysis</p>
            <p className="text-xs text-primary-100 mt-1">
              Configure models, balancing technique, metrics, and experiment settings yourself.
            </p>
          </button>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-1">The Workflow — 5 Steps, Minimal Clicks</h3>
        <p className="text-sm text-slate-400 mb-4">Click any step below to jump straight to it.</p>
        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
          {STEPS.map((s, i) => (
            <Link
              key={s.title}
              to={s.to}
              className="card p-4 relative hover:border-primary-300 hover:shadow-card transition-all group"
            >
              <span className="absolute top-3 right-3 text-xs text-slate-300 font-bold">{i + 1}</span>
              <div className="text-2xl mb-2 group-hover:scale-110 transition-transform">{s.icon}</div>
              <p className="font-semibold text-sm">{s.title}</p>
              <p className="text-xs text-slate-500 mt-1">{s.desc}</p>
            </Link>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card p-5">
          <p className="font-semibold mb-1">📁 Current Dataset</p>
          {dataset ? (
            <div className="text-sm text-slate-600 mt-2 space-y-1">
              <p>{dataset.name} — {dataset.rows.toLocaleString()} rows, {dataset.n_features} features</p>
              <p>Imbalance ratio: <strong>{dataset.imbalance_ratio}:1</strong>{dataset.is_imbalanced ? " (imbalanced)" : ""}</p>
              <Link to="/dataset" className="text-primary-600 font-medium text-sm inline-block mt-2">View details →</Link>
            </div>
          ) : (
            <p className="text-sm text-slate-400 mt-2">No dataset loaded yet. <Link to="/dataset" className="text-primary-600 font-medium">Upload one →</Link></p>
          )}
        </div>
        <div className="card p-5">
          <p className="font-semibold mb-1">🗂️ Recent Experiments</p>
          {experimentHistory.length > 0 ? (
            <ul className="text-sm text-slate-600 mt-2 space-y-1.5">
              {experimentHistory.slice(0, 3).map((e) => (
                <li key={e.id} className="flex justify-between">
                  <span>#{e.id} · {e.dataset_name}</span>
                  <Link to={`/dashboard?experiment=${e.id}`} className="text-primary-600 font-medium">View →</Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-slate-400 mt-2">No experiments yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}