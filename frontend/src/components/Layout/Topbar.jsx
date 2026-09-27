import React from "react";
import { useLocation } from "react-router-dom";
import { useApp } from "../../context/AppContext.jsx";

const TITLES = {
  "/": "Home",
  "/dataset": "Dataset",
  "/models": "Models",
  "/experiments/new": "Experiment Lab",
  "/dashboard": "Dashboard",
  "/compare": "Compare Models",
  "/imbalance-impact": "Imbalance Impact",
  "/insights": "Insights & Conclusion",
  "/history": "Experiment History",
  "/reports": "Reports",
};

export default function Topbar() {
  const { pathname } = useLocation();
  const { dataset, toast } = useApp();
  const title = TITLES[pathname] || (pathname.startsWith("/models/") ? "Model Detail" : "");

  return (
    <div className="sticky top-0 z-10 bg-surface/80 backdrop-blur border-b border-border">
      <div className="flex items-center justify-between px-6 py-4">
        <h1 className="text-lg font-semibold">{title}</h1>
        <div className="flex items-center gap-3">
          {dataset ? (
            <span className="chip chip-active">
              📁 {dataset.name} · {dataset.rows.toLocaleString()} rows
              {dataset.is_imbalanced ? ` · ${dataset.imbalance_ratio}:1 imbalance` : ""}
            </span>
          ) : (
            <span className="chip">No dataset selected</span>
          )}
        </div>
      </div>
      {toast && (
        <div
          className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-xl shadow-card text-sm font-medium ${
            toast.kind === "error"
              ? "bg-rose-50 text-rose-700 border border-rose-200"
              : toast.kind === "success"
              ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
              : "bg-slate-800 text-white"
          }`}
        >
          {toast.message}
        </div>
      )}
    </div>
  );
}
