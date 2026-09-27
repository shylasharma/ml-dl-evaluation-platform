import React from "react";
import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Home", icon: "🏠", end: true },
  { to: "/dataset", label: "Dataset", icon: "📂" },
  { to: "/models", label: "Models", icon: "🧠" },
  { to: "/experiments/new", label: "Experiment Lab", icon: "🧪" },
  { to: "/dashboard", label: "Dashboard", icon: "📊" },
  { to: "/compare", label: "Compare", icon: "🆚" },
  { to: "/imbalance-impact", label: "Imbalance Impact", icon: "⚖️" },
  { to: "/insights", label: "Insights & Conclusion", icon: "💡" },
  { to: "/history", label: "History", icon: "🗂️" },
  { to: "/reports", label: "Reports", icon: "📄" },
];

export default function Sidebar() {
  return (
    <aside className="hidden md:flex flex-col w-60 shrink-0 border-r border-border bg-surface h-screen sticky top-0">
      <div className="px-5 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary-600 text-white grid place-items-center font-bold">Σ</div>
          <div>
            <p className="text-sm font-semibold leading-tight">ML/DL Evaluator</p>
            <p className="text-xs text-slate-400 leading-tight">Imbalance Research Platform</p>
          </div>
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? "bg-primary-50 text-primary-700"
                  : "text-slate-600 hover:bg-panel hover:text-ink"
              }`
            }
          >
            <span className="text-base">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="px-4 py-4 border-t border-border text-xs text-slate-400">
        v1.0 · Research build
      </div>
    </aside>
  );
}
