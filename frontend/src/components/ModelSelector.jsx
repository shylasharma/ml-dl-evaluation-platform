import React, { useMemo, useState } from "react";
import { useApp } from "../context/AppContext.jsx";
import Tooltip from "./common/Tooltip.jsx";

export default function ModelSelector({ selected, onChange }) {
  const { modelsCatalog } = useApp();
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all"); // all | ml | dl | selected

  const all = [...modelsCatalog.ml_models, ...modelsCatalog.dl_models];

  const filtered = useMemo(() => {
    return all.filter((m) => {
      const matchesSearch = m.label.toLowerCase().includes(search.toLowerCase());
      const matchesCategory =
        category === "all" ||
        (category === "ml" && m.family === "ML") ||
        (category === "dl" && m.family === "DL") ||
        (category === "selected" && selected.includes(m.key));
      return matchesSearch && matchesCategory;
    });
  }, [all, search, category, selected]);

  const toggle = (key) => {
    onChange(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  };

  const selectAll = () => onChange(all.map((m) => m.key));
  const clearAll = () => onChange([]);
  const selectFamily = (fam) => onChange(all.filter((m) => m.family === fam).map((m) => m.key));

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
        <input
          type="text"
          placeholder="Search models…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="rounded-xl border border-border px-4 py-2.5 text-sm w-full sm:w-64 focus:outline-none focus:ring-2 focus:ring-primary-300"
        />
        <div className="flex flex-wrap gap-2 text-sm">
          <button onClick={selectAll} className="chip hover:bg-panel">Select All</button>
          <button onClick={() => selectFamily("ML")} className="chip hover:bg-panel">Select ML</button>
          <button onClick={() => selectFamily("DL")} className="chip hover:bg-panel">Select DL</button>
          <button onClick={clearAll} className="chip hover:bg-panel">Clear All</button>
        </div>
      </div>

      <div className="flex gap-2 text-sm">
        {[
          ["all", "All Models"],
          ["ml", "Machine Learning"],
          ["dl", "Deep Learning"],
          ["selected", `Selected (${selected.length})`],
        ].map(([key, label]) => (
          <button
            key={key}
            onClick={() => setCategory(key)}
            className={`chip ${category === key ? "chip-active" : "hover:bg-panel"}`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 max-h-[440px] overflow-y-auto scroll-thin pr-1">
        {filtered.map((m) => {
          const isSelected = selected.includes(m.key);
          return (
            <button
              key={m.key}
              onClick={() => toggle(m.key)}
              className={`text-left p-4 rounded-xl2 border transition-all ${
                isSelected
                  ? "border-primary-400 bg-primary-50 shadow-card"
                  : "border-border bg-surface hover:border-primary-200"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                    m.family === "DL" ? "bg-teal-50 text-accent-teal" : "bg-indigo-50 text-primary-600"
                  }`}>
                    {m.family}
                  </span>
                  <p className="font-semibold mt-2">{m.label}</p>
                </div>
                <div
                  className={`w-5 h-5 rounded-md border-2 grid place-items-center shrink-0 ${
                    isSelected ? "bg-primary-600 border-primary-600 text-white" : "border-slate-300"
                  }`}
                >
                  {isSelected && "✓"}
                </div>
              </div>
              <Tooltip text={m.description}>
                <p className="text-xs text-slate-500 mt-2 line-clamp-2">{m.description}</p>
              </Tooltip>
            </button>
          );
        })}
        {filtered.length === 0 && (
          <p className="text-sm text-slate-400 col-span-full py-6 text-center">No models match your search.</p>
        )}
      </div>
    </div>
  );
}
