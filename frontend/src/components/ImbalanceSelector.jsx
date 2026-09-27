import React from "react";
import { useApp } from "../context/AppContext.jsx";
import Tooltip from "./common/Tooltip.jsx";

export default function ImbalanceSelector({ value, onChange }) {
  const { imbalanceMethods } = useApp();

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {imbalanceMethods.map((m) => (
        <button
          key={m.key}
          onClick={() => onChange(m.key)}
          className={`text-left p-4 rounded-xl2 border transition-all ${
            value === m.key
              ? "border-primary-400 bg-primary-50 shadow-card"
              : "border-border bg-surface hover:border-primary-200"
          }`}
        >
          <div className="flex items-center justify-between">
            <p className="font-semibold text-sm">{m.label}</p>
            <div
              className={`w-4 h-4 rounded-full border-2 grid place-items-center ${
                value === m.key ? "border-primary-600" : "border-slate-300"
              }`}
            >
              {value === m.key && <div className="w-2 h-2 rounded-full bg-primary-600" />}
            </div>
          </div>
          <Tooltip text={m.description}>
            <p className="text-xs text-slate-500 mt-1.5 line-clamp-2">{m.description}</p>
          </Tooltip>
        </button>
      ))}
    </div>
  );
}
