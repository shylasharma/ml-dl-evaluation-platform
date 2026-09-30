import React from "react";

export default function ProgressSteps({ steps, current, onStepClick }) {
  return (
    <div className="flex items-center gap-2 min-w-max">
      {steps.map((label, idx) => {
        const state =
          idx < current
            ? "done"
            : idx === current
            ? "active"
            : "pending";

        const clickable =
          Boolean(onStepClick) && idx >= 1 && idx <= current;

        return (
          <React.Fragment key={label}>
            <button
              type="button"
              onClick={() => clickable && onStepClick(idx)}
              disabled={!clickable}
              className={`flex items-center gap-2 rounded-lg px-1.5 py-1 transition-colors ${
                clickable
                  ? "cursor-pointer hover:bg-panel"
                  : "cursor-default"
              }`}
            >
              <div
                className={`w-7 h-7 rounded-full grid place-items-center text-xs font-semibold ${
                  state === "done"
                    ? "bg-primary-600 text-white"
                    : state === "active"
                    ? "bg-primary-100 text-primary-700 ring-2 ring-primary-300"
                    : "bg-slate-100 text-slate-400"
                }`}
              >
                {state === "done" ? "✓" : idx + 1}
              </div>

              <span
                className={`text-sm font-medium hidden sm:inline ${
                  state === "pending"
                    ? "text-slate-400"
                    : "text-ink"
                }`}
              >
                {label}
              </span>
            </button>

            {idx < steps.length - 1 && (
              <div className="w-6 sm:w-10 h-px bg-border" />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}