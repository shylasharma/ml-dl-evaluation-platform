import React, { useCallback, useRef, useState } from "react";
import api from "../api/client.js";
import { useApp } from "../context/AppContext.jsx";
import ClassDistribution from "./charts/ClassDistribution.jsx";

export default function DatasetUploader({ onLoaded }) {
  const { setDataset, notify } = useApp();
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [samples, setSamples] = useState([]);
  const inputRef = useRef(null);

  React.useEffect(() => {
    api.listSampleDatasets().then(setSamples).catch(() => {});
  }, []);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".csv")) {
        notify("Only CSV files are supported.", "error");
        return;
      }
      setLoading(true);
      try {
        const profile = await api.uploadDataset(file);
        setDataset(profile);
        notify(`Dataset "${profile.name}" loaded successfully.`, "success");
        onLoaded?.(profile);
      } catch (err) {
        notify(err.message, "error");
      } finally {
        setLoading(false);
      }
    },
    [setDataset, notify, onLoaded]
  );

  const loadSample = async (filename) => {
    setLoading(true);
    try {
      const profile = await api.loadSampleDataset(filename);
      setDataset(profile);
      notify(`Sample dataset "${profile.name}" loaded.`, "success");
      onLoaded?.(profile);
    } catch (err) {
      notify(err.message, "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={`card p-10 text-center cursor-pointer transition-colors border-dashed border-2 ${
          dragOver ? "border-primary-400 bg-primary-50" : "border-border"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <div className="text-4xl mb-3">📤</div>
        <p className="font-medium">Drag & drop a CSV file, or click to browse</p>
        <p className="text-sm text-slate-400 mt-1">
          The target/label column is detected automatically where possible.
        </p>
        {loading && <p className="text-sm text-primary-600 mt-3 animate-pulse">Analyzing dataset…</p>}
      </div>

      {samples.length > 0 && (
        <div>
          <p className="text-sm font-medium text-slate-500 mb-2">Or start with a sample dataset</p>
          <div className="flex flex-wrap gap-2">
            {samples.map((s) => (
              <button
                key={s.filename}
                onClick={() => loadSample(s.filename)}
                className="chip hover:bg-panel"
                disabled={loading}
              >
                🗂️ {s.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function DatasetProfileSummary({ profile }) {
  if (!profile) return null;
  return (
    <div className="space-y-5">
      {profile.warnings?.map((w, i) => (
        <div
          key={i}
          className={`rounded-xl px-4 py-3 text-sm font-medium ${
            w.toLowerCase().includes("imbalanced")
              ? "bg-amber-50 text-amber-700 border border-amber-200"
              : "bg-slate-50 text-slate-600 border border-border"
          }`}
        >
          {w.toLowerCase().includes("imbalanced") ? "⚠️ " : "ℹ️ "}
          {w}
        </div>
      ))}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Rows" value={profile.rows.toLocaleString()} />
        <Stat label="Features" value={profile.n_features} />
        <Stat label="Classes" value={profile.n_classes ?? (profile.is_binary ? 2 : "-")} />
        <Stat label="Imbalance Ratio" value={`${profile.imbalance_ratio}:1`} tone={profile.is_imbalanced ? "amber" : "default"} />
        <Stat label="Majority Class" value={profile.majority_class} />
        <Stat label="Minority Class" value={profile.minority_class} />
        <Stat label="Duplicate Rows" value={profile.duplicate_rows} />
        <Stat label="Missing Values" value={Object.values(profile.missing_values || {}).reduce((a, b) => a + b, 0)} />
      </div>

      <div className="card p-4">
        <p className="text-sm font-semibold mb-2">Class Distribution</p>
        <ClassDistribution classCounts={profile.class_counts} />
      </div>

      {profile.recommendations?.length > 0 && (
        <div className="card p-4">
          <p className="text-sm font-semibold mb-2">💡 Smart Recommendations</p>
          <ul className="text-sm text-slate-600 space-y-1.5 list-disc list-inside">
            {profile.recommendations.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </div>
      )}

      <div className="card p-4 overflow-x-auto scroll-thin">
        <p className="text-sm font-semibold mb-2">Preview (first rows)</p>
        <table className="text-xs w-full border-collapse">
          <thead>
            <tr>
              {profile.columns?.map((c) => (
                <th key={c} className="text-left px-2 py-1.5 border-b border-border font-medium text-slate-500 whitespace-nowrap">
                  {c}{c === profile.target_column ? " 🎯" : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {profile.preview?.slice(0, 8).map((row, i) => (
              <tr key={i} className="odd:bg-panel/50">
                {profile.columns?.map((c) => (
                  <td key={c} className="px-2 py-1.5 whitespace-nowrap">{String(row[c] ?? "—")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Stat({ label, value, tone = "default" }) {
  return (
    <div className="card p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`text-lg font-bold ${tone === "amber" ? "text-accent-amber" : "text-ink"}`}>{value}</p>
    </div>
  );
}
