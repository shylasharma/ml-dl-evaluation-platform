import React from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import ModelSelector from "../components/ModelSelector.jsx";

export default function ModelsPage() {
  const { selectedModels, setSelectedModels, dataset, notify } = useApp();
  const navigate = useNavigate();

  const goToExperimentLab = () => {
    if (!dataset) {
      notify("Upload or select a dataset first.", "error");
      navigate("/dataset");
      return;
    }
    if (selectedModels.length === 0) {
      notify("Select at least one model to continue.", "error");
      return;
    }
    navigate("/experiments/new");
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="card p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="font-semibold">Select Models</p>
            <p className="text-sm text-slate-500 mt-0.5">
              Choose one, several, or all ML/DL models. Your selection carries over to Experiment Lab.
            </p>
          </div>
          <span className="chip chip-active">{selectedModels.length} selected</span>
        </div>
        <ModelSelector selected={selectedModels} onChange={setSelectedModels} />
      </div>

      <div className="flex justify-end">
        <button onClick={goToExperimentLab} className="btn-primary">
          Continue to Experiment Lab →
        </button>
      </div>

      <p className="text-xs text-slate-400">
        This catalog is modular: adding a new model on the backend (in <code>app/ml/registry.py</code>)
        makes it appear here automatically — no frontend changes required.
      </p>
    </div>
  );
}