import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import api from "../api/client.js";
import ProgressSteps from "../components/common/ProgressSteps.jsx";
import ModelSelector from "../components/ModelSelector.jsx";
import ImbalanceSelector from "../components/ImbalanceSelector.jsx";
import PreprocessingPanel, { DEFAULT_PREPROCESSING } from "../components/PreprocessingPanel.jsx";

const QUICK_MODEL_PRESET = ["logistic_regression", "random_forest", "xgboost", "svm", "ann"];

export default function NewExperiment() {
  const { dataset, notify, setActiveExperiment, refreshHistory } = useApp();
  const navigate = useNavigate();

  const [mode, setMode] = useState("quick"); // quick | advanced
  const [selectedModels, setSelectedModels] = useState([]);
  const [imbalanceMethod, setImbalanceMethod] = useState("smote");
  const [compareBeforeAfter, setCompareBeforeAfter] = useState(true);
  const [preprocessing, setPreprocessing] = useState({ ...DEFAULT_PREPROCESSING });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advanced, setAdvanced] = useState({
    test_size: 0.25, cv_folds: 0, cv_repeats: 1, random_state: 42,
    scale_features: true, dl_epochs: 30, dl_batch_size: 32, primary_metric: "f1",
  });
  const [running, setRunning] = useState(false);

  const step = !dataset ? 0 : selectedModels.length === 0 ? 1 : 2;

  const useQuickPreset = () => {
    setSelectedModels(QUICK_MODEL_PRESET);
    setMode("quick");
    notify("Quick Analysis preset selected: 4 strong ML models + 1 DL model.", "info");
  };
  const runExperiment = async () => {
    if (!dataset) return notify("Select a dataset first.", "error");
    if (selectedModels.length === 0) return notify("Select at least one model.", "error");

    setRunning(true);
    try {
      const config = {
        dataset_id: dataset.dataset_id,
        target_column: dataset.target_column,
        mode,
        models: selectedModels,
        imbalance_method: imbalanceMethod,
        compare_before_after: compareBeforeAfter && imbalanceMethod !== "none",
        preprocessing,
        ...advanced,
      };
      const result = await api.runExperiment(config);
      setActiveExperiment(result);
      refreshHistory();
      notify("Experiment completed successfully.", "success");
      navigate(`/dashboard?experiment=${result.id}`);
    } catch (err) {
      notify(err.message, "error");
    } finally {
      setRunning(false);
    }
  };

  if (!dataset) {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">📂</div>
        <p className="font-semibold">No dataset selected yet</p>
        <p className="text-sm text-slate-500 mt-1">Upload or choose a dataset before configuring an experiment.</p>
        <Link to="/dataset" className="btn-primary mt-5 inline-flex">Go to Dataset Page →</Link>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="card p-4 overflow-x-auto scroll-thin">
        <ProgressSteps steps={["Dataset", "Models", "Balancing", "Run"]} current={step} />
      </div>

      <div className="card p-5">
        <div className="flex items-center justify-between mb-2">
          <p className="font-semibold">1. Select Models</p>
          <button onClick={useQuickPreset} className="text-sm text-primary-600 font-medium">
            ⚡ Use Quick Analysis Preset
          </button>
        </div>
        <ModelSelector selected={selectedModels} onChange={setSelectedModels} />
      </div>

      <div className="card p-5">
        <p className="font-semibold mb-3">2. Imbalance-Handling Technique</p>
        <ImbalanceSelector value={imbalanceMethod} onChange={setImbalanceMethod} />
                {imbalanceMethod !== "none" && advanced.cv_folds === 0 && (
          <label className="flex items-center gap-2 mt-4 text-sm text-slate-600">
            <input
              type="checkbox"
              checked={compareBeforeAfter}
              onChange={(e) => setCompareBeforeAfter(e.target.checked)}
              className="rounded border-slate-300"
            />
            Also evaluate each model with <strong>No Balancing</strong> for a before/after comparison
          </label>
        )}
        {imbalanceMethod !== "none" && advanced.cv_folds > 0 && (
          <p className="text-xs text-slate-400 mt-3">
            Before/after comparison isn't available in cross-validation mode — run two separate
            experiments (one with "No Balancing", one with {imbalanceMethod}) to compare them.
          </p>
        )}
      </div>

      <div className="card p-5">
        <p className="font-semibold mb-3">3. Preprocessing</p>
        <PreprocessingPanel value={preprocessing} onChange={setPreprocessing} />
      </div>

      <div className="card p-5">
        <button
          onClick={() => setShowAdvanced((s) => !s)}
          className="flex items-center justify-between w-full font-semibold"
        >
          <span>4. Advanced Configuration <span className="text-xs font-normal text-slate-400">(optional)</span></span>
          <span className="text-slate-400">{showAdvanced ? "▲" : "▼"}</span>
        </button>
        {showAdvanced && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-4">
                        <Field label="Test Size">
              <input type="number" min="0.1" max="0.5" step="0.05" value={advanced.test_size}
                     onChange={(e) => { setMode("advanced"); setAdvanced({ ...advanced, test_size: Number(e.target.value) }); }}
                     className="input" />
            </Field>
            <Field label="Cross-Validation Folds">
              <select value={advanced.cv_folds}
                      onChange={(e) => { setMode("advanced"); setAdvanced({ ...advanced, cv_folds: Number(e.target.value) }); }}
                      className="input">
                <option value={0}>Off (single train/test split)</option>
                <option value={5}>5-fold</option>
                <option value={10}>10-fold</option>
              </select>
            </Field>
            {advanced.cv_folds > 0 && (
              <Field label="CV Repeats">
                <input type="number" min="1" max="10" value={advanced.cv_repeats}
                       onChange={(e) => setAdvanced({ ...advanced, cv_repeats: Number(e.target.value) })}
                       className="input" />
              </Field>
            )}
            <Field label="Random State">
              <input type="number" value={advanced.random_state}
                     onChange={(e) => { setMode("advanced"); setAdvanced({ ...advanced, random_state: Number(e.target.value) }); }}
                     className="input" />
            </Field>
            <Field label="DL Epochs">
              <input type="number" min="5" max="200" value={advanced.dl_epochs}
                     onChange={(e) => { setMode("advanced"); setAdvanced({ ...advanced, dl_epochs: Number(e.target.value) }); }}
                     className="input" />
            </Field>
            <Field label="DL Batch Size">
              <input type="number" min="8" max="256" value={advanced.dl_batch_size}
                     onChange={(e) => { setMode("advanced"); setAdvanced({ ...advanced, dl_batch_size: Number(e.target.value) }); }}
                     className="input" />
            </Field>
            <Field label="Primary Metric">
              <select value={advanced.primary_metric}
                      onChange={(e) => setAdvanced({ ...advanced, primary_metric: e.target.value })}
                      className="input">
                <option value="f1">F1 Score</option>
                <option value="recall">Recall</option>
                <option value="pr_auc">PR-AUC</option>
                <option value="mcc">MCC</option>
                <option value="balanced_accuracy">Balanced Accuracy</option>
              </select>
            </Field>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between card p-5">
        <div className="text-sm text-slate-500">
          <strong>{selectedModels.length}</strong> model(s) selected ·{" "}
          Balancing: <strong>{imbalanceMethod === "none" ? "No Balancing" : imbalanceMethod}</strong>
        </div>
        <button onClick={runExperiment} disabled={running || selectedModels.length === 0} className="btn-primary">
          {running ? "Running Analysis…" : "▶ Run Analysis"}
        </button>
      </div>
      {running && (
        <p className="text-sm text-slate-400 text-center">
          Training {selectedModels.length} model(s){compareBeforeAfter && imbalanceMethod !== "none" ? " (with before/after comparison)" : ""}.
          Deep learning models may take longer.
        </p>
      )}
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div>
      <label className="text-xs font-medium text-slate-500">{label}</label>
      <div className="mt-1">{children}</div>
    </div>
  );
}
