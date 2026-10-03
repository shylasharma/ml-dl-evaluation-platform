import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useApp } from "../context/AppContext.jsx";
import api from "../api/client.js";
import ProgressSteps from "../components/common/ProgressSteps.jsx";
import ModelSelector from "../components/ModelSelector.jsx";
import ImbalanceSelector from "../components/ImbalanceSelector.jsx";
import PreprocessingPanel, {
  DEFAULT_PREPROCESSING,
} from "../components/PreprocessingPanel.jsx";
import FeatureSelectionPanel from "../components/FeatureSelectionPanel.jsx";
import PCAPanel from "../components/PCAPanel.jsx";
import ModelRecommendationPanel from "../components/ModelRecommendationPanel.jsx";
import PCAComparisonPanel from "../components/PCAComparisonPanel.jsx";

// -----------------------------------------------------------------------------
// Quick Analysis preset
// -----------------------------------------------------------------------------

const QUICK_MODEL_PRESET = [
  "logistic_regression",
  "random_forest",
  "xgboost",
  "svm",
  "decision_tree",
];

// -----------------------------------------------------------------------------
// Wizard steps
// -----------------------------------------------------------------------------

const STEPS = [
  "Preprocessing",
  "Models",
  "Balancing",
  "Run",
];

// -----------------------------------------------------------------------------
// Main component
// -----------------------------------------------------------------------------

export default function NewExperiment() {
  const {
    dataset,
    notify,
    setActiveExperiment,
    refreshHistory,
  } = useApp();

  const navigate = useNavigate();

  const [currentStep, setCurrentStep] = useState(1);

  const [mode, setMode] = useState("quick");

  const [selectedModels, setSelectedModels] = useState([]);

  const [imbalanceMethod, setImbalanceMethod] =
    useState("smote");

  const [compareBeforeAfter, setCompareBeforeAfter] =
    useState(true);

  // ---------------------------------------------------------------------------
  // Preprocessing
  // ---------------------------------------------------------------------------

  const [preprocessing, setPreprocessing] = useState({
    ...DEFAULT_PREPROCESSING,
  });

  // ---------------------------------------------------------------------------
  // Feature Selection
  // ---------------------------------------------------------------------------

  const [featureSelection, setFeatureSelection] =
    useState({
      enabled: false,
      method: "none",
      k: null,
      threshold: null,
      correlation_threshold: 0.9,
      scoring: "f1_weighted",
      direction: "forward",
      step: 1,
      max_features: null,
    });

  // ---------------------------------------------------------------------------
  // PCA
  //
  // The backend already supports PCA. The current UI version of this file did
  // not define the state even though it attempted to send `pca: pca`.
  //
  // We keep the configuration available and safely disabled by default.
  // ---------------------------------------------------------------------------

  const [pca, setPca] = useState({
    enabled: false,
    mode: "variance",
    variance: 0.95,
    n_components: null,
  });

  const [comparePca, setComparePca] = useState(false);

  // ---------------------------------------------------------------------------
  // Advanced configuration
  // ---------------------------------------------------------------------------

  const [showAdvanced, setShowAdvanced] =
    useState(false);

  const [advanced, setAdvanced] = useState({
    test_size: 0.25,
    cv_folds: 0,
    cv_repeats: 1,
    random_state: 42,
    scale_features: true,
    dl_epochs: 30,
    dl_batch_size: 32,
    primary_metric: "f1",
  });

  const [running, setRunning] = useState(false);

  // ---------------------------------------------------------------------------
  // Quick preset
  // ---------------------------------------------------------------------------

  const useQuickPreset = () => {
    setSelectedModels(QUICK_MODEL_PRESET);
    setMode("quick");

    notify(
      "Quick Analysis preset selected: 5 ML models.",
      "info"
    );
  };

  // ---------------------------------------------------------------------------
  // Navigation
  // ---------------------------------------------------------------------------

  const goNext = () =>
    setCurrentStep((s) => Math.min(4, s + 1));

  const goBack = () =>
    setCurrentStep((s) => Math.max(1, s - 1));

  // ---------------------------------------------------------------------------
  // Run experiment
  // ---------------------------------------------------------------------------

  const runExperiment = async () => {
    if (!dataset) {
      return notify(
        "Select a dataset first.",
        "error"
      );
    }

    if (selectedModels.length === 0) {
      return notify(
        "Select at least one model.",
        "error"
      );
    }

    // -----------------------------------------------------------------------
    // PCA validation
    // -----------------------------------------------------------------------

    if (
      pca?.enabled &&
      pca?.mode === "components" &&
      (
        !pca.n_components ||
        pca.n_components < 1 ||
        (dataset.n_features &&
          pca.n_components > dataset.n_features)
      )
    ) {
      return notify(
        `PCA components must be between 1 and ${dataset.n_features || "the available feature count"}.`,
        "error"
      );
    }

    setRunning(true);

    try {
      const config = {
        dataset_id: dataset.dataset_id,
        target_column: dataset.target_column,

        mode,

        models: selectedModels,

        imbalance_method: imbalanceMethod,

        compare_before_after:
          compareBeforeAfter &&
          imbalanceMethod !== "none",

        preprocessing,

        feature_selection:
          featureSelection,

        pca,

        compare_pca: comparePca,


        ...advanced,
      };

      const result =
        await api.runExperiment(config);

      setActiveExperiment(result);

      refreshHistory();

      notify(
        "Experiment completed successfully.",
        "success"
      );

      navigate(
        `/dashboard?experiment=${result.id}`
      );
    } catch (err) {
      notify(
        err.message ||
          "Experiment failed.",
        "error"
      );
    } finally {
      setRunning(false);
    }
  };

  // ---------------------------------------------------------------------------
  // No dataset
  // ---------------------------------------------------------------------------

  if (!dataset) {
    return (
      <div className="max-w-2xl mx-auto card p-10 text-center">
        <div className="text-4xl mb-3">
          📂
        </div>

        <p className="font-semibold">
          No dataset selected yet
        </p>

        <p className="text-sm text-slate-500 mt-1">
          Upload or choose a dataset before
          configuring an experiment.
        </p>

        <Link
          to="/dataset"
          className="btn-primary mt-5 inline-flex"
        >
          Go to Dataset Page →
        </Link>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Step title
  // ---------------------------------------------------------------------------

  const stepTitle = [
    "Dataset",
    "Prepare your data",
    "Select models to evaluate",
    "Choose imbalance handling",
    "Review experiment & run",
  ][currentStep];

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="max-w-5xl mx-auto space-y-6">

      {/* ------------------------------------------------------------------ */}
      {/* Progress bar                                                       */}
      {/* ------------------------------------------------------------------ */}

      <div className="card p-4 overflow-x-auto scroll-thin">
        <ProgressSteps
          steps={STEPS}
          current={currentStep}
          onStepClick={(idx) => {
            if (
              idx >= 1 &&
              idx <= currentStep
            ) {
              setCurrentStep(idx);
            }
          }}
        />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Header                                                             */}
      {/* ------------------------------------------------------------------ */}

      <div className="flex items-start justify-between gap-4">

        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-primary-600">
            Experiment Lab
          </p>

          <h1 className="text-2xl font-bold mt-1">
            {stepTitle}
          </h1>

          <p className="text-sm text-slate-500 mt-1">
            Configure one experiment at a time so
            every preprocessing, feature-selection,
            balancing and model choice is explicit
            and reproducible.
          </p>
        </div>

        <div className="hidden sm:flex items-center gap-2 chip">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          Dataset ready
        </div>

      </div>

      {/* ================================================================== */}
      {/* STEP 1 — PREPROCESSING                                            */}
      {/* ================================================================== */}

      {currentStep === 1 && (
        <PreprocessingStep
          dataset={dataset}
          value={preprocessing}
          onChange={setPreprocessing}
          featureSelection={featureSelection}
          setFeatureSelection={
            setFeatureSelection
          }
          pca={pca}
          setPca={setPca}
          comparePca={comparePca}
          setComparePca={setComparePca}
          onBack={() =>
            navigate("/dataset")
          }
          onNext={goNext}
        />
      )}

      {/* ================================================================== */}
      {/* STEP 2 — MODELS                                                   */}
      {/* ================================================================== */}

      {currentStep === 2 && (
        <ModelsStep
          dataset={dataset}
          selectedModels={selectedModels}
          setSelectedModels={
            setSelectedModels
          }
          onQuickPreset={useQuickPreset}
          onBack={goBack}
          onNext={goNext}
        />
      )}

      {/* ================================================================== */}
      {/* STEP 3 — BALANCING                                                */}
      {/* ================================================================== */}

      {currentStep === 3 && (
        <BalancingStep
          imbalanceMethod={imbalanceMethod}
          setImbalanceMethod={
            setImbalanceMethod
          }
          compareBeforeAfter={
            compareBeforeAfter
          }
          setCompareBeforeAfter={
            setCompareBeforeAfter
          }
          cvFolds={advanced.cv_folds}
          onBack={goBack}
          onNext={goNext}
        />
      )}

      {/* ================================================================== */}
      {/* STEP 4 — REVIEW / RUN                                             */}
      {/* ================================================================== */}

      {currentStep === 4 && (
        <RunStep
          dataset={dataset}
          preprocessing={preprocessing}
          selectedModels={selectedModels}
          imbalanceMethod={imbalanceMethod}
          pca={pca}
          setPca={setPca}
          comparePca={comparePca}
          compareBeforeAfter={
            compareBeforeAfter
          }
          advanced={advanced}
          setAdvanced={setAdvanced}
          showAdvanced={showAdvanced}
          setShowAdvanced={
            setShowAdvanced
          }
          mode={mode}
          setMode={setMode}
          running={running}
          onBack={goBack}
          onRun={runExperiment}
        />
      )}

    </div>
  );
}


/* ============================================================================
   PREPROCESSING STEP
============================================================================ */

function PreprocessingStep({
  dataset,
  value,
  onChange,
  featureSelection,
  setFeatureSelection,
  pca,
  setPca,
  comparePca,
  setComparePca,
  onBack,
  onNext,
}) {
  return (
    <div className="space-y-5">

      {/* Explanation */}

      <div className="card p-5 border-primary-100 bg-primary-50/40">

        <div className="flex items-start gap-3">

          <div className="w-10 h-10 rounded-xl bg-primary-100 text-primary-700 grid place-items-center text-lg">
            🧹
          </div>

          <div>

            <p className="font-semibold">
              Why preprocessing comes first
            </p>

            <p className="text-sm text-slate-600 mt-1 max-w-3xl">
              These settings define how raw features
              are cleaned and transformed before model
              training. The backend fits learned
              transformations on training data and
              then applies them to validation/test data
              to reduce the risk of data leakage.
            </p>

          </div>

        </div>

      </div>

      {/* Preprocessing controls */}

      <div className="card p-5">

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">

          <div>

            <p className="font-semibold text-lg">
              Data Preprocessing
            </p>

            <p className="text-sm text-slate-500 mt-1">
              Configure the preparation pipeline for
              this experiment.
            </p>

          </div>

          <div className="chip">

            <span className="text-slate-400">
              Dataset
            </span>

            <strong className="text-ink">
              {dataset.name}
            </strong>

          </div>

        </div>

        <PreprocessingPanel
          value={value}
          onChange={onChange}
        />

        <FeatureSelectionPanel
          value={featureSelection}
          onChange={setFeatureSelection}
          featureCount={dataset.n_features}
        />

        <PCAPanel
          value={pca}
          onChange={setPca}
          featureCount={dataset.n_features}
        />

        <PCAComparisonPanel
          value={{ enabled: comparePca }}
          onChange={(enabled) => {
            setComparePca(enabled);
            if (enabled && !pca.enabled) {
              setPca((current) => ({ ...current, enabled: true }));
            }
          }}
        />

      </div>

      {/* Dataset stats */}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">

        <MiniStat
          label="Rows"
          value={
            dataset.rows?.toLocaleString?.() ??
            dataset.rows ??
            "—"
          }
        />

        <MiniStat
          label="Features"
          value={
            dataset.n_features ?? "—"
          }
        />

        <MiniStat
          label="Classes"
          value={
            dataset.n_classes ??
            (dataset.is_binary
              ? 2
              : "—")
          }
        />

        <MiniStat
          label="Target"
          value={
            dataset.target_column ?? "—"
          }
        />

      </div>

      <StepNavigation
        onBack={onBack}
        onNext={onNext}
        nextLabel="Continue to Model Selection →"
      />

    </div>
  );
}


/* ============================================================================
   MODEL STEP
============================================================================ */

function ModelsStep({
  dataset,
  selectedModels,
  setSelectedModels,
  onQuickPreset,
  onBack,
  onNext,
}) {
  const canContinue =
    selectedModels.length > 0;

  const applyRecommendedModels = (payload) => {
    const models = Array.isArray(payload)
      ? payload
      : payload?.recommended_models ||
        payload?.models ||
        payload?.selected_models ||
        [];

    if (!Array.isArray(models) || models.length === 0) {
      return;
    }

    setSelectedModels(models);
  };

  return (
    <div className="space-y-5">

      <ModelRecommendationPanel
        dataset={dataset}
        selectedModels={selectedModels}
        onApplyRecommendations={applyRecommendedModels}
        onUseRecommendations={applyRecommendedModels}
        onModelsRecommended={applyRecommendedModels}
      />

      <div className="card p-5">

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">

          <div>

            <p className="font-semibold text-lg">
              Classifier Selection
            </p>

            <p className="text-sm text-slate-500 mt-1">
              Choose the ML/DL classifiers that will
              receive the same prepared experiment
              configuration.
            </p>

          </div>

          <button
            onClick={onQuickPreset}
            className="btn-secondary text-sm"
          >
            ⚡ Use Quick Analysis Preset
          </button>

        </div>

        <ModelSelector
          selected={selectedModels}
          onChange={setSelectedModels}
        />

      </div>

      <div className="flex items-center justify-between text-sm text-slate-500">

        <span>
          <strong className="text-ink">
            {selectedModels.length}
          </strong>{" "}
          model(s) selected
        </span>

        {!canContinue && (
          <span className="text-amber-600">
            Select at least one model to
            continue.
          </span>
        )}

      </div>

      <StepNavigation
        onBack={onBack}
        onNext={onNext}
        nextLabel="Continue to Imbalance Handling →"
        nextDisabled={!canContinue}
      />

    </div>
  );
}


/* ============================================================================
   BALANCING STEP
============================================================================ */

function BalancingStep({
  imbalanceMethod,
  setImbalanceMethod,
  compareBeforeAfter,
  setCompareBeforeAfter,
  cvFolds,
  onBack,
  onNext,
}) {
  return (
    <div className="space-y-5">

      <div className="card p-5">

        <div className="mb-5">

          <p className="font-semibold text-lg">
            Imbalance Handling
          </p>

          <p className="text-sm text-slate-500 mt-1">
            Choose how the training data should
            handle minority classes. The test data
            remains untouched for evaluation.
          </p>

        </div>

        <ImbalanceSelector
          value={imbalanceMethod}
          onChange={setImbalanceMethod}
        />

        {imbalanceMethod !== "none" &&
          cvFolds === 0 && (

            <label className="flex items-start gap-2 mt-5 text-sm text-slate-600 p-3 rounded-xl bg-panel border border-border">

              <input
                type="checkbox"
                checked={compareBeforeAfter}
                onChange={(e) =>
                  setCompareBeforeAfter(
                    e.target.checked
                  )
                }
                className="rounded border-slate-300 mt-0.5"
              />

              <span>
                Also evaluate each model with{" "}
                <strong>
                  No Balancing
                </strong>{" "}
                as a baseline for before/after
                comparison.
              </span>

            </label>
          )}

        {imbalanceMethod !== "none" &&
          cvFolds > 0 && (

            <p className="text-xs text-slate-500 mt-4 p-3 rounded-xl bg-slate-50 border border-border">
              Before/after comparison is disabled
              in cross-validation mode. Run separate
              experiments for a direct baseline
              comparison.
            </p>

          )}

      </div>

      {/* Pipeline explanation */}

      <div className="card p-5">

        <p className="font-semibold">
          Current experiment flow
        </p>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 mt-4 text-sm">

          <FlowItem
            number="01"
            title="Preprocess"
            text="Clean and transform training features."
          />

          <FlowItem
            number="02"
            title="Features"
            text="Select useful features and optionally reduce dimensions with PCA."
          />

          <FlowItem
            number="03"
            title="Balance"
            text="Apply the selected training-set strategy."
          />

          <FlowItem
            number="04"
            title="Train"
            text="Fit every selected ML and DL classifier for direct comparison."
          />

          <FlowItem
            number="05"
            title="Evaluate"
            text="Calculate imbalance-aware metrics."
          />

        </div>

      </div>

      <StepNavigation
        onBack={onBack}
        onNext={onNext}
        nextLabel="Continue to Review & Run →"
      />

    </div>
  );
}


/* ============================================================================
   RUN / REVIEW STEP
============================================================================ */

function RunStep({
  dataset,
  preprocessing,
  selectedModels,
  imbalanceMethod,
  pca,
  comparePca,
  compareBeforeAfter,
  advanced,
  setAdvanced,
  showAdvanced,
  setShowAdvanced,
  mode,
  setMode,
  running,
  onBack,
  onRun,
}) {
  return (
    <div className="space-y-5">

      {/* ------------------------------------------------------------------ */}
      {/* Summary                                                            */}
      {/* ------------------------------------------------------------------ */}

      <div className="card p-5">

        <div className="flex items-start justify-between gap-4">

          <div>

            <p className="font-semibold text-lg">
              Experiment Summary
            </p>

            <p className="text-sm text-slate-500 mt-1">
              Review the configuration before
              starting model training.
            </p>

          </div>

          <span className="chip chip-active">Ready to run</span>

        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-5">

          <SummaryRow
            label="Dataset"
            value={dataset.name}
          />

          <SummaryRow
            label="Target"
            value={dataset.target_column}
          />

          <SummaryRow
            label="Models"
            value={`${selectedModels.length} selected`}
          />

          <SummaryRow
            label="Balancing"
            value={
              imbalanceMethod === "none"
                ? "No Balancing"
                : imbalanceMethod
            }
          />

          <SummaryRow
            label="Numeric missing values"
            value={
              preprocessing.missing_numeric
            }
          />

          <SummaryRow
            label="Categorical missing values"
            value={
              preprocessing.missing_categorical
            }
          />

          <SummaryRow
            label="Scaling"
            value={preprocessing.scaling}
          />

          <SummaryRow
            label="Encoding"
            value={preprocessing.encoding}
          />

          <SummaryRow
            label="Duplicates"
            value={preprocessing.duplicates}
          />

          <SummaryRow
            label="Baseline comparison"
            value={
              compareBeforeAfter &&
              imbalanceMethod !== "none"
                ? "Enabled"
                : "Disabled"
            }
          />

          <SummaryRow
            label="Feature Selection"
            value={
              featureSelectionLabel()
            }
          />

          <SummaryRow
            label="PCA"
            value={
              pca?.enabled
                ? pca.mode === "components"
                  ? `Enabled — ${pca.n_components ?? "Custom"} components`
                  : `Enabled — ${Math.round(
                      Number(pca.variance ?? 0.95) * 100
                    )}% variance`
                : "Disabled"
            }
          />

        </div>

      </div>

      {/* ------------------------------------------------------------------ */}
      {/* PCA status                                                         */}
      {/* ------------------------------------------------------------------ */}

      <div className="card p-5">

        <div className="flex items-start justify-between gap-4">

          <div>

            <p className="font-semibold">
              PCA / Dimensionality Reduction
            </p>

            <p className="text-sm text-slate-500 mt-1">
              PCA is configured in the preprocessing stage and
              will be applied after preprocessing and feature
              selection when enabled.
            </p>

          </div>

          <span className="chip">
            {pca?.enabled
              ? "Enabled"
              : "Disabled"}
          </span>

        </div>

        {pca?.enabled && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">

            <SummaryRow
              label="Mode"
              value={
                pca.mode === "components"
                  ? "Custom components"
                  : "Target variance"
              }
            />

            <SummaryRow
              label="Variance"
              value={
                `${Math.round(
                  Number(pca.variance) * 100
                )}%`
              }
            />

            <SummaryRow
              label="Components"
              value={
                pca.n_components ?? "Automatic"
              }
            />

          </div>
        )}

      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Advanced configuration                                             */}
      {/* ------------------------------------------------------------------ */}

      <div className="card p-5">

        <button
          onClick={() =>
            setShowAdvanced((s) => !s)
          }
          className="flex items-center justify-between w-full font-semibold text-left"
        >

          <span>
            Advanced Configuration{" "}
            <span className="text-xs font-normal text-slate-400">
              (optional)
            </span>
          </span>

          <span className="text-slate-400">
            {showAdvanced ? "▲" : "▼"}
          </span>

        </button>

        {showAdvanced && (

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-5">

            {/* Test size */}

            <Field label="Test Size">

              <input
                type="number"
                min="0.1"
                max="0.5"
                step="0.05"
                value={advanced.test_size}
                onChange={(e) => {

                  setMode("advanced");

                  setAdvanced({
                    ...advanced,
                    test_size:
                      Number(
                        e.target.value
                      ),
                  });

                }}
                className="input"
              />

            </Field>

            {/* CV folds */}

            <Field label="Cross-Validation Folds">

              <select
                value={advanced.cv_folds}
                onChange={(e) => {

                  setMode("advanced");

                  const nextCvFolds =
                    Number(
                      e.target.value
                    );

                  setAdvanced({
                    ...advanced,
                    cv_folds:
                      nextCvFolds,
                  });

                }}
                className="input"
              >

                <option value={0}>
                  Off (single train/test split)
                </option>

                <option value={5}>
                  5-fold
                </option>

                <option value={10}>
                  10-fold
                </option>

              </select>

            </Field>

            {/* CV repeats */}

            {advanced.cv_folds > 0 && (

              <Field label="CV Repeats">

                <input
                  type="number"
                  min="1"
                  max="10"
                  value={
                    advanced.cv_repeats
                  }
                  onChange={(e) =>
                    setAdvanced({
                      ...advanced,
                      cv_repeats:
                        Number(
                          e.target.value
                        ),
                    })
                  }
                  className="input"
                />

              </Field>

            )}

            {/* Random state */}

            <Field label="Random State">

              <input
                type="number"
                value={
                  advanced.random_state
                }
                onChange={(e) => {

                  setMode("advanced");

                  setAdvanced({
                    ...advanced,
                    random_state:
                      Number(
                        e.target.value
                      ),
                  });

                }}
                className="input"
              />

            </Field>

            {/* DL epochs */}

            <Field label="DL Epochs">

              <input
                type="number"
                min="5"
                max="200"
                value={
                  advanced.dl_epochs
                }
                onChange={(e) =>
                  setAdvanced({
                    ...advanced,
                    dl_epochs:
                      Number(
                        e.target.value
                      ),
                  })
                }
                className="input"
              />

            </Field>

            {/* DL batch size */}

            <Field label="DL Batch Size">

              <input
                type="number"
                min="8"
                max="256"
                value={
                  advanced.dl_batch_size
                }
                onChange={(e) =>
                  setAdvanced({
                    ...advanced,
                    dl_batch_size:
                      Number(
                        e.target.value
                      ),
                  })
                }
                className="input"
              />

            </Field>

            {/* Primary metric */}

            <Field label="Primary Metric">

              <select
                value={
                  advanced.primary_metric
                }
                onChange={(e) =>
                  setAdvanced({
                    ...advanced,
                    primary_metric:
                      e.target.value,
                  })
                }
                className="input"
              >

                <option value="f1">
                  F1 Score
                </option>

                <option value="recall">
                  Recall
                </option>

                <option value="pr_auc">
                  PR-AUC
                </option>

                <option value="mcc">
                  MCC
                </option>

                <option value="balanced_accuracy">
                  Balanced Accuracy
                </option>

              </select>

            </Field>

          </div>

        )}

      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Run                                                                 */}
      {/* ------------------------------------------------------------------ */}

      <div className="card p-5 border-primary-200 bg-primary-50/40">

        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">

          <div>

            <p className="font-semibold">
              Ready to run?
            </p>

            <p className="text-sm text-slate-600 mt-1">

              {selectedModels.length} model(s)
              will be evaluated using the
              configuration above.

            </p>

            {comparePca && (
              <p className="mt-2 text-sm text-primary-700">
                PCA comparison: the experiment will run once without PCA and once with PCA using the same configuration.
              </p>
            )}

          </div>

          <button
            onClick={onRun}
            disabled={
              running ||
              selectedModels.length === 0
            }
            className="btn-primary px-7"
          >
            {running
              ? "Running Analysis…"
              : "▶ Run Analysis"}
          </button>

        </div>

        {running && (

          <p className="text-sm text-slate-500 mt-4 text-center">

            Training{" "}
            {selectedModels.length}
            {" "}
            model(s)

            {compareBeforeAfter &&
            imbalanceMethod !== "none"
              ? " with baseline comparison"
              : ""}

            . Deep learning models may take
            longer.

          </p>

        )}

      </div>

      <StepNavigation
        onBack={onBack}
        nextLabel=""
        hideNext
      />

    </div>
  );

  // -------------------------------------------------------------------------
  // Helper used only by this component
  // -------------------------------------------------------------------------

  function featureSelectionLabel() {
    return "Configured in previous step";
  }
}


/* ============================================================================
   COMMON COMPONENTS
============================================================================ */

function StepNavigation({
  onBack,
  onNext,
  nextLabel,
  nextDisabled = false,
  hideNext = false,
}) {
  return (
    <div className="flex items-center justify-between gap-3 pt-1">

      <button
        onClick={onBack}
        className="btn-secondary"
      >
        ← Back
      </button>

      {!hideNext && (
        <button
          onClick={onNext}
          disabled={nextDisabled}
          className="btn-primary"
        >
          {nextLabel}
        </button>
      )}

    </div>
  );
}


/* ============================================================================
   MINI STAT
============================================================================ */

function MiniStat({
  label,
  value,
}) {
  return (
    <div className="card p-4">

      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p
        className="font-bold mt-1 truncate"
        title={String(value)}
      >
        {value}
      </p>

    </div>
  );
}


/* ============================================================================
   FLOW ITEM
============================================================================ */

function FlowItem({
  number,
  title,
  text,
}) {
  return (
    <div className="rounded-xl border border-border bg-panel p-3">

      <span className="text-xs font-bold text-primary-600">
        {number}
      </span>

      <p className="font-semibold mt-1">
        {title}
      </p>

      <p className="text-xs text-slate-500 mt-1">
        {text}
      </p>

    </div>
  );
}


/* ============================================================================
   SUMMARY ROW
============================================================================ */

function SummaryRow({
  label,
  value,
}) {
  return (
    <div className="rounded-xl border border-border p-3 bg-panel/50">

      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p className="text-sm font-semibold mt-1 break-words">
        {String(value ?? "—")}
      </p>

    </div>
  );
}


/* ============================================================================
   FIELD
============================================================================ */

function Field({
  label,
  children,
}) {
  return (
    <div>

      <label className="text-xs font-medium text-slate-500">
        {label}
      </label>

      <div className="mt-1">
        {children}
      </div>

    </div>
  );
}