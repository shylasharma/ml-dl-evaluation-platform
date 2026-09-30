import React, { useState } from "react";
import api from "../api/client.js";

const DEFAULTS = {
  primary_goal: "balanced_performance",
  model_family: "both",
  interpretability: "medium",
  training_speed: "balanced",
  probability_outputs: "useful",
};

export default function ModelRecommendationPanel({
  dataset,
  selectedModels = [],
  onApplyRecommendations,
}) {
  const [criteria, setCriteria] = useState(DEFAULTS);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const update = (key, value) => {
    setCriteria((current) => ({
      ...current,
      [key]: value,
    }));
  };

  const recommend = async () => {
    if (!dataset) {
      setError("No dataset information is available.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = {
        rows:
          dataset?.rows ??
          dataset?.row_count ??
          dataset?.n_rows ??
          0,

        features:
          dataset?.n_features ??
          dataset?.features ??
          dataset?.feature_count ??
          0,

        classes:
          dataset?.n_classes ??
          dataset?.classes ??
          dataset?.class_count ??
          2,

        imbalance_ratio:
          dataset?.imbalance_ratio ??
          dataset?.imbalanceRatio ??
          1,

        ...criteria,
      };

      console.log(
        "Sending model recommendation request:",
        payload
      );

      const data = await api.recommendModels(payload);

      console.log(
        "Model recommendation response:",
        data
      );

      setResult(data);
    } catch (err) {
      console.error(
        "Model recommendation error:",
        err
      );

      setError(
        err?.message ||
          "Could not connect to the model recommendation service."
      );
    } finally {
      setLoading(false);
    }
  };

  const apply = () => {
    if (
      !result?.recommendations?.length ||
      !onApplyRecommendations
    ) {
      return;
    }

    const recommendedKeys =
      result.recommendations.map(
        (item) => item.key
      );

    onApplyRecommendations(
      recommendedKeys
    );
  };

  return (
    <div className="card p-5 space-y-5 border-primary-100 bg-primary-50/20">

      {/* ---------------------------------------------------------------- */}
      {/* Header                                                           */}
      {/* ---------------------------------------------------------------- */}

      <div>
        <div className="flex items-start gap-3">

          <div className="w-10 h-10 rounded-xl bg-primary-100 text-primary-700 grid place-items-center text-lg">
            🧠
          </div>

          <div>
            <p className="font-semibold text-lg">
              Model Recommendation
            </p>

            <p className="text-sm text-slate-500 mt-1 max-w-3xl">
              Answer a few questions and the platform
              will create a transparent, rule-based
              shortlist. The recommendation does not
              assume that any model will be the best
              before it is experimentally evaluated.
            </p>
          </div>

        </div>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Questions                                                        */}
      {/* ---------------------------------------------------------------- */}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        <Field label="What matters most?">
          <select
            value={criteria.primary_goal}
            onChange={(e) =>
              update(
                "primary_goal",
                e.target.value
              )
            }
            className="input"
          >
            <option value="balanced_performance">
              Balanced performance
            </option>

            <option value="minority_recall">
              Minority-class detection
            </option>

            <option value="interpretability">
              Interpretability
            </option>

            <option value="speed">
              Training speed
            </option>

            <option value="nonlinear_patterns">
              Nonlinear patterns
            </option>
          </select>
        </Field>

        <Field label="Model family">
          <select
            value={criteria.model_family}
            onChange={(e) =>
              update(
                "model_family",
                e.target.value
              )
            }
            className="input"
          >
            <option value="both">
              ML + DL
            </option>

            <option value="ml">
              Machine Learning only
            </option>

            <option value="dl">
              Deep Learning only
            </option>
          </select>
        </Field>

        <Field label="Interpretability priority">
          <select
            value={criteria.interpretability}
            onChange={(e) =>
              update(
                "interpretability",
                e.target.value
              )
            }
            className="input"
          >
            <option value="low">
              Low
            </option>

            <option value="medium">
              Medium
            </option>

            <option value="high">
              High
            </option>
          </select>
        </Field>

        <Field label="Training-speed priority">
          <select
            value={criteria.training_speed}
            onChange={(e) =>
              update(
                "training_speed",
                e.target.value
              )
            }
            className="input"
          >
            <option value="fast">
              Fast
            </option>

            <option value="balanced">
              Balanced
            </option>

            <option value="flexible">
              Flexible / research comparison
            </option>
          </select>
        </Field>

        <Field label="Probability outputs">
          <select
            value={
              criteria.probability_outputs
            }
            onChange={(e) =>
              update(
                "probability_outputs",
                e.target.value
              )
            }
            className="input"
          >
            <option value="important">
              Important
            </option>

            <option value="useful">
              Useful
            </option>

            <option value="not_required">
              Not required
            </option>
          </select>
        </Field>

      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Dataset summary                                                  */}
      {/* ---------------------------------------------------------------- */}

      <div className="rounded-xl border border-border bg-white p-4">

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">

          <Stat
            label="Rows"
            value={
              dataset?.rows ??
              dataset?.row_count ??
              dataset?.n_rows ??
              "—"
            }
          />

          <Stat
            label="Features"
            value={
              dataset?.n_features ??
              dataset?.features ??
              "—"
            }
          />

          <Stat
            label="Classes"
            value={
              dataset?.n_classes ??
              dataset?.classes ??
              "—"
            }
          />

          <Stat
            label="Imbalance"
            value={
              dataset?.imbalance_ratio
                ? `${dataset.imbalance_ratio}:1`
                : "—"
            }
          />

        </div>

      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Generate button                                                  */}
      {/* ---------------------------------------------------------------- */}

      <button
        type="button"
        onClick={recommend}
        disabled={loading}
        className="btn-primary w-full"
      >
        {loading
          ? "Generating Recommendations…"
          : "🧠 Generate Model Recommendations"}
      </button>

      {/* ---------------------------------------------------------------- */}
      {/* Error                                                            */}
      {/* ---------------------------------------------------------------- */}

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">

          <p className="font-semibold">
            Recommendation request failed
          </p>

          <p className="mt-1">
            {error}
          </p>

          <p className="mt-2 text-xs text-red-600">
            Check that the FastAPI backend is running
            on port 8000 and reload the page.
          </p>

        </div>
      )}

      {/* ---------------------------------------------------------------- */}
      {/* Results                                                          */}
      {/* ---------------------------------------------------------------- */}

      {result && (
        <div className="space-y-4">

          <div className="rounded-xl border border-primary-100 bg-primary-50 p-4">

            <p className="font-semibold text-primary-900">
              Transparent recommendation engine
            </p>

            <p className="text-sm text-primary-800 mt-1">
              {result.disclaimer}
            </p>

          </div>

          <div className="space-y-3">

            {result.recommendations?.map(
              (item) => (
                <div
                  key={item.key}
                  className="rounded-xl border border-border bg-white p-4"
                >

                  <div className="flex items-start justify-between gap-3">

                    <div>

                      <p className="font-semibold">
                        {item.label}
                      </p>

                      <p className="text-xs text-slate-500 mt-0.5">
                        {item.family}
                        {" · "}
                        recommendation score{" "}
                        {item.score}
                      </p>

                    </div>

                    <span className="chip">
                      Candidate
                    </span>

                  </div>

                  <ul className="mt-2 space-y-1 text-sm text-slate-600">

                    {item.reasons?.map(
                      (reason, index) => (
                        <li key={index}>
                          • {reason}
                        </li>
                      )
                    )}

                  </ul>

                </div>
              )
            )}

          </div>

          {result.caveats?.length > 0 && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">

              <p className="font-semibold text-amber-900">
                Research considerations
              </p>

              <ul className="mt-2 space-y-1 text-sm text-amber-800">

                {result.caveats.map(
                  (item, index) => (
                    <li key={index}>
                      • {item}
                    </li>
                  )
                )}

              </ul>

            </div>
          )}

          {onApplyRecommendations &&
            result.recommendations?.length > 0 && (
              <button
                type="button"
                onClick={apply}
                className="btn-secondary w-full"
              >
                Apply Recommended Models to Selection
              </button>
            )}

        </div>
      )}

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


/* ============================================================================
   STAT
============================================================================ */

function Stat({
  label,
  value,
}) {
  return (
    <div>

      <p className="text-xs text-slate-500">
        {label}
      </p>

      <p className="font-semibold mt-1">
        {value}
      </p>

    </div>
  );
}