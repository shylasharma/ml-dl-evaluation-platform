/* =====================================================
   PCA / DIMENSIONALITY REDUCTION PANEL
===================================================== */

import React from "react";

const DEFAULT_PCA = {
  enabled: false,
  mode: "variance",
  variance: 0.95,
  n_components: null,
};

function PCAPanel({ value, onChange, featureCount }) {
  const pca = {
    ...DEFAULT_PCA,
    ...(value || {}),
  };

  /*
   * Important:
   * The parent component must provide a function here.
   * This guard prevents a confusing "onChange is not a function"
   * crash and makes the actual integration problem explicit.
   */
  const update = (changes) => {
    if (typeof onChange !== "function") {
      console.error(
        "PCAPanel: expected onChange to be a function, but received:",
        onChange
      );
      return;
    }

    onChange({
      ...pca,
      ...changes,
    });
  };

  const handleModeChange = (mode) => {
    if (mode === "variance") {
      update({
        mode: "variance",
        n_components: null,
      });
      return;
    }

    const maxComponents = Math.max(
      1,
      Number(featureCount) || 1
    );

    update({
      mode: "components",
      n_components: Math.min(2, maxComponents),
    });
  };

  const handleComponentChange = (event) => {
    const rawValue = event.target.value;

    if (rawValue === "") {
      update({
        n_components: null,
      });
      return;
    }

    const number = Number(rawValue);

    update({
      n_components:
        Number.isFinite(number) && number >= 1
          ? number
          : null,
    });
  };

  const handleEnable = () => {
    update({
      enabled: true,
      mode: "variance",
      variance: 0.95,
      n_components: null,
    });
  };

  const handleDisable = () => {
    update({
      enabled: false,
      mode: "variance",
      variance: 0.95,
      n_components: null,
    });
  };

  const handleVarianceChange = (variance) => {
    update({
      enabled: true,
      mode: "variance",
      variance,
      n_components: null,
    });
  };

  const componentError =
    pca.enabled &&
    pca.mode === "components" &&
    (
      !pca.n_components ||
      pca.n_components < 1 ||
      (
        featureCount &&
        pca.n_components > featureCount
      )
    );

  return (
    <div className="card p-5 space-y-5">

      {/* Header */}
      <div className="flex items-start gap-3">

        <div className="w-10 h-10 rounded-xl bg-violet-100 text-violet-700 grid place-items-center text-lg">
          📉
        </div>

        <div>
          <h3 className="font-semibold text-slate-900">
            Dimensionality Reduction — PCA
          </h3>

          <p className="text-sm text-slate-600 mt-1 max-w-3xl">
            PCA reduces the number of input features while
            preserving as much information as possible.
            It is applied after preprocessing and feature
            selection and fitted only on training data to
            prevent data leakage.
          </p>
        </div>

      </div>

      {/* Enable PCA */}
      <div className="border rounded-xl p-4 bg-slate-50">

        <p className="font-medium text-slate-800 mb-3">
          Would you like to apply PCA?
        </p>

        <div className="flex gap-3">

          <button
            type="button"
            onClick={handleDisable}
            className={`px-5 py-2 rounded-lg border transition ${
              !pca.enabled
                ? "bg-slate-800 text-white border-slate-800"
                : "bg-white text-slate-700 border-slate-300 hover:bg-slate-100"
            }`}
          >
            No
          </button>

          <button
            type="button"
            onClick={handleEnable}
            className={`px-5 py-2 rounded-lg border transition ${
              pca.enabled
                ? "bg-violet-600 text-white border-violet-600"
                : "bg-white text-slate-700 border-slate-300 hover:bg-slate-100"
            }`}
          >
            Yes
          </button>

        </div>

      </div>

      {/* PCA Configuration */}
      {pca.enabled && (
        <div className="space-y-4">

          {/* Mode */}
          <div>

            <label className="block text-sm font-medium text-slate-700 mb-2">
              PCA configuration
            </label>

            <select
              value={pca.mode}
              onChange={(event) =>
                handleModeChange(event.target.value)
              }
              className="w-full border border-slate-300 rounded-lg px-3 py-2 bg-white"
            >
              <option value="variance">
                Retain target variance
              </option>

              <option value="components">
                Custom number of components
              </option>
            </select>

          </div>

          {/* Variance selection */}
          {pca.mode === "variance" && (
            <div>

              <label className="block text-sm font-medium text-slate-700 mb-2">
                Variance to retain
              </label>

              <div className="grid grid-cols-3 gap-3">

                {[0.95, 0.90, 0.85].map(
                  (variance) => (
                    <button
                      key={variance}
                      type="button"
                      onClick={() =>
                        handleVarianceChange(
                          variance
                        )
                      }
                      className={`px-4 py-3 rounded-lg border text-sm font-medium transition ${
                        pca.variance === variance
                          ? "bg-violet-600 text-white border-violet-600"
                          : "bg-white text-slate-700 border-slate-300 hover:bg-violet-50"
                      }`}
                    >
                      {variance * 100}% variance
                    </button>
                  )
                )}

              </div>

              <p className="text-xs text-slate-500 mt-2">
                PCA will automatically determine the number
                of components required to retain the selected
                amount of variance.
              </p>

            </div>
          )}

          {/* Custom components */}
          {pca.mode === "components" && (
            <div>

              <label className="block text-sm font-medium text-slate-700 mb-2">
                Number of PCA components
              </label>

              <input
                type="number"
                min="1"
                max={featureCount || undefined}
                value={pca.n_components ?? ""}
                onChange={handleComponentChange}
                placeholder="e.g. 10"
                className={`w-full border rounded-lg px-3 py-2 ${
                  componentError
                    ? "border-red-400"
                    : "border-slate-300"
                }`}
              />

              {featureCount && (
                <p className="text-xs text-slate-500 mt-2">
                  Available features:{" "}
                  <strong>{featureCount}</strong>
                </p>
              )}

              {componentError && (
                <p className="text-xs text-red-600 mt-2">
                  PCA components must be between 1 and{" "}
                  {featureCount ||
                    "the available feature count"}.
                </p>
              )}

            </div>
          )}

          {/* Research explanation */}
          <div className="rounded-lg border border-violet-100 bg-violet-50 p-4">

            <p className="text-sm font-medium text-violet-900">
              🔬 Research note
            </p>

            <p className="text-sm text-violet-800 mt-1">
              The experiment records the original feature
              count, PCA component count, and variance
              retained. This allows PCA and no-PCA
              configurations to be compared experimentally
              rather than assuming PCA always improves
              performance.
            </p>

          </div>

        </div>
      )}

    </div>
  );
}

export default PCAPanel;