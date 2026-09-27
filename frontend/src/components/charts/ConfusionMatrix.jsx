import React from "react";

export default function ConfusionMatrix({ matrix, labels }) {
  if (!matrix || matrix.length === 0) return null;
  const max = Math.max(...matrix.flat());
  const classLabels = labels && labels.length === matrix.length
    ? labels
    : matrix.map((_, i) => `Class ${i}`);

  return (
    <div className="overflow-x-auto scroll-thin">
      <table className="border-collapse text-sm">
        <thead>
          <tr>
            <th className="p-2"></th>
            <th className="p-2 text-xs font-medium text-slate-500" colSpan={matrix.length}>
              Predicted
            </th>
          </tr>
          <tr>
            <th className="p-2"></th>
            {classLabels.map((l) => (
              <th key={l} className="p-2 text-xs font-medium text-slate-500">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              {i === 0 && (
                <th
                  rowSpan={matrix.length}
                  className="p-2 text-xs font-medium text-slate-500 align-middle"
                  style={{ writingMode: "vertical-rl" }}
                >
                  Actual
                </th>
              )}
              <th className="p-2 text-xs font-medium text-slate-500 text-left pr-3">{classLabels[i]}</th>
              {row.map((val, j) => {
                const intensity = max > 0 ? val / max : 0;
                const isDiag = i === j;
                return (
                  <td
                    key={j}
                    className="p-4 text-center font-semibold rounded-md"
                    style={{
                      background: isDiag
                        ? `rgba(79,70,229,${0.15 + intensity * 0.55})`
                        : `rgba(225,29,72,${0.06 + intensity * 0.35})`,
                      color: intensity > 0.55 ? "#fff" : "#1e293b",
                      minWidth: 56,
                    }}
                  >
                    {val}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
