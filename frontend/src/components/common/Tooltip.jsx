import React, { useState } from "react";

export default function Tooltip({ text, children }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      {children}
      {open && (
        <span className="absolute z-30 bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 rounded-lg bg-ink text-white text-xs px-3 py-2 shadow-card">
          {text}
        </span>
      )}
    </span>
  );
}
