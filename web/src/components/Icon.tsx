// Line icons for the menu, drawn on a 24px grid. Kept here rather than pulled
// from a library: the menu needs a dozen, and each is a path or two.

const PATHS: Record<string, string> = {
  home: "M4 11 12 4l8 7M6 9.5V20h12V9.5",
  build: "M4 20h4L19 9l-4-4L4 16v4ZM13.5 6.5l4 4",
  results: "M4 20V4M4 20h16M8 16l4-5 3 3 5-6",
  strategies: "M5 4h14v16l-7-4-7 4V4Z",
  paper: "M7 4v4M7 16v4M5 8h4v8H5zM17 3v5M17 15v6M15 8h4v7h-4z",
  orders: "M12 7v5l3 2M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z",
  activity: "M3 12h4l3-7 4 14 3-7h4",
  safety: "M12 3 5 6v5c0 4.5 3 8.3 7 10 4-1.7 7-5.5 7-10V6l-7-3Z",
  bell: "M6 16V11a6 6 0 1 1 12 0v5l2 2H4l2-2ZM10 20a2 2 0 0 0 4 0",
  running: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM10 8.5v7l6-3.5-6-3.5Z",
  sun: "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4",
  moon: "M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z",
  logout: "M15 4h4v16h-4M10 8l-4 4 4 4M6 12h10",
  more: "M5 12h.01M12 12h.01M19 12h.01",
  close: "M6 6l12 12M18 6 6 18",
};

export type IconName = keyof typeof PATHS;

export function Icon({ name }: { name: IconName }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={name === "more" ? 3 : 1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}

/** The wordmark's glyph: two candles, the second one marked. */
export function Glyph() {
  return (
    <svg className="wordmark-glyph" viewBox="0 0 26 26" aria-hidden="true">
      <rect x="1" y="1" width="24" height="24" rx="7" fill="currentColor" />
      <path d="M9 6v14M16.5 5v15" stroke="#1e2550" strokeWidth="1.6" strokeLinecap="round" />
      <rect x="6.5" y="9" width="5" height="7" rx="1.2" fill="#1e2550" />
      <rect
        x="14"
        y="7.5"
        width="5"
        height="8"
        rx="1.2"
        fill="#fff"
        stroke="#1e2550"
        strokeWidth="1.6"
      />
    </svg>
  );
}
