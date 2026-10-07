import type { ReactNode } from "react";

// Status is never colour alone: each kind carries its own symbol, and the
// first line says in words what kind of message it is.
const ICONS = { critical: "!", warning: "!", info: "i", good: "✓" } as const;

export type Tone = keyof typeof ICONS;

export function Callout({
  tone = "info",
  title,
  children,
}: {
  tone?: Tone;
  title?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className={`callout ${tone}`} role={tone === "critical" ? "alert" : undefined}>
      <span className="icon" aria-hidden="true">
        {ICONS[tone]}
      </span>
      <div>
        {title && <strong>{title}</strong>}
        {children}
      </div>
    </div>
  );
}

export function Working({ children }: { children: ReactNode }) {
  return (
    <span className="working" role="status">
      <span className="spinner" aria-hidden="true" />
      {children}
    </span>
  );
}

export function ErrorLine({ error }: { error: string | null }) {
  if (!error) return null;
  return <Callout tone="critical">{error}</Callout>;
}
