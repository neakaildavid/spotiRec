interface Props {
  label: string;
  active?: boolean;
  count?: number;
  onClick: () => void;
}

/** Pill-shaped filter chip; the active one uses the accent (used sparingly elsewhere). */
export function GenreChip({ label, active = false, count, onClick }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`shrink-0 rounded-full px-4 py-1.5 text-sm font-semibold transition-colors duration-150 ${
        active ? "bg-accent text-accent-fg" : "bg-elevated text-fg hover:bg-hover"
      }`}
    >
      {label}
      {count !== undefined && <span className={`ml-1.5 tabular-nums ${active ? "text-accent-fg/70" : "text-subtle"}`}>{count}</span>}
    </button>
  );
}
