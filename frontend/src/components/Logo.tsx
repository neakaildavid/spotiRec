/**
 * Constellation mark: five stars joined like a star chart, with the
 * brightest one in the accent color. It doubles as a nod to the algorithm:
 * spectrogram peaks linked into pairs.
 */
export function LogoMark({ className = "size-8", mono = false }: { className?: string; mono?: boolean }) {
  // `mono` renders everything in currentColor, for use on an accent background.
  const accent = mono ? "currentColor" : "var(--color-accent)";
  return (
    <svg viewBox="0 0 32 32" fill="none" className={className} aria-hidden="true">
      <g stroke={accent} strokeOpacity={mono ? ".75" : ".6"} strokeWidth="1.3" strokeLinecap="round">
        <path d="M5 23 L11.5 14.5 L19 17.5 L26 7.5" />
        <path d="M19 17.5 L23.5 25" />
      </g>
      <circle cx="5" cy="23" r="1.8" fill="currentColor" />
      <circle cx="11.5" cy="14.5" r="2" fill="currentColor" />
      <circle cx="23.5" cy="25" r="1.6" fill="currentColor" />
      <circle cx="26" cy="7.5" r="1.8" fill="currentColor" />
      <circle cx="19" cy="17.5" r="5.5" fill={accent} opacity={mono ? ".25" : ".18"} />
      <circle cx="19" cy="17.5" r="3.2" fill={accent} />
    </svg>
  );
}

export function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark className="size-8 text-fg" />
      <span className="text-lg font-black tracking-tight">Constellation</span>
    </span>
  );
}
