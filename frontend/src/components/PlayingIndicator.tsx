/** Tiny animated equalizer shown next to the song that's currently playing. */
export function PlayingIndicator({ active = true }: { active?: boolean }) {
  return (
    <span className="inline-flex h-3.5 items-end gap-[2px]" aria-label="Now playing" role="img">
      {[0, 200, 400].map((delay) => (
        <span
          key={delay}
          className={`w-[3px] origin-bottom rounded-full bg-accent ${active ? "animate-eq" : "scale-y-50"}`}
          style={{ height: "100%", animationDelay: `${delay}ms` }}
        />
      ))}
    </span>
  );
}
