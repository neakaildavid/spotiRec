import { Pause, Play, RotateCcw, RotateCw, Sparkles, Volume1, Volume2, VolumeX } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { formatTime, songArtist, songTitle } from "../lib/format";
import { usePlayer, usePlayerTime } from "../player/PlayerContext";
import { CoverArt } from "./CoverArt";

/**
 * Persistent "now playing" bar. Space toggles playback anywhere outside a
 * text field; the scrub bar and volume are native range inputs (keyboard and
 * screen-reader accessible) restyled in index.css.
 */
export function PlayerBar() {
  const player = usePlayer();
  const { song, playing, buffering, toggle } = player;
  const navigate = useNavigate();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (e.code !== "Space" || el.closest("input, textarea, select, button, [contenteditable]")) return;
      e.preventDefault();
      toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);

  return (
    <footer className="relative z-20 border-t border-line bg-surface/95 backdrop-blur" aria-label="Player">
      {/* Mobile: thin progress bar along the top edge */}
      <div className="absolute inset-x-0 -top-px md:hidden">
        <ProgressLine />
      </div>
      <div className="grid h-16 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 md:h-20 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)_minmax(0,1fr)] md:px-4">
        {/* Now playing */}
        <div className="flex min-w-0 items-center gap-3">
          {song ? (
            <>
              <CoverArt songId={song.id} className="size-11 shrink-0 rounded-md shadow-md shadow-black/40 md:size-14" />
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold">{songTitle(song)}</p>
                <p className="truncate text-xs text-muted">{songArtist(song)}</p>
              </div>
              <button
                type="button"
                onClick={() => navigate(`/discover?song=${song.id}`)}
                aria-label="More like this"
                title="More like this"
                className="hidden size-8 shrink-0 place-items-center rounded-full text-muted transition-colors hover:text-accent md:grid"
              >
                <Sparkles className="size-4" />
              </button>
            </>
          ) : (
            <p className="truncate text-sm text-subtle">Nothing playing. Identify a song or pick one from the library.</p>
          )}
        </div>

        {/* Transport + scrub */}
        <div className="flex flex-col items-center gap-1">
          <div className="flex items-center gap-2 md:gap-4">
            <SkipButton seconds={-10} />
            <button
              type="button"
              onClick={toggle}
              disabled={!song}
              aria-label={playing ? "Pause" : "Play"}
              className="grid size-10 place-items-center rounded-full bg-fg text-bg transition duration-150 hover:scale-105 active:scale-95 disabled:opacity-30 disabled:hover:scale-100 md:size-9"
            >
              {buffering && playing ? (
                <span className="size-4 rounded-full border-2 border-bg/30 border-t-bg motion-safe:animate-spin" aria-hidden="true" />
              ) : playing ? (
                <Pause className="size-4 fill-current" />
              ) : (
                <Play className="ml-0.5 size-4 fill-current" />
              )}
            </button>
            <SkipButton seconds={10} />
          </div>
          <div className="hidden w-full max-w-xl md:block">
            <Scrubber />
          </div>
        </div>

        {/* Volume */}
        <div className="hidden justify-end md:flex">
          <VolumeControl />
        </div>
      </div>
    </footer>
  );
}

function SkipButton({ seconds }: { seconds: number }) {
  const { song, seek } = usePlayer();
  const { currentTime, duration } = usePlayerTime();
  const Icon = seconds < 0 ? RotateCcw : RotateCw;
  return (
    <button
      type="button"
      disabled={!song}
      onClick={() => seek(Math.max(0, Math.min(duration || Infinity, currentTime + seconds)))}
      aria-label={seconds < 0 ? "Back 10 seconds" : "Forward 10 seconds"}
      className="hidden size-8 place-items-center rounded-full text-muted transition-colors hover:text-fg disabled:opacity-30 md:grid"
    >
      <Icon className="size-4" />
    </button>
  );
}

function Scrubber() {
  const { song, seek } = usePlayer();
  const { currentTime, duration } = usePlayerTime();
  // While dragging, show the drag position instead of the (lagging) audio time.
  const [dragValue, setDragValue] = useState<number | null>(null);
  const value = dragValue ?? currentTime;
  const max = duration || song?.duration_s || 0;
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="flex items-center gap-2 text-[11px] text-muted tabular-nums">
      <span className="w-9 text-right">{formatTime(value)}</span>
      <input
        type="range"
        className="range"
        min={0}
        max={max || 1}
        step={0.1}
        value={value}
        disabled={!song}
        aria-label="Seek"
        aria-valuetext={`${formatTime(value)} of ${formatTime(max)}`}
        style={{ "--progress": `${pct}%` } as React.CSSProperties}
        onChange={(e) => setDragValue(Number(e.target.value))}
        onPointerUp={() => { if (dragValue !== null) seek(dragValue); setDragValue(null); }}
        onKeyUp={() => { if (dragValue !== null) seek(dragValue); setDragValue(null); }}
      />
      <span className="w-9">{formatTime(max)}</span>
    </div>
  );
}

function ProgressLine() {
  const { song } = usePlayer();
  const { currentTime, duration } = usePlayerTime();
  const max = duration || song?.duration_s || 0;
  return (
    <div className="h-0.5 w-full bg-line" aria-hidden="true">
      <div className="h-full bg-accent" style={{ width: `${max ? (currentTime / max) * 100 : 0}%` }} />
    </div>
  );
}

function VolumeControl() {
  const { volume, setVolume } = usePlayer();
  const lastNonZero = useRef(volume || 0.9);
  if (volume > 0) lastNonZero.current = volume;
  const Icon = volume === 0 ? VolumeX : volume < 0.5 ? Volume1 : Volume2;
  return (
    <div className="flex w-40 items-center gap-2">
      <button
        type="button"
        onClick={() => setVolume(volume === 0 ? lastNonZero.current : 0)}
        aria-label={volume === 0 ? "Unmute" : "Mute"}
        className="text-muted transition-colors hover:text-fg"
      >
        <Icon className="size-4" />
      </button>
      <input
        type="range"
        className="range"
        min={0}
        max={1}
        step={0.01}
        value={volume}
        aria-label="Volume"
        style={{ "--progress": `${volume * 100}%` } as React.CSSProperties}
        onChange={(e) => setVolume(Number(e.target.value))}
      />
    </div>
  );
}
