import { Play, RotateCcw, SearchX } from "lucide-react";
import type { IdentifyResult, Song } from "../lib/api";
import { coverSpec } from "../lib/cover";
import { confidenceLabel, formatTime, songArtist, songTitle } from "../lib/format";
import { usePlayer } from "../player/PlayerContext";
import { CoverArt } from "./CoverArt";
import { Skeleton } from "./Skeleton";

/** Signal-strength style meter: 5 bars + label. */
export function ConfidenceMeter({ value }: { value: number }) {
  const filled = Math.max(1, Math.round(value * 5));
  return (
    <div className="flex items-center gap-2" role="meter" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)} aria-label="Match confidence">
      <span className="flex h-4 items-end gap-0.5" aria-hidden="true">
        {[1, 2, 3, 4, 5].map((i) => (
          <span key={i} className={`w-1 rounded-sm ${i <= filled ? "bg-accent" : "bg-line"}`} style={{ height: `${30 + i * 14}%` }} />
        ))}
      </span>
      <span className="text-sm font-semibold">{confidenceLabel(value)}</span>
      <span className="text-sm text-muted tabular-nums">{Math.round(value * 100)}%</span>
    </div>
  );
}

interface MatchProps {
  song: Song;
  result: IdentifyResult;
  onReset: () => void;
}

/** Hero card for a successful match, tinted with the song's cover palette. */
export function ResultCard({ song, result, onReset }: MatchProps) {
  const player = usePlayer();
  const offset = result.offset_s ?? 0;
  // Display rounds to the nearest second (3.98 s reads "0:04"); playback uses the exact offset.
  const offsetLabel = formatTime(Math.round(offset));
  const tint = coverSpec(song.id).colors[1];

  return (
    <section
      aria-live="polite"
      className="animate-fade-up overflow-hidden rounded-2xl border border-line/60 bg-surface"
      style={{ backgroundImage: `linear-gradient(160deg, ${tint}40 0%, transparent 65%)` }}
    >
      <div className="flex flex-col items-center gap-6 p-6 text-center md:flex-row md:items-end md:gap-8 md:p-8 md:text-left">
        <CoverArt songId={song.id} className="size-44 shrink-0 rounded-xl shadow-2xl shadow-black/60 md:size-56" />
        <div className="min-w-0 flex-1">
          <p className="text-xs font-bold tracking-[0.2em] text-accent uppercase">Match found</p>
          <h2 className="mt-2 text-4xl leading-[0.95] font-black tracking-tight text-balance md:text-6xl">{songTitle(song)}</h2>
          <p className="mt-3 truncate text-lg text-muted">
            <span className="font-semibold text-fg">{songArtist(song)}</span>
            {song.album && <span> · {song.album}</span>}
          </p>
          <div className="mt-5 flex flex-wrap items-center justify-center gap-x-5 gap-y-3 md:justify-start">
            <span className="rounded-full bg-elevated px-3 py-1 text-sm font-semibold tabular-nums">
              Matched at {offsetLabel}
            </span>
            <ConfidenceMeter value={result.confidence} />
          </div>
          <div className="mt-6 flex flex-wrap justify-center gap-3 md:justify-start">
            <button
              type="button"
              onClick={() => player.play(song, offset)}
              className="inline-flex items-center gap-2 rounded-full bg-accent px-6 py-3 font-bold text-accent-fg transition duration-150 hover:scale-[1.03] hover:bg-accent-hover active:scale-[0.97]"
            >
              <Play className="size-5 fill-current" /> Play from {offsetLabel}
            </button>
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-2 rounded-full border border-line px-5 py-3 font-semibold text-fg transition-colors duration-150 hover:border-muted"
            >
              <RotateCcw className="size-4" /> Identify another
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

export function NoMatchCard({ result, onReset }: { result: IdentifyResult; onReset: () => void }) {
  return (
    <section aria-live="polite" className="animate-fade-up rounded-2xl border border-line/60 bg-surface p-6 text-center md:p-8">
      <SearchX className="mx-auto size-10 text-subtle" aria-hidden="true" />
      <h2 className="mt-3 text-2xl font-black tracking-tight">No match found</h2>
      <p className="mx-auto mt-2 max-w-md text-muted">
        That clip isn't in the library, or it was too quiet or noisy to be sure. Try 8–10 seconds, closer to the speaker.
      </p>
      <p className="mt-2 text-xs text-subtle tabular-nums">
        Best candidate: {result.aligned_matches} aligned hashes, {Math.round(result.confidence * 100)}% confidence
      </p>
      <button
        type="button"
        onClick={onReset}
        className="mt-5 inline-flex items-center gap-2 rounded-full bg-fg px-6 py-3 font-bold text-bg transition duration-150 hover:scale-[1.03] active:scale-[0.97]"
      >
        <RotateCcw className="size-4" /> Try again
      </button>
    </section>
  );
}

export function ResultCardSkeleton() {
  return (
    <div className="flex flex-col items-center gap-6 rounded-2xl bg-surface p-6 md:flex-row md:items-end md:p-8" aria-hidden="true">
      <Skeleton className="size-44 rounded-xl md:size-56" />
      <div className="w-full flex-1">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="mt-3 h-12 w-3/4" />
        <Skeleton className="mt-3 h-5 w-1/2" />
        <Skeleton className="mt-6 h-11 w-48 rounded-full" />
      </div>
    </div>
  );
}
