import { ChevronDown, ChevronUp, Clock3, Pause, Play } from "lucide-react";
import { memo } from "react";
import type { Song, SortField, SortOrder } from "../lib/api";
import { formatTime, songArtist, songTitle } from "../lib/format";
import { usePlayer } from "../player/PlayerContext";
import { CoverArt } from "./CoverArt";
import { PlayingIndicator } from "./PlayingIndicator";

/** Shared column template so the header and rows always line up. */
export const TRACK_GRID =
  "grid grid-cols-[2rem_minmax(0,1fr)_3.5rem] md:grid-cols-[2.5rem_minmax(0,1.3fr)_minmax(0,1fr)_4rem] items-center gap-4";

interface RowProps {
  song: Song;
  index: number;
}

export const TrackRow = memo(function TrackRow({ song, index }: RowProps) {
  const player = usePlayer();
  const isCurrent = player.song?.id === song.id;
  const isPlaying = isCurrent && player.playing;
  const title = songTitle(song);
  const artist = songArtist(song);

  return (
    <button
      type="button"
      onClick={() => (isCurrent ? player.toggle() : player.play(song))}
      aria-label={`${isPlaying ? "Pause" : "Play"} ${title} by ${artist}`}
      className={`group ${TRACK_GRID} w-full rounded-md px-3 py-2 text-left transition-colors duration-150 hover:bg-hover focus-visible:bg-hover ${isCurrent ? "bg-elevated" : ""}`}
    >
      <span className="relative flex justify-end text-sm text-muted tabular-nums">
        <span className="group-hover:invisible group-focus-visible:invisible">
          {isPlaying ? <PlayingIndicator /> : index + 1}
        </span>
        <span className="invisible absolute inset-0 flex items-center justify-end text-fg group-hover:visible group-focus-visible:visible" aria-hidden="true">
          {isPlaying ? <Pause className="size-4 fill-current" /> : <Play className="size-4 fill-current" />}
        </span>
      </span>
      <span className="flex min-w-0 items-center gap-3">
        <CoverArt songId={song.id} className="size-10 shrink-0 rounded" />
        <span className="min-w-0">
          <span className={`block truncate font-medium ${isCurrent ? "text-accent" : "text-fg"}`}>{title}</span>
          <span className="block truncate text-sm text-muted md:hidden">{artist}</span>
        </span>
      </span>
      <span className="hidden truncate text-sm text-muted md:block">{artist}</span>
      <span className="text-right text-sm text-muted tabular-nums">{formatTime(song.duration_s)}</span>
    </button>
  );
});

interface HeaderProps {
  sort: SortField;
  order: SortOrder;
  onSort: (field: SortField) => void;
}

/** Column headers; clicking one sorts by it, clicking again flips the order. */
export function TrackListHeader({ sort, order, onSort }: HeaderProps) {
  const col = (field: SortField, label: React.ReactNode, className = "", ariaLabel?: string) => {
    const active = sort === field;
    const Arrow = order === "asc" ? ChevronUp : ChevronDown;
    return (
      <button
        type="button"
        onClick={() => onSort(field)}
        aria-label={ariaLabel}
        aria-sort={active ? (order === "asc" ? "ascending" : "descending") : "none"}
        className={`flex items-center gap-1 text-xs font-semibold tracking-wider uppercase transition-colors hover:text-fg ${active ? "text-fg" : "text-subtle"} ${className}`}
      >
        {label}
        {active && <Arrow className="size-3.5 text-accent" aria-hidden="true" />}
      </button>
    );
  };
  return (
    <div role="row" className={`${TRACK_GRID} border-b border-line px-3 pb-2`}>
      {col("id", "#", "justify-end", "Sort by date added")}
      {col("title", "Title")}
      {col("artist", "Artist", "hidden md:flex")}
      {col("duration_s", <Clock3 className="size-4" aria-hidden="true" />, "justify-end", "Sort by duration")}
    </div>
  );
}
