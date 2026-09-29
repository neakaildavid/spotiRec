import { Pause, Play } from "lucide-react";
import { memo } from "react";
import type { Song } from "../lib/api";
import { songArtist, songTitle } from "../lib/format";
import { usePlayer } from "../player/PlayerContext";
import { CoverArt } from "./CoverArt";

interface Props {
  song: Song;
  /** Optional line under the artist (e.g. "92% similar" in Phase 2's Discover row). */
  caption?: string;
}

/**
 * Square cover card. The whole card is one button (play/pause) so there's a
 * single, large tap target and no nested interactive elements; the round play
 * badge is decorative and appears on hover/focus (always on touch screens).
 */
export const SongCard = memo(function SongCard({ song, caption }: Props) {
  const player = usePlayer();
  const isCurrent = player.song?.id === song.id;
  const isPlaying = isCurrent && player.playing;
  const title = songTitle(song);

  return (
    <button
      type="button"
      onClick={() => (isCurrent ? player.toggle() : player.play(song))}
      aria-label={`${isPlaying ? "Pause" : "Play"} ${title} by ${songArtist(song)}`}
      className="group w-full rounded-xl bg-surface p-3 text-left transition duration-200 ease-out-soft hover:-translate-y-1 hover:bg-elevated hover:shadow-xl hover:shadow-black/40 focus-visible:-translate-y-1 focus-visible:bg-elevated motion-reduce:hover:translate-y-0"
    >
      <div className="relative aspect-square overflow-hidden rounded-lg shadow-lg shadow-black/40">
        <CoverArt songId={song.id} className="size-full" />
        <span
          className={`absolute right-2 bottom-2 grid size-11 place-items-center rounded-full bg-accent text-accent-fg shadow-lg shadow-black/50 transition duration-200 ease-out-soft group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100 pointer-coarse:translate-y-0 pointer-coarse:opacity-100 ${
            isCurrent ? "translate-y-0 opacity-100" : "translate-y-2 opacity-0"
          }`}
          aria-hidden="true"
        >
          {isPlaying ? <Pause className="size-5 fill-current" /> : <Play className="ml-0.5 size-5 fill-current" />}
        </span>
      </div>
      <h3 className={`mt-3 truncate text-sm font-semibold ${isCurrent ? "text-accent" : "text-fg"}`}>{title}</h3>
      <p className="truncate text-sm text-muted">{songArtist(song)}</p>
      {caption && <p className="mt-0.5 truncate text-xs text-subtle">{caption}</p>}
    </button>
  );
});
