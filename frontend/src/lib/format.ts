/** 83.4 -> "1:23" */
export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function songTitle(song: { title: string | null }): string {
  return song.title?.trim() || "Untitled";
}

export function songArtist(song: { artist: string | null }): string {
  return song.artist?.trim() || "Unknown artist";
}

/** Human label for a 0..1 confidence score. */
export function confidenceLabel(c: number): string {
  if (c >= 0.8) return "Very high";
  if (c >= 0.55) return "High";
  if (c >= 0.35) return "Good";
  return "Fair";
}
