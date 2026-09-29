/**
 * Global audio player: one <audio> element for the whole app.
 *
 * State is split into two contexts on purpose. `PlayerContext` (current song,
 * playing flag, actions) changes rarely; `PlayerTimeContext` (current time)
 * changes every animation frame while playing. Components that only need to
 * know "what's playing" (every SongCard/TrackRow) subscribe to the first and
 * don't re-render 60x per second.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { audioUrl, type Song } from "../lib/api";
import { songArtist, songTitle } from "../lib/format";

interface PlayerApi {
  song: Song | null;
  playing: boolean;
  buffering: boolean;
  volume: number;
  /** Play `song`, optionally from `startAt` seconds (e.g. the matched offset). */
  play: (song: Song, startAt?: number) => void;
  toggle: () => void;
  seek: (seconds: number) => void;
  setVolume: (v: number) => void;
}

interface PlayerTime {
  currentTime: number;
  duration: number;
}

const PlayerContext = createContext<PlayerApi | null>(null);
const PlayerTimeContext = createContext<PlayerTime>({ currentTime: 0, duration: 0 });

export function PlayerProvider({ children }: { children: ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  if (audioRef.current === null && typeof Audio !== "undefined") {
    audioRef.current = new Audio();
    audioRef.current.preload = "metadata";
  }
  const pendingSeek = useRef<number | null>(null);

  const [song, setSong] = useState<Song | null>(null);
  const [playing, setPlaying] = useState(false);
  const [buffering, setBuffering] = useState(false);
  const [volume, setVolumeState] = useState(0.9);
  const [time, setTime] = useState<PlayerTime>({ currentTime: 0, duration: 0 });

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    const sync = () => setTime({ currentTime: a.currentTime, duration: Number.isFinite(a.duration) ? a.duration : 0 });
    const onMeta = () => {
      if (pendingSeek.current !== null) {
        // Seeking before metadata loads is ignored by some browsers, so the
        // matched offset is applied here. The server supports Range requests,
        // so the browser fetches from the offset instead of the whole file.
        a.currentTime = Math.min(pendingSeek.current, Math.max(0, a.duration - 0.5));
        pendingSeek.current = null;
      }
      sync();
    };
    const handlers: [string, () => void][] = [
      ["loadedmetadata", onMeta],
      ["durationchange", sync],
      ["seeked", sync],
      ["play", () => setPlaying(true)],
      ["pause", () => setPlaying(false)],
      ["ended", () => { setPlaying(false); sync(); }],
      ["waiting", () => setBuffering(true)],
      ["playing", () => setBuffering(false)],
      ["canplay", () => setBuffering(false)],
      ["error", () => { setPlaying(false); setBuffering(false); }],
    ];
    handlers.forEach(([e, h]) => a.addEventListener(e, h));
    return () => handlers.forEach(([e, h]) => a.removeEventListener(e, h));
  }, []);

  // Smooth scrub bar: poll currentTime once per frame while playing
  // (the native timeupdate event only fires ~4x per second).
  useEffect(() => {
    const a = audioRef.current;
    if (!a || !playing) return;
    let raf = 0;
    const tick = () => {
      setTime({ currentTime: a.currentTime, duration: Number.isFinite(a.duration) ? a.duration : 0 });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const play = useCallback((next: Song, startAt = 0) => {
    const a = audioRef.current;
    if (!a) return;
    if (song?.id === next.id && a.readyState >= 1) {
      a.currentTime = startAt;
    } else {
      pendingSeek.current = startAt > 0 ? startAt : null;
      a.src = audioUrl(next);
      setSong(next);
      setTime({ currentTime: startAt, duration: next.duration_s });
      setBuffering(true);
    }
    a.play().catch(() => setPlaying(false)); // autoplay can be refused; the UI stays consistent
  }, [song?.id]);

  const toggle = useCallback(() => {
    const a = audioRef.current;
    if (!a || !a.src) return;
    if (a.paused) a.play().catch(() => setPlaying(false));
    else a.pause();
  }, []);

  const seek = useCallback((seconds: number) => {
    const a = audioRef.current;
    if (!a) return;
    a.currentTime = seconds;
    setTime((t) => ({ ...t, currentTime: seconds }));
  }, []);

  const setVolume = useCallback((v: number) => {
    if (audioRef.current) audioRef.current.volume = v;
    setVolumeState(v);
  }, []);

  // Lock-screen / hardware media keys.
  useEffect(() => {
    if (!song || !("mediaSession" in navigator)) return;
    navigator.mediaSession.metadata = new MediaMetadata({
      title: songTitle(song),
      artist: songArtist(song),
      album: song.album ?? "Constellation",
    });
    navigator.mediaSession.setActionHandler("play", toggle);
    navigator.mediaSession.setActionHandler("pause", toggle);
  }, [song, toggle]);

  const api = useMemo<PlayerApi>(
    () => ({ song, playing, buffering, volume, play, toggle, seek, setVolume }),
    [song, playing, buffering, volume, play, toggle, seek, setVolume],
  );

  return (
    <PlayerContext.Provider value={api}>
      <PlayerTimeContext.Provider value={time}>{children}</PlayerTimeContext.Provider>
    </PlayerContext.Provider>
  );
}

export function usePlayer(): PlayerApi {
  const ctx = useContext(PlayerContext);
  if (!ctx) throw new Error("usePlayer must be used inside <PlayerProvider>");
  return ctx;
}

export function usePlayerTime(): PlayerTime {
  return useContext(PlayerTimeContext);
}
