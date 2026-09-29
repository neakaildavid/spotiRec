import { CircleAlert, Shuffle, Upload } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { ConstellationViz, type VizPeak } from "../components/ConstellationViz";
import { RecordButton, type RecordState } from "../components/RecordButton";
import { NoMatchCard, ResultCard, ResultCardSkeleton } from "../components/ResultCard";
import { useRecorder, type CapturedPeak } from "../hooks/useRecorder";
import { identify, randomSong, type IdentifyResult, type Song } from "../lib/api";
import { songArtist, songTitle } from "../lib/format";
import { usePlayer } from "../player/PlayerContext";

const MAX_RECORD_MS = 10_000; // eval: 10 s clips identify ~99% even at 0 dB SNR
const MIN_MATCHING_MS = 700; // avoid a sub-second flash of the matching state

type Phase =
  | { kind: "idle" }
  | { kind: "matching"; peaks: VizPeak[]; duration: number }
  | { kind: "result"; result: IdentifyResult }
  | { kind: "error"; message: string };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function IdentifyPage() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [nowPlayingHint, setNowPlayingHint] = useState<Song | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const player = usePlayer();

  const run = useCallback(async (blob: Blob, filename: string, peaks: VizPeak[], duration: number) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setPhase({ kind: "matching", peaks, duration });
    try {
      const [result] = await Promise.all([identify(blob, filename, ctrl.signal), sleep(MIN_MATCHING_MS)]);
      setPhase({ kind: "result", result });
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setPhase({ kind: "error", message: (e as Error).message || "Something went wrong." });
    }
  }, []);

  const recorder = useRecorder({
    maxMs: MAX_RECORD_MS,
    onComplete: (blob, filename, peaks: CapturedPeak[]) =>
      run(blob, filename, peaks.map((p) => ({ t: p.t, f: p.f })), peaks.at(-1)?.t ?? MAX_RECORD_MS / 1000),
  });

  const reset = () => {
    abortRef.current?.abort();
    setPhase({ kind: "idle" });
  };

  const onRecordClick = () => {
    if (recorder.status === "recording") recorder.stop();
    else {
      setPhase({ kind: "idle" });
      void recorder.start();
    }
  };

  const onFile = (file: File | undefined) => {
    if (!file) return;
    void run(file, file.name, [], 10);
    if (fileRef.current) fileRef.current.value = "";
  };

  const onRandom = async () => {
    try {
      const song = await randomSong();
      player.play(song);
      setNowPlayingHint(song);
    } catch (e) {
      setPhase({ kind: "error", message: (e as Error).message });
    }
  };

  const recState: RecordState =
    recorder.status === "recording" ? "listening"
      : recorder.status === "requesting" ? "requesting"
      : phase.kind === "matching" ? "matching"
      : "idle";

  const secondsLeft = Math.max(0, Math.ceil((MAX_RECORD_MS - recorder.elapsedMs) / 1000));
  const status =
    recState === "listening" ? `Listening… ${secondsLeft}s`
      : recState === "requesting" ? "Waiting for microphone…"
      : recState === "matching" ? "Matching against the library…"
      : "Tap to identify";

  const result = phase.kind === "result" ? phase.result : null;
  const showHero = phase.kind === "idle" || phase.kind === "matching" || phase.kind === "error" || recState === "listening";

  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col px-4 pt-6 pb-10 md:px-8 md:pt-10">
      {showHero && !result && (
        // On phones the button sits in the lower half of the screen (thumb
        // reach); on larger screens it's centered.
        <section className="flex flex-1 flex-col items-center justify-end text-center md:justify-center">
          <h1 className="text-4xl font-black tracking-tight text-balance md:text-6xl">What's playing?</h1>
          <p className="mt-3 max-w-md text-muted">
            Hold your device near the music. Constellation listens for up to 10 seconds and matches the sound's
            fingerprint against {`the library`}.
          </p>

          <div className="mt-4 md:mt-8">
            <RecordButton
              state={recState}
              progress={recorder.status === "recording" ? Math.min(1, recorder.elapsedMs / MAX_RECORD_MS) : 0}
              analyser={recorder.analyser}
              onClick={onRecordClick}
              label={recState === "listening" ? "Stop listening and identify" : "Start listening"}
            />
          </div>
          <p className="mt-1 text-lg font-bold tabular-nums" role="status" aria-live="polite">
            {status}
          </p>

          {(phase.kind === "error" || recorder.status === "error") && (
            <p className="mt-4 flex max-w-md items-center gap-2 rounded-lg bg-elevated px-4 py-3 text-left text-sm text-fg" role="alert">
              <CircleAlert className="size-4 shrink-0 text-accent" aria-hidden="true" />
              {phase.kind === "error" ? phase.message : recorder.error}
            </p>
          )}

          {phase.kind !== "matching" && recState !== "listening" && (
            <div className="mt-8 flex flex-wrap justify-center gap-3">
              <button
                type="button"
                onClick={() => fileRef.current?.click()}
                className="inline-flex items-center gap-2 rounded-full border border-line px-5 py-2.5 text-sm font-semibold transition-colors duration-150 hover:border-muted"
              >
                <Upload className="size-4" aria-hidden="true" /> Upload a clip
              </button>
              <button
                type="button"
                onClick={onRandom}
                className="inline-flex items-center gap-2 rounded-full border border-line px-5 py-2.5 text-sm font-semibold transition-colors duration-150 hover:border-muted"
              >
                <Shuffle className="size-4" aria-hidden="true" /> Play a random library song
              </button>
              <input ref={fileRef} type="file" accept="audio/*,video/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
            </div>
          )}
          {nowPlayingHint && phase.kind === "idle" && recState === "idle" && (
            <p className="mt-3 max-w-sm text-sm text-subtle">
              Now playing <span className="text-muted">{songTitle(nowPlayingHint)}</span> by{" "}
              <span className="text-muted">{songArtist(nowPlayingHint)}</span>. Identify it from another device, or
              from this one with the volume up.
            </p>
          )}
        </section>
      )}

      {phase.kind === "matching" && (
        <section className="mt-8 animate-fade-up space-y-4" aria-label="Matching">
          <div className="rounded-2xl bg-surface p-4">
            <p className="mb-2 text-xs font-bold tracking-[0.2em] text-subtle uppercase">Constellation map</p>
            <ConstellationViz peaks={phase.peaks.length ? phase.peaks : ambientStars()} duration={phase.duration} mode="scanning" className="h-40 md:h-48" />
          </div>
          <ResultCardSkeleton />
        </section>
      )}

      {result && (
        <div className="space-y-4">
          {result.match && result.song ? (
            <ResultCard song={result.song} result={result} onReset={reset} />
          ) : (
            <NoMatchCard result={result} onReset={reset} />
          )}
          <HowItMatched result={result} />
        </div>
      )}
    </div>
  );
}

/** The server's real constellation map for the query, plus the numbers behind the match. */
function HowItMatched({ result }: { result: IdentifyResult }) {
  const peaks = result.query.peaks.map(([t, f]) => ({ t, f }));
  const stats: [string, string][] = [
    ["Spectral peaks", result.query.peaks.length.toLocaleString()],
    ["Hashes (peak pairs)", result.query.hashes.toLocaleString()],
    ["Time-aligned matches", `${result.aligned_matches} vs ${result.runner_up_matches} runner-up`],
    ["Server time", `${Math.round(result.timing.total_ms)} ms`],
  ];
  return (
    <section className="animate-fade-up rounded-2xl bg-surface p-4 md:p-6" style={{ animationDelay: "120ms" }}>
      <h3 className="text-xs font-bold tracking-[0.2em] text-subtle uppercase">How it matched</h3>
      <ConstellationViz peaks={peaks} duration={result.query.duration_s} mode="resolved" className="mt-3 h-40 md:h-48" />
      <dl className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-4">
        {stats.map(([k, v]) => (
          <div key={k}>
            <dt className="text-xs text-subtle">{k}</dt>
            <dd className="mt-0.5 font-semibold tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-sm text-muted">
        Each line pairs two peaks into a hash of (frequency 1, frequency 2, time gap). A true match is a song whose hashes
        line up at one consistent time offset; chance collisions scatter.
      </p>
    </section>
  );
}

/** Decorative placeholder stars while an uploaded file is being matched
 *  (we have no client-side peaks for uploads). */
function ambientStars(): VizPeak[] {
  let s = 7;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  return Array.from({ length: 90 }, () => ({ t: rnd() * 10, f: 80 * Math.pow(5500 / 80, rnd()) }));
}
