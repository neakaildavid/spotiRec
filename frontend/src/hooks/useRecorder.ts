/**
 * Microphone capture: MediaRecorder for the clip we upload, plus a Web Audio
 * AnalyserNode for the live visualization.
 *
 * While recording we also sample the analyser ~10x/second and keep the
 * strongest spectral peaks: a rough, client-side constellation map that the
 * UI animates while the server does the real matching.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderStatus = "idle" | "requesting" | "recording" | "error";

export interface CapturedPeak {
  t: number; // seconds since start
  f: number; // Hz
  m: number; // 0..1 relative magnitude
}

interface Options {
  maxMs: number;
  onComplete: (blob: Blob, filename: string, peaks: CapturedPeak[]) => void;
}

const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];

function pickMime(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m));
}

function extensionFor(mime: string): string {
  if (mime.includes("mp4")) return "m4a";
  if (mime.includes("ogg")) return "ogg";
  return "webm";
}

function describeError(e: unknown): string {
  const name = (e as DOMException)?.name;
  if (name === "NotAllowedError") return "Microphone access was blocked. Allow it in your browser's site settings.";
  if (name === "NotFoundError") return "No microphone was found on this device.";
  if (name === "NotReadableError") return "The microphone is being used by another app.";
  return "Couldn't start recording.";
}

export function useRecorder({ maxMs, onComplete }: Options) {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const recorderRef = useRef<MediaRecorder | null>(null);
  const cleanupRef = useRef<() => void>(() => {});
  const cancelledRef = useRef(false);

  const stop = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state === "recording") rec.stop();
  }, []);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    stop();
  }, [stop]);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("Recording needs a secure context (https or localhost) and a modern browser.");
      setStatus("error");
      return;
    }
    setError(null);
    setStatus("requesting");
    cancelledRef.current = false;

    let stream: MediaStream;
    try {
      // Voice-call processing is designed to *remove* music and speaker
      // playback, which is exactly the signal we want, so it's all disabled.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
    } catch (e) {
      setError(describeError(e));
      setStatus("error");
      return;
    }

    const ctx = new AudioContext();
    const node = ctx.createAnalyser();
    node.fftSize = 2048;
    node.smoothingTimeConstant = 0.55;
    ctx.createMediaStreamSource(stream).connect(node);

    const mime = pickMime();
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks: Blob[] = [];
    const peaks: CapturedPeak[] = [];
    const bins = new Float32Array(node.frequencyBinCount);
    const hzPerBin = ctx.sampleRate / node.fftSize;
    const lo = Math.ceil(100 / hzPerBin);
    const hi = Math.floor(5000 / hzPerBin);
    const startedAt = performance.now();

    const interval = window.setInterval(() => {
      const elapsed = performance.now() - startedAt;
      setElapsedMs(elapsed);
      // Top local maxima of the current spectrum -> constellation points.
      node.getFloatFrequencyData(bins);
      const cands: [number, number][] = [];
      for (let i = lo + 1; i < hi - 1; i++) {
        const v = bins[i];
        if (v > -90 && v >= bins[i - 1] && v >= bins[i + 1]) cands.push([v, i]);
      }
      cands.sort((a, b) => b[0] - a[0]);
      for (const [v, i] of cands.slice(0, 3)) {
        peaks.push({ t: elapsed / 1000, f: i * hzPerBin, m: Math.min(1, Math.max(0, (v + 90) / 60)) });
      }
      if (elapsed >= maxMs) stop();
    }, 100);

    const cleanup = () => {
      window.clearInterval(interval);
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
      setAnalyser(null);
    };
    cleanupRef.current = cleanup;

    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      cleanup();
      recorderRef.current = null;
      setStatus("idle");
      if (cancelledRef.current) return;
      const type = rec.mimeType || mime || "audio/webm";
      onCompleteRef.current(new Blob(chunks, { type }), `recording.${extensionFor(type)}`, peaks);
    };

    recorderRef.current = rec;
    rec.start(250);
    setAnalyser(node);
    setElapsedMs(0);
    setStatus("recording");
  }, [maxMs, stop]);

  // Release the mic if the component unmounts mid-recording.
  useEffect(() => () => {
    cancelledRef.current = true;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    cleanupRef.current();
  }, []);

  return { status, error, elapsedMs, analyser, start, stop, cancel };
}
