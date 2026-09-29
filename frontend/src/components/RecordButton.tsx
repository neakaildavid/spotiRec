import { useEffect, useRef } from "react";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { LogoMark } from "./Logo";

export type RecordState = "idle" | "requesting" | "listening" | "matching";

interface Props {
  state: RecordState;
  /** 0..1 progress through the maximum recording length. */
  progress: number;
  analyser: AnalyserNode | null;
  onClick: () => void;
  label: string;
}

const SIZE = 208; // button diameter, px
const HALO = 1.75; // halo canvas size relative to the button

/**
 * The big circular "identify" button.
 *
 * While listening: pulsing rings (CSS), a progress ring (SVG), and a radial
 * frequency halo drawn from the live AnalyserNode on a canvas, so users can
 * see the mic is actually hearing the music.
 */
export function RecordButton({ state, progress, analyser, onClick, label }: Props) {
  const listening = state === "listening";
  const busy = state === "matching" || state === "requesting";
  const circumference = 2 * Math.PI * (SIZE / 2 + 10);

  return (
    <div className="relative grid place-items-center" style={{ width: SIZE * HALO, height: SIZE * HALO }}>
      {listening && <FrequencyHalo analyser={analyser} />}

      {listening &&
        [0, 600].map((delay) => (
          <span
            key={delay}
            className="pointer-events-none absolute rounded-full border-2 border-accent/60 animate-pulse-ring"
            style={{ width: SIZE, height: SIZE, animationDelay: `${delay}ms` }}
            aria-hidden="true"
          />
        ))}

      {/* Progress ring */}
      <svg className="pointer-events-none absolute -rotate-90" width={SIZE + 28} height={SIZE + 28} aria-hidden="true">
        <circle cx="50%" cy="50%" r={SIZE / 2 + 10} fill="none" stroke="var(--color-line)" strokeWidth="3" opacity={listening ? 1 : 0} />
        <circle
          cx="50%" cy="50%" r={SIZE / 2 + 10} fill="none" stroke="var(--color-accent)" strokeWidth="3" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={circumference * (1 - progress)}
          style={{ transition: "stroke-dashoffset 120ms linear", opacity: listening ? 1 : 0 }}
        />
      </svg>

      <button
        type="button"
        onClick={onClick}
        disabled={busy}
        aria-label={label}
        aria-pressed={listening}
        className={`relative grid place-items-center rounded-full text-white shadow-[0_20px_60px_-15px] shadow-accent/60 transition duration-200 ease-out-soft focus-visible:outline-offset-8 ${
          busy ? "scale-95 cursor-wait" : "hover:scale-[1.03] active:scale-[0.97]"
        }`}
        style={{
          width: SIZE,
          height: SIZE,
          background: "radial-gradient(circle at 30% 25%, var(--color-accent-hover), var(--color-accent) 45%, var(--color-accent-deep))",
        }}
      >
        <span className="absolute inset-3 rounded-full border border-white/15" aria-hidden="true" />
        {busy ? (
          <span className="absolute inset-0 rounded-full skeleton opacity-40" aria-hidden="true" />
        ) : null}
        <LogoMark mono className={`size-24 text-white drop-shadow-lg transition-transform duration-200 ${listening ? "scale-110" : ""}`} />
      </button>
    </div>
  );
}

/** Radial bars around the button, one per log-spaced frequency band. */
function FrequencyHalo({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const px = SIZE * HALO;
    canvas.width = px * dpr;
    canvas.height = px * dpr;
    ctx.scale(dpr, dpr);

    const data = new Uint8Array(analyser.frequencyBinCount);
    const BARS = 72;
    // Log-spaced band edges from ~60 Hz to ~8 kHz: music energy is spread
    // logarithmically, so linear bins would put almost everything in a few bars.
    const nyquist = analyser.context.sampleRate / 2;
    const edges = Array.from({ length: BARS + 1 }, (_, i) =>
      Math.min(data.length - 1, Math.round(((60 * Math.pow(8000 / 60, i / BARS)) / nyquist) * data.length)),
    );
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--color-accent").trim() || "#a06cff";
    const smooth = new Float32Array(BARS);
    let raf = 0;
    let last = 0;

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (reduced && now - last < 200) return; // calmer update rate with reduced motion
      last = now;
      analyser.getByteFrequencyData(data);
      ctx.clearRect(0, 0, px, px);
      const c = px / 2;
      const inner = SIZE / 2 + 18;
      ctx.lineCap = "round";
      for (let i = 0; i < BARS; i++) {
        let peak = 0;
        for (let b = edges[i]; b <= Math.max(edges[i], edges[i + 1] - 1); b++) peak = Math.max(peak, data[b]);
        smooth[i] = smooth[i] * 0.6 + (peak / 255) * 0.4;
        const len = 4 + smooth[i] * (px / 2 - inner - 6);
        const angle = (i / BARS) * Math.PI * 2 - Math.PI / 2;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        ctx.strokeStyle = accent;
        ctx.globalAlpha = 0.25 + smooth[i] * 0.75;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(c + cos * inner, c + sin * inner);
        ctx.lineTo(c + cos * (inner + len), c + sin * (inner + len));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [analyser, reduced]);

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full" aria-hidden="true" />;
}
