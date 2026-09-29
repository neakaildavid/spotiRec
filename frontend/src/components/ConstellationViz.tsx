import { useEffect, useMemo, useRef } from "react";
import { useReducedMotion } from "../hooks/useReducedMotion";

export interface VizPeak {
  t: number; // seconds
  f: number; // Hz
}

interface Props {
  peaks: VizPeak[];
  duration: number;
  /** "scanning": a sweep lights stars as it passes (while matching).
   *  "resolved": stars and their hash pairs light up once, then stay lit. */
  mode: "scanning" | "resolved";
  className?: string;
}

const F_MIN = 80;
const F_MAX = 5500;
const PAIR_DT = 1.2; // s, roughly the fingerprint target zone
const FAN_OUT = 2;

/**
 * Animated constellation map: spectrogram peaks as stars (x = time, y = log
 * frequency), with lines for anchor -> target pairs. It visualizes what
 * the fingerprinter actually hashes: (f1, f2, dt) pairs, not single peaks.
 */
export function ConstellationViz({ peaks, duration, mode, className = "" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reduced = useReducedMotion();

  const layout = useMemo(() => {
    const dur = Math.max(duration, 0.5);
    const logSpan = Math.log(F_MAX / F_MIN);
    const stars = peaks
      .filter((p) => p.f >= F_MIN && p.f <= F_MAX && p.t <= dur)
      .map((p) => ({ x: p.t / dur, y: 1 - Math.log(p.f / F_MIN) / logSpan, t: p.t }))
      .sort((a, b) => a.x - b.x);
    const pairs: [number, number][] = [];
    for (let i = 0; i < stars.length; i++) {
      let n = 0;
      for (let j = i + 1; j < stars.length && n < FAN_OUT; j++) {
        const dt = stars[j].t - stars[i].t;
        if (dt <= 0.05) continue;
        if (dt > PAIR_DT) break;
        if (Math.abs(stars[j].y - stars[i].y) < 0.35) {
          pairs.push([i, j]);
          n++;
        }
      }
    }
    return { stars, pairs };
  }, [peaks, duration]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--color-accent").trim() || "#a06cff";
    const line = css.getPropertyValue("--color-line").trim() || "#2c2838";
    const subtle = css.getPropertyValue("--color-subtle").trim() || "#6f6a7e";

    let w = 0;
    let h = 0;
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const PAD_L = 44;
    const PAD = 10;
    const X = (x: number) => PAD_L + x * (w - PAD_L - PAD);
    const Y = (y: number) => PAD + y * (h - 2 * PAD);
    const { stars, pairs } = layout;
    const start = performance.now();
    const SWEEP_MS = 1600;
    const REVEAL_MS = 1100;

    const drawAxes = () => {
      ctx.font = "10px Inter, sans-serif";
      ctx.textBaseline = "middle";
      for (const f of [250, 1000, 4000]) {
        const y = Y(1 - Math.log(f / F_MIN) / Math.log(F_MAX / F_MIN));
        ctx.strokeStyle = line;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.moveTo(PAD_L, y);
        ctx.lineTo(w - PAD, y);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = subtle;
        ctx.fillText(f >= 1000 ? `${f / 1000} kHz` : `${f} Hz`, 0, y);
      }
    };

    let raf = 0;
    const frame = (now: number) => {
      const elapsed = now - start;
      ctx.clearRect(0, 0, w, h);
      drawAxes();

      // brightness(x): how lit a star at horizontal position x is right now
      let brightness: (x: number) => number;
      let sweepX: number | null = null;
      if (reduced) {
        brightness = () => (mode === "resolved" ? 1 : 0.6);
      } else if (mode === "scanning") {
        sweepX = (elapsed % SWEEP_MS) / SWEEP_MS;
        const sx = sweepX;
        brightness = (x) => {
          const behind = sx - x;
          return behind >= 0 ? 0.25 + 0.75 * Math.exp(-behind * 6) : 0.25;
        };
      } else {
        const p = Math.min(1, elapsed / REVEAL_MS);
        brightness = (x) => (x <= p ? 1 : 0.2);
      }

      // Pair lines (the actual hashes)
      ctx.lineWidth = 1;
      for (const [i, j] of pairs) {
        const a = stars[i];
        const b = stars[j];
        const lit = Math.min(brightness(a.x), brightness(b.x));
        ctx.strokeStyle = accent;
        ctx.globalAlpha = mode === "resolved" ? 0.12 + 0.4 * lit : 0.08 + 0.3 * (lit - 0.25);
        ctx.beginPath();
        ctx.moveTo(X(a.x), Y(a.y));
        ctx.lineTo(X(b.x), Y(b.y));
        ctx.stroke();
      }

      // Stars
      for (const s of stars) {
        const lit = brightness(s.x);
        const x = X(s.x);
        const y = Y(s.y);
        if (lit > 0.6) {
          ctx.globalAlpha = (lit - 0.6) * 0.9;
          ctx.fillStyle = accent;
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 0.35 + 0.65 * lit;
        ctx.fillStyle = lit > 0.6 ? "#fff" : subtle;
        ctx.beginPath();
        ctx.arc(x, y, 1.6 + lit * 0.8, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      // Sweep line
      if (sweepX !== null) {
        const x = X(sweepX);
        const g = ctx.createLinearGradient(x - 40, 0, x, 0);
        g.addColorStop(0, "transparent");
        g.addColorStop(1, accent);
        ctx.globalAlpha = 0.25;
        ctx.fillStyle = g;
        ctx.fillRect(x - 40, PAD, 40, h - 2 * PAD);
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = accent;
        ctx.fillRect(x - 1, PAD, 2, h - 2 * PAD);
        ctx.globalAlpha = 1;
      }

      const animating = !reduced && (mode === "scanning" || elapsed < REVEAL_MS + 50);
      if (animating) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [layout, mode, reduced]);

  return (
    <canvas
      ref={canvasRef}
      className={`block w-full ${className}`}
      role="img"
      aria-label={`Constellation map of ${layout.stars.length} spectral peaks and ${layout.pairs.length} peak pairs`}
    />
  );
}
