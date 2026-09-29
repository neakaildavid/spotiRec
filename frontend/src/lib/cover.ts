/**
 * Deterministic generative cover art.
 *
 * FMA has no reliable album art, so every song gets an abstract cover derived
 * from its id: same id -> same cover on every device and render, with no
 * storage and no network requests. A seeded PRNG picks a palette and one of a
 * few compositions; palettes are hand-picked so every combination looks
 * intentional next to the violet accent.
 */

export type CoverStyle = "orbs" | "rings" | "constellation" | "waves";

export interface CoverSpec {
  style: CoverStyle;
  colors: [string, string, string];
  angle: number;
  rng: () => number; // fresh PRNG positioned after the choices above
}

const PALETTES: [string, string, string][] = [
  ["#2a1b5e", "#a06cff", "#ff7ac6"],
  ["#0f2a3f", "#2fb5c9", "#b5f5e6"],
  ["#3a1030", "#e2487a", "#ffb36b"],
  ["#14213d", "#5a7bff", "#c2a8ff"],
  ["#1f2d16", "#6fbf73", "#e8f59a"],
  ["#2b1a0f", "#e0863f", "#ffe0a3"],
  ["#1b1030", "#7b4dff", "#46e0d4"],
  ["#301014", "#ff5d5d", "#ffc7a8"],
  ["#0e1b2c", "#3f8cff", "#9ff0ff"],
  ["#261033", "#c054ff", "#ffd36b"],
];

const STYLES: CoverStyle[] = ["orbs", "rings", "constellation", "waves"];

/** mulberry32: tiny, fast, good-enough seeded PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Scramble sequential ids so neighbours (1, 2, 3...) don't look alike. */
function hashId(id: number): number {
  let h = Math.imul(id ^ 0x9e3779b9, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

export function coverSpec(id: number): CoverSpec {
  const rng = mulberry32(hashId(id));
  const colors = PALETTES[Math.floor(rng() * PALETTES.length)];
  const style = STYLES[Math.floor(rng() * STYLES.length)];
  const angle = Math.floor(rng() * 360);
  return { style, colors, angle, rng };
}
