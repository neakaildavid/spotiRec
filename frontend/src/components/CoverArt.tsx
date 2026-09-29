import { memo, useId, useMemo, type ReactNode } from "react";
import { coverSpec } from "../lib/cover";

interface Props {
  songId: number;
  className?: string;
}

/**
 * Generated square cover for a song (see lib/cover.ts). Pure SVG, so it's
 * crisp at every size and costs no network request. Memoized because library
 * grids render dozens at once.
 */
export const CoverArt = memo(function CoverArt({ songId, className = "" }: Props) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const art = useMemo(() => renderCover(songId, `c${uid}`), [songId, uid]);
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice" className={`block ${className}`} aria-hidden="true">
      {art}
    </svg>
  );
});

function renderCover(id: number, uid: string): ReactNode {
  const { style, colors, angle, rng } = coverSpec(id);
  const [dark, mid, light] = colors;
  const r = (lo: number, hi: number) => lo + rng() * (hi - lo);

  const defs = (
    <defs>
      <linearGradient id={`${uid}bg`} gradientTransform={`rotate(${angle} .5 .5)`}>
        <stop offset="0" stopColor={dark} />
        <stop offset="1" stopColor={mid} stopOpacity=".85" />
      </linearGradient>
      <radialGradient id={`${uid}glow`}>
        <stop offset="0" stopColor={light} stopOpacity=".95" />
        <stop offset=".45" stopColor={mid} stopOpacity=".45" />
        <stop offset="1" stopColor={mid} stopOpacity="0" />
      </radialGradient>
      <radialGradient id={`${uid}vig`} cx=".5" cy=".4" r=".8">
        <stop offset=".55" stopColor="#000" stopOpacity="0" />
        <stop offset="1" stopColor="#000" stopOpacity=".45" />
      </radialGradient>
    </defs>
  );

  let body: ReactNode;
  switch (style) {
    case "orbs": {
      body = Array.from({ length: 3 }, (_, i) => (
        <circle key={i} cx={r(10, 90)} cy={r(10, 90)} r={r(22, 48)} fill={`url(#${uid}glow)`} opacity={r(0.55, 0.95)} />
      ));
      break;
    }
    case "rings": {
      const cx = r(20, 80);
      const cy = r(20, 80);
      body = (
        <>
          {Array.from({ length: 9 }, (_, i) => (
            <circle key={i} cx={cx} cy={cy} r={6 + i * 9} fill="none" stroke={i % 3 === 0 ? light : mid}
              strokeWidth={i % 3 === 0 ? 2.2 : 1} opacity={1 - i * 0.09} />
          ))}
          <circle cx={cx} cy={cy} r={5} fill={light} />
        </>
      );
      break;
    }
    case "constellation": {
      const pts = Array.from({ length: 7 }, (_, i) => [8 + i * 13 + r(-4, 4), r(18, 82)] as const);
      const hero = Math.floor(r(1, 6));
      body = (
        <>
          <path d={`M${pts.map(([x, y]) => `${x} ${y}`).join(" L")}`} fill="none" stroke={light} strokeOpacity=".45" strokeWidth=".8" />
          {pts.map(([x, y], i) =>
            i === hero ? (
              <g key={i}>
                <circle cx={x} cy={y} r={16} fill={`url(#${uid}glow)`} />
                <circle cx={x} cy={y} r={2.8} fill="#fff" />
              </g>
            ) : (
              <circle key={i} cx={x} cy={y} r={r(1, 1.8)} fill="#fff" opacity={r(0.6, 1)} />
            ),
          )}
          {Array.from({ length: 18 }, (_, i) => (
            <circle key={`s${i}`} cx={r(0, 100)} cy={r(0, 100)} r={r(0.2, 0.55)} fill="#fff" opacity={r(0.2, 0.6)} />
          ))}
        </>
      );
      break;
    }
    case "waves": {
      body = Array.from({ length: 5 }, (_, i) => {
        const y = 30 + i * 14 + r(-4, 4);
        const a = r(6, 14);
        return (
          <path key={i} d={`M-5 ${y} Q 25 ${y - a} 50 ${y} T 105 ${y} V 110 H -5 Z`}
            fill={i % 2 ? mid : light} opacity={0.12 + i * 0.1} />
        );
      });
      break;
    }
  }

  return (
    <>
      {defs}
      <rect width="100" height="100" fill={`url(#${uid}bg)`} />
      {body}
      <rect width="100" height="100" fill={`url(#${uid}vig)`} />
    </>
  );
}
