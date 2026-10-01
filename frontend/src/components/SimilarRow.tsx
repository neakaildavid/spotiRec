import { ChevronLeft, ChevronRight, CircleAlert } from "lucide-react";
import { useRef, type ReactNode } from "react";
import { useNavigate } from "react-router";
import { useAsync } from "../hooks/useAsync";
import type { DiscoverResult } from "../lib/api";
import { SongCardSkeleton } from "./Skeleton";
import { SongCard } from "./SongCard";

interface Props {
  title: string;
  subtitle?: ReactNode;
  /** Changing the key reloads the row. */
  loadKey: string;
  load: (signal: AbortSignal) => Promise<DiscoverResult>;
  action?: ReactNode;
}

/**
 * Horizontally scrolling row of SongCards ("Sounds similar", "It sounds like…").
 * Captions show the genre rather than a "% similar": cosine scores aren't
 * calibrated percentages, so the raw score is only in the tooltip.
 */
export function SimilarRow({ title, subtitle, loadKey, load, action }: Props) {
  const { data, error, loading, retry } = useAsync(load, loadKey);
  const scroller = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const scroll = (dir: 1 | -1) => scroller.current?.scrollBy({ left: dir * scroller.current.clientWidth * 0.8, behavior: "smooth" });

  return (
    <section className="animate-fade-up" aria-label={title}>
      <div className="mb-3 flex items-end justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-xl font-black tracking-tight md:text-2xl">{title}</h3>
          {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {action}
          <div className="hidden gap-1 md:flex">
            {([-1, 1] as const).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => scroll(d)}
                aria-label={d < 0 ? "Scroll left" : "Scroll right"}
                className="grid size-8 place-items-center rounded-full bg-elevated text-muted transition-colors hover:text-fg"
              >
                {d < 0 ? <ChevronLeft className="size-4" /> : <ChevronRight className="size-4" />}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error ? (
        <p className="flex items-center gap-2 rounded-lg bg-surface px-4 py-3 text-sm text-muted">
          <CircleAlert className="size-4 shrink-0 text-accent" aria-hidden="true" />
          {error}
          <button type="button" onClick={retry} className="ml-auto font-semibold text-fg underline">Retry</button>
        </p>
      ) : (
        <div ref={scroller} className="-mx-1 flex snap-x gap-3 overflow-x-auto px-1 pb-2 [scrollbar-width:none] md:gap-4">
          {loading
            ? Array.from({ length: 6 }, (_, i) => (
                <div key={i} className="w-40 shrink-0 md:w-44">
                  <SongCardSkeleton />
                </div>
              ))
            : data?.items.map((item) => (
                <div key={item.song.id} className="w-40 shrink-0 snap-start md:w-44">
                  <SongCard
                    song={item.song}
                    caption={item.song.genre ?? undefined}
                    hint={`Similarity ${item.score.toFixed(3)}`}
                    onMoreLike={(s) => navigate(`/discover?song=${s.id}`)}
                  />
                </div>
              ))}
          {!loading && data?.items.length === 0 && <p className="py-6 text-sm text-muted">Nothing similar found yet.</p>}
        </div>
      )}
    </section>
  );
}
