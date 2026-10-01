import { CircleAlert, LayoutGrid, List, Search, SearchX } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { GenreChip } from "../components/GenreChip";
import { SongCard } from "../components/SongCard";
import { SongCardSkeleton, TrackRowSkeleton } from "../components/Skeleton";
import { TrackListHeader, TrackRow } from "../components/TrackRow";
import { useAsync } from "../hooks/useAsync";
import { useDebounced, useSongs } from "../hooks/useSongs";
import { listGenres, type SortField, type SortOrder } from "../lib/api";

type View = "grid" | "list";
const VIEW_KEY = "constellation.libraryView";

function readView(): View {
  try {
    return localStorage.getItem(VIEW_KEY) === "list" ? "list" : "grid";
  } catch {
    return "grid"; // storage can be unavailable (private mode, blocked cookies)
  }
}

const SORT_LABELS: Record<SortField, string> = {
  id: "Recently added",
  title: "Title",
  artist: "Artist",
  duration_s: "Duration",
};

export function LibraryPage() {
  const [view, setView] = useState<View>(readView);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortField>("id");
  const [order, setOrder] = useState<SortOrder>("desc");
  const [genre, setGenre] = useState("");
  const q = useDebounced(query.trim(), 250);
  const { items, total, loading, error, hasMore, loadMore, retry } = useSongs(q, sort, order, genre);
  const genres = useAsync((s) => listGenres(s), "genres");
  const navigate = useNavigate();

  useEffect(() => {
    try {
      localStorage.setItem(VIEW_KEY, view);
    } catch {
      /* ignore */
    }
  }, [view]);

  // Infinite scroll: load the next page when the sentinel nears the viewport.
  const sentinel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = sentinel.current;
    if (!el) return;
    const io = new IntersectionObserver((entries) => entries[0].isIntersecting && loadMore(), { rootMargin: "600px" });
    io.observe(el);
    return () => io.disconnect();
  }, [loadMore]);

  const onHeaderSort = (field: SortField) => {
    if (field === sort) setOrder(order === "asc" ? "desc" : "asc");
    else {
      setSort(field);
      setOrder(field === "id" ? "desc" : "asc");
    }
  };

  const initialLoading = loading && items.length === 0;

  return (
    <div className="mx-auto max-w-7xl px-4 pt-6 pb-10 md:px-8 md:pt-10">
      <header className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-4xl font-black tracking-tight md:text-5xl">Library</h1>
          <p className="mt-1 text-muted tabular-nums">
            {total === null ? " " : `${total.toLocaleString()} ${q || genre ? "matching " : ""}tracks`}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="relative w-full md:w-72">
            <span className="sr-only">Search the library</span>
            <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-subtle" aria-hidden="true" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search titles or artists"
              className="h-10 w-full rounded-full bg-elevated pr-4 pl-9 text-sm placeholder:text-subtle transition-shadow duration-150 outline-none focus:ring-2 focus:ring-accent"
            />
          </label>
          <label className="sr-only" htmlFor="sort">Sort by</label>
          <select
            id="sort"
            value={`${sort}:${order}`}
            onChange={(e) => {
              const [s, o] = e.target.value.split(":") as [SortField, SortOrder];
              setSort(s);
              setOrder(o);
            }}
            className="h-10 rounded-full bg-elevated px-4 text-sm font-medium outline-none focus:ring-2 focus:ring-accent"
          >
            {(Object.keys(SORT_LABELS) as SortField[]).flatMap((f) =>
              (f === "id" ? (["desc"] as const) : (["asc", "desc"] as const)).map((o) => (
                <option key={`${f}:${o}`} value={`${f}:${o}`}>
                  {SORT_LABELS[f]}
                  {f !== "id" ? (o === "asc" ? " ↑" : " ↓") : ""}
                </option>
              )),
            )}
          </select>
          <div className="flex rounded-full bg-elevated p-1" role="group" aria-label="View">
            {(["grid", "list"] as const).map((v) => {
              const Icon = v === "grid" ? LayoutGrid : List;
              return (
                <button
                  key={v}
                  type="button"
                  onClick={() => setView(v)}
                  aria-pressed={view === v}
                  aria-label={v === "grid" ? "Grid view" : "List view"}
                  className={`grid size-8 place-items-center rounded-full transition-colors duration-150 ${view === v ? "bg-hover text-fg" : "text-subtle hover:text-fg"}`}
                >
                  <Icon className="size-4" />
                </button>
              );
            })}
          </div>
        </div>
      </header>

      {genres.data && genres.data.length > 0 && (
        <div className="-mx-1 mt-6 flex gap-2 overflow-x-auto px-1 pb-1 [scrollbar-width:none]" role="group" aria-label="Filter by genre">
          <GenreChip label="All" active={!genre} onClick={() => setGenre("")} />
          {genres.data.map((g) => (
            <GenreChip key={g.genre} label={g.genre} active={genre === g.genre} onClick={() => setGenre(genre === g.genre ? "" : g.genre)} />
          ))}
        </div>
      )}

      <div className="mt-8">
        {error && items.length === 0 ? (
          <EmptyState icon={<CircleAlert className="size-8" />} title="Couldn't load the library" body={error}>
            <button type="button" onClick={retry} className="mt-4 rounded-full bg-fg px-5 py-2 text-sm font-bold text-bg">Retry</button>
          </EmptyState>
        ) : !initialLoading && items.length === 0 ? (
          <EmptyState icon={<SearchX className="size-8" />} title={q ? `No tracks match "${q}"` : "No tracks here yet"} body="Try a different title or artist." />
        ) : view === "grid" ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:gap-4 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
            {items.map((s) => (
              <SongCard key={s.id} song={s} caption={s.genre ?? undefined} onMoreLike={(x) => navigate(`/discover?song=${x.id}`)} />
            ))}
            {loading && Array.from({ length: initialLoading ? 12 : 6 }, (_, i) => <SongCardSkeleton key={`sk${i}`} />)}
          </div>
        ) : (
          <div>
            <TrackListHeader sort={sort} order={order} onSort={onHeaderSort} />
            <div className="mt-2">
              {items.map((s, i) => <TrackRow key={s.id} song={s} index={i} />)}
              {loading && Array.from({ length: initialLoading ? 12 : 4 }, (_, i) => <TrackRowSkeleton key={`sk${i}`} />)}
            </div>
          </div>
        )}
        {hasMore && <div ref={sentinel} className="h-px" aria-hidden="true" />}
        {error && items.length > 0 && (
          <p className="mt-4 text-center text-sm text-muted">
            {error} <button type="button" className="underline" onClick={loadMore}>Retry</button>
          </p>
        )}
      </div>
    </div>
  );
}

function EmptyState({ icon, title, body, children }: { icon: React.ReactNode; title: string; body: string; children?: React.ReactNode }) {
  return (
    <div className="animate-fade-up rounded-2xl bg-surface px-6 py-16 text-center">
      <div className="mx-auto grid place-items-center text-subtle">{icon}</div>
      <h2 className="mt-3 text-xl font-bold">{title}</h2>
      <p className="mt-1 text-muted">{body}</p>
      {children}
    </div>
  );
}
