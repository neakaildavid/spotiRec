import { ArrowRight, CircleAlert, Search, Sparkles, X } from "lucide-react";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { CoverArt } from "../components/CoverArt";
import { GenreChip } from "../components/GenreChip";
import { SongCardSkeleton } from "../components/Skeleton";
import { SongCard } from "../components/SongCard";
import { useAsync } from "../hooks/useAsync";
import { discoverByText, getSong, listGenres, listSongs, similarSongs, type Song } from "../lib/api";
import { songArtist, songTitle } from "../lib/format";
import { usePlayer } from "../player/PlayerContext";

const PROMPTS = [
  "mellow acoustic guitar with soft vocals",
  "upbeat electronic dance music with a heavy bassline",
  "aggressive distorted guitars and pounding drums",
  "slow ambient drone",
  "hip-hop beat with rap vocals",
  "solo piano",
];

/**
 * Discover: find music by how it sounds.
 *  ?q=…      text search (CLAP text -> audio)
 *  ?song=ID  songs that sound like a library song
 *  ?genre=…  browse a genre
 * State lives in the URL so results are shareable and Back works.
 */
export function DiscoverPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q")?.trim() || "";
  const songId = Number(params.get("song")) || null;
  const genre = params.get("genre") || "";
  const [draft, setDraft] = useState(q);
  useEffect(() => setDraft(q), [q]);

  const genres = useAsync((s) => listGenres(s), "genres");
  const go = (next: Record<string, string>) => setParams(next);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (draft.trim()) go({ q: draft.trim() });
  };

  return (
    <div className="mx-auto max-w-7xl px-4 pt-6 pb-10 md:px-8 md:pt-10">
      <header>
        <h1 className="text-4xl font-black tracking-tight md:text-5xl">Discover</h1>
        <p className="mt-1 text-muted">Find music by how it sounds: describe it, or start from a song you like.</p>
      </header>

      <form onSubmit={submit} className="mt-6 flex gap-2" role="search">
        <label className="relative flex-1">
          <span className="sr-only">Describe a sound</span>
          <Sparkles className="pointer-events-none absolute top-1/2 left-4 size-5 -translate-y-1/2 text-accent" aria-hidden="true" />
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={200}
            placeholder="Describe a sound: mellow acoustic guitar, heavy bassline…"
            className="h-12 w-full rounded-full bg-elevated pr-11 pl-12 text-base placeholder:text-subtle outline-none transition-shadow duration-150 focus:ring-2 focus:ring-accent"
          />
          {draft && (
            <button type="button" onClick={() => { setDraft(""); go({}); }} aria-label="Clear"
              className="absolute top-1/2 right-3 grid size-7 -translate-y-1/2 place-items-center rounded-full text-subtle hover:text-fg">
              <X className="size-4" />
            </button>
          )}
        </label>
        <button type="submit" disabled={!draft.trim()}
          className="inline-flex h-12 items-center gap-2 rounded-full bg-accent px-5 font-bold text-accent-fg transition duration-150 hover:bg-accent-hover disabled:opacity-40">
          <Search className="size-4" aria-hidden="true" /><span className="hidden sm:inline">Search</span>
        </button>
      </form>

      {!q && !songId && (
        <div className="mt-4 flex flex-wrap gap-2" aria-label="Example searches">
          {PROMPTS.map((p) => (
            <button key={p} type="button" onClick={() => go({ q: p })}
              className="rounded-full border border-line px-3.5 py-1.5 text-sm text-muted transition-colors duration-150 hover:border-muted hover:text-fg">
              {p}
            </button>
          ))}
        </div>
      )}

      <div className="mt-8">
        <p className="mb-3 text-xs font-bold tracking-[0.2em] text-subtle uppercase">Browse by genre</p>
        <div className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-1 [scrollbar-width:none]">
          {(genres.data ?? []).map((g) => (
            <GenreChip key={g.genre} label={g.genre} active={genre === g.genre}
              onClick={() => go(genre === g.genre ? {} : { genre: g.genre })} />
          ))}
        </div>
      </div>

      <div className="mt-10">
        {songId ? <SimilarToSong id={songId} /> : q ? <TextResults q={q} /> : genre ? <GenreResults genre={genre} /> : <HowItWorks />}
      </div>
    </div>
  );
}

function ResultsGrid({ songs, loading, hint }: { songs: Song[] | undefined; loading: boolean; hint?: (s: Song) => string | undefined }) {
  const navigate = useNavigate();
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:gap-4 lg:grid-cols-4 xl:grid-cols-6">
      {loading
        ? Array.from({ length: 12 }, (_, i) => <SongCardSkeleton key={i} />)
        : songs?.map((s) => (
            <SongCard key={s.id} song={s} caption={s.genre ?? undefined} hint={hint?.(s)}
              onMoreLike={(x) => navigate(`/discover?song=${x.id}`)} />
          ))}
    </div>
  );
}

function ErrorNote({ message, retry }: { message: string; retry: () => void }) {
  return (
    <p className="flex items-center gap-2 rounded-lg bg-surface px-4 py-3 text-sm text-muted" role="alert">
      <CircleAlert className="size-4 shrink-0 text-accent" aria-hidden="true" /> {message}
      <button type="button" onClick={retry} className="ml-auto font-semibold text-fg underline">Retry</button>
    </p>
  );
}

function SectionTitle({ children, note }: { children: ReactNode; note?: ReactNode }) {
  return (
    <div className="mb-4">
      <h2 className="text-2xl font-black tracking-tight md:text-3xl">{children}</h2>
      {note && <p className="mt-1 text-sm text-muted">{note}</p>}
    </div>
  );
}

function TextResults({ q }: { q: string }) {
  const { data, error, loading, retry } = useAsync((s) => discoverByText(q, 24, s), `text:${q}`);
  const scores = new Map(data?.items.map((i) => [i.song.id, i.score]));
  return (
    <section className="animate-fade-up">
      <SectionTitle note={data && `Matched by meaning with CLAP, a model that embeds text and audio in one space. ${Math.round(data.embed_ms + data.search_ms)} ms.`}>
        “{q}”
      </SectionTitle>
      {error ? <ErrorNote message={error} retry={retry} /> : (
        <ResultsGrid songs={data?.items.map((i) => i.song)} loading={loading}
          hint={(s) => `Text-audio similarity ${scores.get(s.id)?.toFixed(3)}`} />
      )}
    </section>
  );
}

function SimilarToSong({ id }: { id: number }) {
  const player = usePlayer();
  const seed = useAsync((s) => getSong(id, s), `song:${id}`);
  const similar = useAsync((s) => similarSongs(id, 24, s), `similar:${id}`);
  const scores = new Map(similar.data?.items.map((i) => [i.song.id, i.score]));
  const song = seed.data;
  return (
    <section className="animate-fade-up">
      <div className="mb-6 flex items-center gap-4 rounded-2xl bg-surface p-4">
        {song ? <CoverArt songId={song.id} className="size-20 shrink-0 rounded-lg shadow-lg shadow-black/40" /> : <div className="skeleton size-20 shrink-0" />}
        <div className="min-w-0 flex-1">
          <p className="text-xs font-bold tracking-[0.2em] text-accent uppercase">Sounds like</p>
          <h2 className="truncate text-2xl font-black tracking-tight md:text-3xl">{song ? songTitle(song) : " "}</h2>
          <p className="truncate text-muted">{song ? `${songArtist(song)}${song.genre ? ` · ${song.genre}` : ""}` : " "}</p>
        </div>
        {song && (
          <button type="button" onClick={() => player.play(song)}
            className="hidden shrink-0 items-center gap-2 rounded-full border border-line px-4 py-2 text-sm font-semibold hover:border-muted sm:inline-flex">
            Play <ArrowRight className="size-4" aria-hidden="true" />
          </button>
        )}
      </div>
      {similar.error ? <ErrorNote message={similar.error} retry={similar.retry} /> : (
        <ResultsGrid songs={similar.data?.items.map((i) => i.song)} loading={similar.loading}
          hint={(s) => `Similarity ${scores.get(s.id)?.toFixed(3)}`} />
      )}
      <p className="mt-4 text-xs text-subtle">At most two songs per artist, so the list isn't just the rest of the album.</p>
    </section>
  );
}

function GenreResults({ genre }: { genre: string }) {
  const { data, error, loading, retry } = useAsync((s) => listSongs({ genre, limit: 48, sort: "id", order: "desc" }, s), `genre:${genre}`);
  return (
    <section className="animate-fade-up">
      <SectionTitle note={data && `${data.total.toLocaleString()} tracks`}>{genre}</SectionTitle>
      {error ? <ErrorNote message={error} retry={retry} /> : <ResultsGrid songs={data?.items} loading={loading} />}
    </section>
  );
}

function HowItWorks() {
  const steps = [
    ["Listen", "Every song is split into 10-second windows and run through CLAP, an audio model trained on music and descriptions."],
    ["Embed", "Each song becomes a 512-number vector. Songs that sound alike end up close together, and so do matching descriptions."],
    ["Search", "A vector index (HNSW in Postgres) finds the nearest songs in a couple of milliseconds."],
  ];
  return (
    <section className="grid gap-4 md:grid-cols-3" aria-label="How Discover works">
      {steps.map(([t, d], i) => (
        <div key={t} className="rounded-2xl bg-surface p-5">
          <p className="text-sm font-bold text-accent tabular-nums">0{i + 1}</p>
          <h3 className="mt-1 text-lg font-bold">{t}</h3>
          <p className="mt-1 text-sm text-muted">{d}</p>
        </div>
      ))}
    </section>
  );
}
