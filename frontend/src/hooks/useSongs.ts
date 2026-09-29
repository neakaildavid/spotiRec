import { useCallback, useEffect, useRef, useState } from "react";
import { listSongs, type Song, type SortField, type SortOrder } from "../lib/api";

const PAGE = 60;

/**
 * Server-side paginated song list. Changing the query/sort resets the list and
 * aborts any in-flight request, so a slow response for an old search can never
 * overwrite a newer one.
 */
export function useSongs(q: string, sort: SortField, order: SortOrder) {
  const [items, setItems] = useState<Song[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  const fetchPage = useCallback(
    async (offset: number) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setLoading(true);
      setError(null);
      try {
        const page = await listSongs({ q: q || undefined, sort, order, limit: PAGE, offset }, ctrl.signal);
        setItems((prev) => (offset === 0 ? page.items : [...prev, ...page.items]));
        setTotal(page.total);
      } catch (e) {
        if ((e as Error).name !== "AbortError") setError((e as Error).message);
      } finally {
        if (abortRef.current === ctrl) setLoading(false);
      }
    },
    [q, sort, order],
  );

  useEffect(() => {
    setItems([]);
    setTotal(null);
    void fetchPage(0);
    return () => abortRef.current?.abort();
  }, [fetchPage, reloadKey]);

  const hasMore = total !== null && items.length < total;
  const loadMore = useCallback(() => {
    if (!loading && hasMore) void fetchPage(items.length);
  }, [loading, hasMore, fetchPage, items.length]);
  const retry = useCallback(() => setReloadKey((k) => k + 1), []);

  return { items, total, loading, error, hasMore, loadMore, retry };
}

/** Debounce a fast-changing value (search box -> API query). */
export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}
