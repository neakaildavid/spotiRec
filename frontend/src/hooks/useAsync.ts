import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Run an async loader whenever `key` changes. The previous request is aborted,
 * so a slow response for an old key can never overwrite a newer one.
 * Pass `key = null` to skip loading.
 */
export function useAsync<T>(loader: (signal: AbortSignal) => Promise<T>, key: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(key !== null);
  const [attempt, setAttempt] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    if (key === null) {
      setData(null);
      setLoading(false);
      return;
    }
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    loaderRef
      .current(ctrl.signal)
      .then((d) => !ctrl.signal.aborted && setData(d))
      .catch((e: Error) => !ctrl.signal.aborted && e.name !== "AbortError" && setError(e.message))
      .finally(() => !ctrl.signal.aborted && setLoading(false));
    return () => ctrl.abort();
  }, [key, attempt]);

  const retry = useCallback(() => setAttempt((a) => a + 1), []);
  return { data, error, loading, retry };
}
